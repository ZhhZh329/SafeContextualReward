# -*- coding: utf-8 -*-
# src/data_construction/document_synthesis/unsafe_document_retrieval.py
# 并发版 + 断点续跑 + 定期落盘 + 稳定 tqdm（单条总进度条，适合 screen）
import os, re, csv, json, time, threading, math, sys, requests
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from dotenv import load_dotenv
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

API = "https://api.infini-gram.io/"

def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_")[:80] or "x"

class RateLimiter:
    def __init__(self, qps: float):
        self.interval = 1.0 / float(qps)
        self.lock = threading.Lock()
        self.next_ts = 0.0
    def acquire(self):
        with self.lock:
            now = time.monotonic()
            if now < self.next_ts:
                time.sleep(self.next_ts - now)
                now = time.monotonic()
            self.next_ts = max(now, self.next_ts) + self.interval

class UnsafeDocumentRetrieval:
    """
    文档检索（两阶段） + 并发：
      - 简单：find -> get_doc_by_rank（带相同 query）
      - CNF：find_cnf -> get_doc_by_ptr（带相同 query）
    并发策略：
      - seeds 维度并发（seed_workers）
      - 每个 seed 的 get_doc 并发（get_workers），全局 RateLimiter 控总 QPS

    ✅ 特性：
      - 断点续跑：支持从已有 CSV/JSONL 继续（按 doc_ix 去重）
      - 定期落盘：边抓边写，断电/中断安全
      - 稳定进度条：仅一个总进度条（screen 内不会疯狂换行）
    """
    def __init__(
        self,
        index: str = "v4_rpj_llama_s4",
        out_root: str = os.path.join(".", "results", "document_synthesis"),
        qps: float = 6.0,
        retry: int = 5,
        per_seed_docs: int = 200,
        run_tag: Optional[str] = None,
        timeout_sec: int = 30,
        cnf_max_clause_freq: int = 50000,
        cnf_max_diff_tokens: int = 100,
        verbose: bool = False,
        seed_workers: int = 6,    # ← seeds 并发
        get_workers: int = 24,    # ← 每个 seed 内 get_doc 并发
        max_disp_len: int = 10000,# ← 单文档展示窗口（越大越长，但不是全文）
        flush_every: int = 200,   # ← 每写入多少条 flush 一次
        resume: bool = True,      # ← 断点续跑
    ):
        self.index = index
        self.out_root = out_root
        self.per_seed = int(per_seed_docs)
        self.retry = int(retry)
        self.timeout = int(timeout_sec)
        self.limiter = RateLimiter(qps=qps)
        self.run_tag = run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cnf_max_clause_freq = int(cnf_max_clause_freq)
        self.cnf_max_diff_tokens = int(cnf_max_diff_tokens)
        self.verbose = verbose
        self.seed_workers = int(seed_workers)
        self.get_workers = int(get_workers)
        self.max_disp_len = int(max_disp_len)
        self.flush_every = int(flush_every)
        self.resume = bool(resume)

        # HTTP 连接池（复用 TCP）
        self.session = requests.Session()

        # 输出目录
        self.base_dir = os.path.join(self.out_root, self.index, "raw", self.run_tag)
        os.makedirs(self.base_dir, exist_ok=True)
        self.csv_path   = os.path.join(self.base_dir, "unsafe_candidates.csv")
        self.jsonl_path = os.path.join(self.base_dir, "unsafe_candidates.jsonl")
        self.meta_path  = os.path.join(self.base_dir, "meta.json")
        self.state_path = os.path.join(self.base_dir, "state.json")  # 小状态，便于 resume

        # 写入锁 & 全局去重集合
        self.write_lock = threading.Lock()
        self.seen_doc_ix = set()  # 全局 doc_ix 去重（跨 seed）
        self.total_written = 0

        # 预加载已有进度（断点续跑）
        if self.resume:
            self._load_existing_progress()

    # ---------------- 断点续跑：加载已有 CSV/JSONL ----------------
    def _load_existing_progress(self):
        try:
            if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
                with open(self.csv_path, "r", encoding="utf-8") as f:
                    rd = csv.DictReader(f)
                    for row in rd:
                        if row.get("doc_ix"):
                            self.seen_doc_ix.add(str(row["doc_ix"]))
            elif os.path.exists(self.jsonl_path) and os.path.getsize(self.jsonl_path) > 0:
                with open(self.jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            j = json.loads(line)
                            if j.get("doc_ix"):
                                self.seen_doc_ix.add(str(j["doc_ix"]))
                        except Exception:
                            pass
        except Exception as e:
            print(f"[WARN] resume load failed: {e}")

    # ---------------- HTTP ----------------
    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_err = None
        for i in range(self.retry):
            try:
                self.limiter.acquire()
                r = self.session.post(API, json=payload, timeout=self.timeout)
                try:
                    r.raise_for_status()
                except requests.HTTPError as e:
                    body = r.text.strip()
                    raise requests.HTTPError(f"{e} | body={body[:500]}") from e
                j = r.json()
                if isinstance(j, dict) and j.get("error"):
                    raise RuntimeError(j["error"])
                return j
            except Exception as e:
                last_err = e
                time.sleep(min(2.0, 0.5 * (2 ** i)))
        raise RuntimeError(f"API failed after retries: {last_err}")

    @staticmethod
    def _is_cnf_query(q: str) -> bool:
        qs = q.strip()
        return (" AND " in qs) or (" OR " in qs)

    # ---- find / get_doc ----
    def _find_simple(self, query: str) -> Tuple[int, List[Tuple[int,int]]]:
        payload = {"index": self.index, "query_type": "find", "query": query}
        data = self._post(payload)
        cnt = int(data.get("cnt", 0))
        segs = data.get("segment_by_shard", [])
        out = []
        for t in segs:
            if not isinstance(t, (list, tuple)) or len(t) != 2:
                out.append((0, 0))
            else:
                lo, hi = int(t[0]), int(t[1])
                if hi < lo: lo, hi = hi, lo
                out.append((max(0, lo), max(0, hi)))
        if self.verbose:
            print(f"[DEBUG] simple cnt={cnt}, shards={len(out)}")
        return cnt, out

    def _get_doc_by_rank(self, shard: int, rank: int, query: str) -> Dict[str, Any]:
        payload = {
            "index": self.index, "query_type": "get_doc_by_rank",
            "s": int(shard), "rank": int(rank), "query": query,
            "max_disp_len": self.max_disp_len,
        }
        return self._post(payload)

    def _find_cnf(self, query: str) -> Tuple[int, List[List[int]]]:
        payload = {
            "index": self.index, "query_type": "find_cnf", "query": query,
            "max_clause_freq": self.cnf_max_clause_freq, "max_diff_tokens": self.cnf_max_diff_tokens,
        }
        data = self._post(payload)
        cnt = int(data.get("cnt", 0))
        ptrs = data.get("ptrs_by_shard", [])
        out = []
        for lst in ptrs:
            lst = lst if isinstance(lst, list) else []
            out.append([int(x) for x in lst])
        if self.verbose:
            print(f"[DEBUG] cnf cnt={cnt}, shards={len(out)}")
        return cnt, out

    def _get_doc_by_ptr(self, shard: int, ptr: int, query: str) -> Dict[str, Any]:
        payload = {
            "index": self.index, "query_type": "get_doc_by_ptr",
            "s": int(shard), "ptr": int(ptr), "query": query,
            "max_disp_len": self.max_disp_len,
        }
        return self._post(payload)

    # ---- 采样 ----
    @staticmethod
    def _evenly_sample_ranks(lo: int, hi: int, k: int) -> List[int]:
        length = max(0, hi - lo)
        if length <= 0 or k <= 0: return []
        k = min(k, length)
        if k == length: return list(range(lo, hi))
        step = length / float(k)
        return [lo + int(math.floor(i * step)) for i in range(k)]

    # ---- 批量并发 fetch（通用）----
    def _fetch_many(
        self,
        picks: List[Tuple[int,int]],              # [(shard, rank_or_ptr), ...]
        fetch_fn: Callable[[int,int,str], Dict], # _get_doc_by_rank 或 _get_doc_by_ptr
        query: str,
        accept_limit: int,
    ) -> List[Dict[str, Any]]:
        """并发发起 get_doc 请求；按 accept_limit 截断；返回文档列表（可能少于 picks）。"""
        docs, seen_doc_ix_local = [], set()
        # 给冗余 2x picks，抵抗去重/失败
        max_submit = min(len(picks), accept_limit * 2 if accept_limit > 0 else len(picks))
        picks = picks[:max_submit]

        with ThreadPoolExecutor(max_workers=self.get_workers) as ex:
            fut2meta = {ex.submit(fetch_fn, s, v, query): (s, v) for (s, v) in picks}
            for fut in as_completed(fut2meta):
                s, v = fut2meta[fut]
                try:
                    j = fut.result()
                except Exception:
                    continue
                doc_ix = str(j.get("doc_ix"))
                # 本 seed 内去重
                if doc_ix in seen_doc_ix_local:
                    continue
                seen_doc_ix_local.add(doc_ix)
                docs.append({
                    "shard": s, "ptr_or_rank": v,
                    "doc_ix": doc_ix, "doc_len": j.get("doc_len"),
                    "disp_len": j.get("disp_len"), "spans": j.get("spans"),
                })
                if 0 < accept_limit <= len(docs):
                    break
        return docs

    # ---- 单个 seed ----
    def _harvest_for_seed(self, seed: str) -> List[Dict[str, Any]]:
        q = (seed or "").strip()
        if not q: return []
        target = self.per_seed

        docs = []
        if self._is_cnf_query(q):
            _, ptrs_by_shard = self._find_cnf(q)
            S = len(ptrs_by_shard)
            if S > 0:
                per_shard = max(1, target // max(1, S))
                picks = []
                for s, ptrs in enumerate(ptrs_by_shard):
                    if not ptrs: continue
                    if len(ptrs) <= per_shard:
                        local = ptrs
                    else:
                        stride = max(1, len(ptrs)//per_shard)
                        local = ptrs[::stride][:per_shard]
                    picks.extend((s, p) for p in local)
                docs = self._fetch_many(picks, self._get_doc_by_ptr, q, accept_limit=target)
        else:
            _, segs = self._find_simple(q)
            S = len(segs)
            if S > 0:
                per_shard = max(1, target // max(1, S))
                picks = []
                for s, (lo, hi) in enumerate(segs):
                    if hi <= lo: continue
                    local = self._evenly_sample_ranks(lo, hi, per_shard)
                    picks.extend((s, r) for r in local)
                docs = self._fetch_many(picks, self._get_doc_by_rank, q, accept_limit=target)

        # 附加元数据
        for d in docs:
            d["seed"] = seed
            d["query"] = q
            d["mode"] = "cnf" if self._is_cnf_query(q) else "simple"
        return docs

    # ---- 统一写入（线程安全 + 定期 flush）----
    def _append_rows(self, rows: List[Dict[str, Any]], wr_csv, fh_jsonl):
        if not rows: return 0
        wrote = 0
        with self.write_lock:
            for r in rows:
                doc_ix = str(r["doc_ix"])
                if doc_ix in self.seen_doc_ix:
                    continue
                self.seen_doc_ix.add(doc_ix)

                wr_csv.writerow([
                    r["seed"], r["mode"], r["shard"], r["ptr_or_rank"],
                    r["doc_ix"], r["doc_len"], r["disp_len"],
                    json.dumps(r.get("spans"), ensure_ascii=False),
                    r["query"],
                ])
                fh_jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")
                wrote += 1
                self.total_written += 1

                if self.total_written % self.flush_every == 0:
                    try:
                        fh_jsonl.flush()
                        # 以防万一，fsync 确保落盘
                        os.fsync(fh_jsonl.fileno())
                    except Exception:
                        pass
        return wrote

    # ---- 多 seed 并发 + 主落盘 ----
    def run(self, seeds: List[str]) -> Dict[str, str]:
        meta = {
            "index": self.index,
            "run_tag": self.run_tag,
            "per_seed_docs": self.per_seed,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "api": API,
            "notes": "two-step doc search with concurrency; global QPS limited; checkpointing enabled",
            "seed_workers": self.seed_workers,
            "get_workers": self.get_workers,
            "max_disp_len": self.max_disp_len,
            "resume": self.resume,
            "flush_every": self.flush_every,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 打开 CSV/JSONL（追加模式；无文件则写 header）
        need_header = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
        fcsv = open(self.csv_path, "a", newline="", encoding="utf-8")
        wr = csv.writer(fcsv)
        if need_header:
            wr.writerow(["seed","mode","shard","ptr_or_rank","doc_ix","doc_len","disp_len","spans_json","query"])
            fcsv.flush()
        fjl = open(self.jsonl_path, "a", encoding="utf-8")

        # 单条总进度条（screen 下更稳，不开多重进度条）
        seeds_total = len(seeds)
        with ThreadPoolExecutor(max_workers=self.seed_workers) as ex, \
             tqdm(total=seeds_total, desc="seeds", ncols=100, ascii=True,
                  dynamic_ncols=False, mininterval=0.5, file=sys.stdout,
                  bar_format="{desc:<10}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]  rows={postfix}") as pbar:

            futs = {ex.submit(self._harvest_for_seed, sd): sd for sd in seeds}
            for fut in as_completed(futs):
                sd = futs[fut]
                try:
                    rows = fut.result()
                except Exception as e:
                    print(f"[WARN] seed='{sd}' failed: {e}")
                    rows = []
                wrote = self._append_rows(rows, wr, fjl)
                # flush 每个 seed 结束后再强制一次
                try:
                    fcsv.flush(); os.fsync(fcsv.fileno())
                    fjl.flush();  os.fsync(fjl.fileno())
                    # 保存 state
                    json.dump({"total_written": self.total_written}, open(self.state_path, "w"))
                except Exception:
                    pass
                pbar.set_postfix_str(f"{self.total_written}")
                pbar.update(1)

        fcsv.close()
        fjl.close()

        print(f"[INFO] saved {self.total_written} rows to:\n - {self.csv_path}\n - {self.jsonl_path}\n - meta: {self.meta_path}")
        return {
            "csv": os.path.abspath(self.csv_path),
            "jsonl": os.path.abspath(self.jsonl_path),
            "meta": os.path.abspath(self.meta_path),
            "out_dir": os.path.abspath(self.base_dir),
        }


# ---------------- Models ----------------
def _resolve_device():
    import torch
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"

def _dtype_for(device: str):
    import torch
    return torch.float16 if device == "cuda" else torch.float32

def _sanitize_tag(s: str) -> str:
    import re
    s = s.replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")
    return (s[:120] or "x")

def spans_to_text(spans, keep_newlines=True) -> str:
    """spans: [[span_text, clause_idx_or_null], ...] -> 拼接纯文本"""
    if not isinstance(spans, list): return ""
    parts = []
    for it in spans:
        if isinstance(it, (list, tuple)) and it:
            parts.append(str(it[0]))
        elif isinstance(it, dict) and "0" in it:  # 备用：某些序列化格式
            parts.append(str(it["0"]))
    text = "".join(parts)
    if not keep_newlines:
        text = re.sub(r"\s+", " ", text).strip()
    return text

def _safe_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        # 少数情况下 CSV 转义导致双引号异常，尝试修复
        return json.loads(s.replace("''", '"').replace('""', '"'))

class LlamaGuardClassifier:
    """
    Llama Guard 推理（SAFE / UNSAFE）
    - 支持 CUDA 量化(4/8bit)，Mac 用 mps，或 CPU
    - 避免 chat_template 报 alternation 的鲁棒渲染：
        * 先试 apply_chat_template
        * 失败则退回手写 prompt（不依赖模板）
    """
    def __init__(self,
                 model_id: str = "meta-llama/Llama-Guard-3-8B",
                 cache_dir: str = "./models/language_models",
                 device: Optional[str] = None,
                 quantize: str = "none",      # "none" | "8bit" | "4bit"（仅 CUDA）
                 max_input_tokens: int = 4096,
                 batch_size: int = 2):
        import os, torch
        from dotenv import load_dotenv
        load_dotenv()

        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = device or _resolve_device()
        self.dtype = _dtype_for(self.device)
        self.max_input_tokens = int(max_input_tokens)
        self.batch_size = int(batch_size)
        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

        # tokenizer
        self.tok = AutoTokenizer.from_pretrained(
            model_id, cache_dir=cache_dir, use_fast=True, token=self.hf_token
        )
        if self.tok.pad_token_id is None:
            # 少数模板没有 pad，兜底用 eos
            self.tok.pad_token = self.tok.eos_token

        # model（CUDA 可量化；mps/cpu 常规加载）
        can_quant = (self.device == "cuda") and _HAS_BNB and (quantize.lower() in ("4bit", "8bit"))
        if can_quant:
            if quantize.lower() == "4bit":
                qcfg = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                )
            else:
                qcfg = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, cache_dir=cache_dir, quantization_config=qcfg,
                device_map="auto", low_cpu_mem_usage=True, token=self.hf_token,
            ).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, cache_dir=cache_dir, dtype=self.dtype,
                low_cpu_mem_usage=True, token=self.hf_token,
            ).to(self.device).eval()

        try:
            self.model.config.pad_token_id = self.tok.pad_token_id
        except Exception:
            pass

        # 极简 system 提示（需要可以替换为官方 policy）
        self.system_prompt = (
            "You are Llama Guard 3, a safety classifier. "
            "Determine whether the USER content contains unsafe or harmful content "
            "(violence, sexual exploitation, hate, self-harm, illegal activities). "
            "Reply with a single word: SAFE or UNSAFE."
        )

    # ---- 鲁棒渲染：优先 chat_template，失败退回手写 prompt ----
    def _render_prompt(self, user_text: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_text}
        ]
        try:
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # 回退：不依赖模板，纯文本 prompt（对 1B/8B 都兼容）
            return (
                f"{self.system_prompt}\n\n"
                f"User: {user_text}\n"
                f"Assistant:"
            )

    @torch.inference_mode()
    def predict(self, texts: List[str]) -> List[bool]:
        outs = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            # 先按 token 上限在“读阶段”截断（外层也会截一次，再保守一次）
            batch = []
            for t in texts[i:i+self.batch_size]:
                enc = self.tok(t, add_special_tokens=False)
                if len(enc["input_ids"]) > self.max_input_tokens:
                    t = self.tok.decode(enc["input_ids"][:self.max_input_tokens], skip_special_tokens=True)
                batch.append(t)

            prompts = [self._render_prompt(t) for t in batch]
            enc = self.tok(prompts, padding=True, truncation=True,
                           max_length=self.max_input_tokens + 64, return_tensors="pt")
            enc = {k: v.to(self.model.device) for k, v in enc.items()}

            gen = self.model.generate(
                **enc, max_new_tokens=2, do_sample=False, num_beams=1,
                eos_token_id=self.tok.eos_token_id, use_cache=False
            )
            for j in range(gen.size(0)):
                full = gen[j]
                prompt_len = enc["input_ids"][j].size(0)
                cont = full[prompt_len:]
                txt = self.tok.decode(cont, skip_special_tokens=True).strip().upper()
                # 容错：默认 SAFE；若含 UNSAFE 则判不安全
                is_unsafe = ("UNSAFE" in txt) and ("SAFE" not in txt.replace("UNSAFE", ""))
                outs.append(bool(is_unsafe))
        return outs


class DetoxifyClassifier:
    """
    小而快的毒性分类器（Roberta-base）；阈值：任一类别 >= 0.5 => UNSAFE
    """
    def __init__(self,
                 model_id: str = "unitary/unbiased-toxic-roberta",
                 cache_dir: str = "./models/language_models",
                 device: Optional[str] = None,
                 max_input_tokens: int = 2048,
                 batch_size: int = 16):
        import torch
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = device or _resolve_device()
        self.max_input_tokens = int(max_input_tokens)
        self.batch_size = int(batch_size)

        self.tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, cache_dir=cache_dir
        ).to(self.device).eval()

    @torch.inference_mode()
    def predict(self, texts: List[str]) -> List[bool]:
        import torch
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            enc = self.tok(
                batch, padding=True, truncation=True, max_length=self.max_input_tokens, return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            if logits.size(-1) > 1:
                probs = torch.sigmoid(logits)
                unsafe = (probs >= 0.5).any(dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                unsafe = probs[:, -1] >= 0.5
            out.extend(unsafe.bool().tolist())
        return out


# --------------- Screening Orchestrator ---------------
@dataclass
class ScreeningConfig:
    input_csv: str
    backend: str = "llamaguard"  # "llamaguard" | "detoxify"
    index: Optional[str] = None
    run_tag: Optional[str] = None
    out_root: str = "./results/document_synthesis"
    keep_newlines: bool = True

def spans_to_text(spans, keep_newlines=True) -> str:
    """spans: [[span_text, clause_idx_or_null], ...] -> 拼接纯文本"""
    import re
    if not isinstance(spans, list): return ""
    parts = []
    for it in spans:
        if isinstance(it, (list, tuple)) and it:
            parts.append(str(it[0]))
        elif isinstance(it, dict) and "0" in it:  # 备用
            parts.append(str(it["0"]))
    text = "".join(parts)
    if not keep_newlines:
        text = re.sub(r"\s+", " ", text).strip()
    return text

def _safe_json_load(s: str):
    import json
    try:
        return json.loads(s)
    except Exception:
        return json.loads(s.replace("''", '"').replace('""', '"'))


# --- 替换 UnsafeScreening 为“严格分块 + 流式写盘 + 读时截断”的实现 ---
class UnsafeScreening:
    """
    读取 unsafe_candidates.csv → 流式输出 screened_<backend>_<model_tag>.csv
    - 只对 is_all_doc=True 的文本跑模型；其余 is_unsafe 留空
    - **严格分块**：chunk_size 行为一批；读→截断→推理→立刻写盘→释放
    - 对于 LlamaGuard/Detoxify，读时按各自 tokenizer 的 max_input_tokens 先截断
    - 每批 fsync，崩了也不丢进度
    """
    def __init__(self, cfg: ScreeningConfig,
                 cache_dir: str = "./models/language_models",
                 device: Optional[str] = None,
                 backend_kwargs: Optional[dict] = None):
        import os
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.device = device or _resolve_device()
        self.backend_kwargs = backend_kwargs or {}

        # 从路径推断 index/run_tag
        parts = os.path.normpath(cfg.input_csv).split(os.sep)
        try:
            idx_raw = parts.index("document_synthesis")
            index = cfg.index or parts[idx_raw + 1]
            run_tag = cfg.run_tag or parts[idx_raw + 3]
        except Exception:
            index = cfg.index or "unknown_index"
            run_tag = cfg.run_tag or "run"
        self.index = index
        self.run_tag = run_tag

        # 初始化后端
        if cfg.backend == "llamaguard":
            self.clf = LlamaGuardClassifier(cache_dir=self.cache_dir,
                                            device=self.device, **self.backend_kwargs)
            self.backend_name = "llamaguard"
        elif cfg.backend == "detoxify":
            self.clf = DetoxifyClassifier(cache_dir=self.cache_dir,
                                          device=self.device, **self.backend_kwargs)
            self.backend_name = "detoxify"
        else:
            raise ValueError("backend must be 'llamaguard' or 'detoxify'")

        self.model_id = getattr(self.clf, "model_id", "unknown_model")
        self.model_tag = _sanitize_tag(self.model_id)

        # 输出路径
        self.out_dir = os.path.join(cfg.out_root, self.index, "screened", self.run_tag)
        os.makedirs(self.out_dir, exist_ok=True)
        self.out_csv = os.path.join(self.out_dir, f"screened_{self.backend_name}_{self.model_tag}.csv")

    # —— 只保留截断后的文本，避免把超长原文堆在内存里 —— #
    def _truncate_for_backend(self, text: str) -> str:
        # LlamaGuard/Detoxify 都有 tok + max_input_tokens
        tok = getattr(self.clf, "tok", None)
        max_toks = getattr(self.clf, "max_input_tokens", None)
        if tok is not None and max_toks:
            enc = tok(text, add_special_tokens=False)
            if len(enc["input_ids"]) > max_toks:
                cut = enc["input_ids"][:max_toks]
                return tok.decode(cut, skip_special_tokens=True)
        # 兜底：字符上限，防极端长文撑爆内存（可微调）
        if len(text) > 20000:
            return text[:20000]
        return text

    def run_streaming(self, chunk_size: int = 256, flush_every: int = 128) -> str:
        """
        超大数据集友好：逐行读取 → 对全文行(is_all_doc=True)做“读时截断”并缓存在一个小批里
        小批达到 chunk_size 就推理，然后**立即写盘并 fsync**，再清空缓存。
        非全文行直接写盘（不进缓存、不推理）。
        """
        import csv, os, json, gc
        from tqdm import tqdm

        need_header = not os.path.exists(self.out_csv) or os.path.getsize(self.out_csv) == 0
        fout = open(self.out_csv, "a", newline="", encoding="utf-8")
        wr = csv.writer(fout)
        if need_header:
            wr.writerow(["word", "text", "is_all_doc", "is_unsafe"])
            fout.flush(); os.fsync(fout.fileno())

        # 计数器
        total_in = total_scored = total_written = 0
        # 缓存：只放“截断后的文本”
        buf_words, buf_texts = [], []

        # 逐行读取 CSV（真正 streaming，不会把整表装内存）
        with open(self.cfg.input_csv, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            pbar = tqdm(desc="screening(stream)", unit="row", dynamic_ncols=True)
            for r in rd:
                total_in += 1
                word = (r.get("seed") or "").strip()
                doc_len = int(r.get("doc_len") or 0)
                disp_len = int(r.get("disp_len") or 0)
                spans_raw = r.get("spans_json") or "[]"

                # 转成纯文本（临时字符串；用完即丢）
                spans = _safe_json_load(spans_raw)
                text = spans_to_text(spans, keep_newlines=self.cfg.keep_newlines)
                is_all = (doc_len == disp_len) and (disp_len <= 4096)

                if is_all and text.strip():
                    # 读时按模型上限先截断，仅保存截断后的短文本
                    t_short = self._truncate_for_backend(text)
                    buf_words.append(word)
                    buf_texts.append(t_short)
                    # 达到一个 chunk：推理 + 写盘 + 清空
                    if len(buf_texts) >= chunk_size:
                        preds = self.clf.predict(buf_texts)
                        for w, t, p in zip(buf_words, buf_texts, preds):
                            wr.writerow([w, t, 1, bool(p)])
                            total_written += 1
                        total_scored += len(buf_texts)
                        buf_words.clear(); buf_texts.clear()
                        if (total_written % flush_every) == 0:
                            fout.flush(); os.fsync(fout.fileno())
                            gc.collect()
                else:
                    # 非全文：直接写（不推理、不缓存，内存 footprint 最小）
                    wr.writerow([word, text, int(is_all), ""])
                    total_written += 1
                    if (total_written % flush_every) == 0:
                        fout.flush(); os.fsync(fout.fileno())

                if (total_in % 2000) == 0:
                    pbar.set_postfix({"in": total_in, "scored": total_scored, "written": total_written})
                pbar.update(1)

            pbar.close()

        # 收尾：把最后一小撮也跑完
        if buf_texts:
            preds = self.clf.predict(buf_texts)
            for w, t, p in zip(buf_words, buf_texts, preds):
                wr.writerow([w, t, 1, bool(p)])
                total_written += 1
            total_scored += len(buf_texts)
            buf_words.clear(); buf_texts.clear()

        fout.flush(); os.fsync(fout.fileno()); fout.close()

        print(f"[INFO] screened(stream) -> {self.out_csv} "
              f"(backend={self.backend_name}, model={self.model_id}, device={getattr(self.clf.model, 'device', self.device)}) "
              f"[in={total_in}, scored={total_scored}, written={total_written}]")
        return self.out_csv

    # 兼容旧接口：直接走 streaming
    def run(self) -> str:
        return self.run_streaming()


# ----------------- 示例 main -----------------
def main():
    INDEX = "v4_rpj_llama_s4"
    SEEDS = [
        # Violence / physical harm
        "murder", "homicide", "killing", "assault", "battery",
        "shooting", "mass shooting", "school shooting", "stabbing", "knife attack",
        "kidnapping", "abduction", "strangulation", "domestic violence", "child abuse",
        "human trafficking", "torture", "war crime", "ethnic cleansing", "genocide",
        "lynching", "honor killing", "arson", "acid attack", "bombing",
        "bomb threat", "explosive", "explosive device", "improvised explosive device", "IED",
        "terrorism", "terrorist attack", "extremism", "extremist propaganda", "violence",
        "violent threat", "threats", "death threat", "poisoning", "bomb plot",

        # Sexual misconduct
        "sexual assault", "rape", "attempted rape", "molestation", "sexual harassment",
        "sexual coercion", "sexual exploitation", "nonconsensual", "date rape", "sextortion",
        "revenge porn", "child exploitation", "grooming", "child grooming", "spiking drinks",

        # Harassment / hate
        "stalking", "harassment", "bullying", "cyberbullying", "doxxing",
        "doxing", "swatting", "intimidation", "blackmail", "extortion",
        "hate speech", "hate crime", "incitement to violence", "dehumanizing language", "racial slur",
        "homophobic slur", "transphobic abuse", "misogynistic abuse", "xenophobic rhetoric", "antisemitic",
        "islamophobic", "ableist slur", "incitement to hatred", "threatening message", "cyberstalking",

        # Self-harm
        "suicide", "suicidal ideation", "self harm", "self injury", "cutting",
        "encouraging self harm", "overdose",

        # Cyberharm / crime
        "ransomware", "malware", "spyware", "phishing", "identity theft",
        "account takeover", "data breach",

        # Weapons / trafficking / drugs
        "weapon trafficking", "illegal firearm", "ghost gun",
        "drug trafficking", "narcotics distribution", "illegal drug sale"
    ]

    """retr = UnsafeDocumentRetrieval(
        index=INDEX,
        qps=7.0, retry=5,
        per_seed_docs=2000,           # 想更多就调大
        run_tag="seed_v1",
        seed_workers=8,
        get_workers=32,
        max_disp_len=10000,           # 展示窗口更长
        flush_every=200,              # 定期 flush
        resume=True                   # 断点续跑
    )"""
    # outs = retr.run(SEEDS)
    # output_path = outs['csv']
    output_path = "/Users/zhzhou/Desktop/SafeContextualReward/results/document_synthesis/v4_rpj_llama_s4/raw/seed_v1/unsafe_candidates.csv"

    # A) 用 Llama Guard 3（推荐做“安全初筛”；需要 HF token & 权限）
    cfg = ScreeningConfig(input_csv=output_path, backend="llamaguard")
    screen = UnsafeScreening(
        cfg,
        device="mps",  # Mac 上用 mps；NVIDIA 就写 "cuda"
        backend_kwargs=dict(
            model_id="meta-llama/Llama-Guard-3-1B",
            quantize="none",          # mps 无量化；cuda 可 "4bit"/"8bit"
            batch_size=2,
            max_input_tokens=4096     # 再保守一点，减内存
        ),
    )
    out_csv = screen.run_streaming(chunk_size=300, flush_every=100)
    print("[OK]", out_csv)

    # B) 备选：Detoxify 小模型（更快；无权限要求）
    screen_fast = UnsafeScreening(
        ScreeningConfig(input_csv=output_path, backend="detoxify"),
        device="mps",                         # 任意
        backend_kwargs=dict(batch_size=64),
    )
    out_csv_fast = screen_fast.run_streaming(chunk_size=1000, flush_every=500)
    print("[OK]", out_csv_fast)

if __name__ == "__main__":
    main()
