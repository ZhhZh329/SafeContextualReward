# -*- coding: utf-8 -*-
# src/data_construction/document_synthesis/toxic_document.py
# 目标：各数据集独立排序+选Top，份额按“每源可用候选占比”自动分配；不足则按剩余占比补位。
# 新增：max_samples_per_source —— 限制每源处理条数（用于快速打通 pipeline）
# 依赖：pip install datasets langdetect unidecode tqdm

import os, re, json, heapq, math
from dataclasses import dataclass, field
from typing import Iterable, Dict, Any, List, Tuple, Optional
from tqdm import tqdm
from datasets import load_dataset, Dataset
from langdetect import detect, DetectorFactory
from unidecode import unidecode

DetectorFactory.seed = 42

# ---------- 文本清洗 & 语言 ----------
def normalize_text(s: str) -> str:
    if s is None: return ""
    s = unidecode(str(s))
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()

def is_english(s: str) -> bool:
    try: return detect(s) == "en"
    except Exception: return True

# ---------- 各源 streaming 迭代器（输出 text + hint + source） ----------
def iter_civil_comments(debug=False, debug_n=200_000) -> Iterable[Dict[str, Any]]:
    ds = load_dataset("google/civil_comments", split="train", streaming=True)
    i = 0
    for ex in ds:
        if debug and i >= debug_n: break
        i += 1
        txt = normalize_text(ex.get("text") or ex.get("comment_text") or "")
        h = ex.get("toxicity")
        try: h = float(h) if h is not None else None
        except: h = None
        if h is None: continue
        yield {"text": txt, "hint": h, "source": "civil_comments"}

# 修复：Real Toxicity Prompts（分数在 prompt/continuation 字典中）
def iter_real_toxicity_prompts(debug=False, debug_n=150_000):
    ds = load_dataset("allenai/real-toxicity-prompts", split="train", streaming=True)
    i = 0
    for ex in ds:
        if debug and i >= debug_n: break
        i += 1
        cont = ex.get("continuation") or {}
        prompt = ex.get("prompt") or {}
        txt = normalize_text(cont.get("text") or prompt.get("text") or "")
        if not txt: continue

        def _max_attr(d):
            vals = []
            for k, v in d.items():
                if k == "text": continue
                try: vals.append(float(v))
                except: pass
            return max(vals) if vals else None

        h = _max_attr(cont) or _max_attr(prompt)
        if h is None: continue
        yield {"text": txt, "hint": h, "source": "real_toxicity_prompts"}

# 修复：HateXplain（用 parquet 镜像，支持 streaming）
def iter_hatexplain(debug=False, debug_n=120_000):
    ds = load_dataset("Abhi0072/HateXplain", split="train", streaming=True)  # ✅ parquet 镜像
    i = 0
    for ex in ds:
        if debug and i >= debug_n: break
        i += 1
        txt = normalize_text(ex.get("text") or "")
        lab = (ex.get("label") or "").lower().strip()
        if not txt: continue
        if "hate" in lab: h = 1.0
        elif "offen" in lab: h = 0.8
        else: continue  # 跳过 normal
        yield {"text": txt, "hint": h, "source": "hatexplain"}

# 替换你文件中的 iter_jigsaw_toxicity_pred 为以下实现
def iter_jigsaw_toxicity_pred(debug=False, debug_n=400_000):
    """
    使用 Jigsaw Kaggle 任务的社区镜像（已转 parquet，支持 streaming）：
      首选:  thesofakillers/jigsaw-toxic-comment-classification-challenge
      备  选: tcapelle/jigsaw-toxic-comment-classification-challenge
    字段：comment_text + 六个二分类标签(toxic, severe_toxic, obscene, threat, insult, identity_hate)
    我们将 hint = 这六个字段的最大值（>=1 视作高毒；若是 float 也能取 max）。
    """
    ds_id_candidates = [
        "thesofakillers/jigsaw-toxic-comment-classification-challenge",
        "tcapelle/jigsaw-toxic-comment-classification-challenge",
    ]
    last_err = None
    ds = None
    for ds_id in ds_id_candidates:
        try:
            # 先试 "train"；有些镜像只有 "default"
            try:
                ds = load_dataset(ds_id, split="train", streaming=True)
            except Exception:
                ds = load_dataset(ds_id, split="default", streaming=True)
            print(f"[INFO] using Jigsaw mirror: {ds_id}")
            break
        except Exception as e:
            last_err = e
            continue
    if ds is None:
        raise RuntimeError(f"Failed to load a parquet mirror for Jigsaw: {last_err}")

    KEYS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

    i = 0
    for ex in ds:
        if debug and i >= debug_n:
            break
        i += 1
        txt = normalize_text(ex.get("comment_text") or ex.get("text") or "")
        if not txt:
            continue
        vals = []
        for k in KEYS:
            v = ex.get(k)
            try:
                if v is not None:
                    vals.append(float(v))
            except Exception:
                pass
        if not vals:
            continue
        h = max(vals)
        yield {"text": txt, "hint": h, "source": "jigsaw_toxicity_pred"}


def iter_toxic_conversations(debug=False, debug_n=200_000):
    ds = load_dataset("SetFit/toxic_conversations", split="train", streaming=True)
    i = 0
    for ex in ds:
        if debug and i >= debug_n: break
        i += 1
        txt = normalize_text(ex.get("text") or "")
        lab = ex.get("label")
        try: h = float(lab)
        except: h = None
        if h is None or h <= 0.0: continue
        yield {"text": txt, "hint": h, "source": "toxic_conversations"}

SOURCES = {
    "civil_comments": iter_civil_comments,
    "real_toxicity_prompts": iter_real_toxicity_prompts,
    "hatexplain": iter_hatexplain,
    "jigsaw_toxicity_pred": iter_jigsaw_toxicity_pred,
    "toxic_conversations": iter_toxic_conversations,
}

# ---------- 配置 ----------
@dataclass
class AutoAllocCfg:
    index: str = "v4_rpj_llama_s4"
    run_tag: str = "tox_top_autoalloc_v1"
    out_root: str = "./results/document_synthesis"
    debug: bool = False

    keep_english_only: bool = True
    min_len: int = 5
    max_len_chars: Optional[int] = None   # 不截断；可设超大上限保险

    target_size: int = 100_000
    reserve_factor: float = 1.2           # 每源额外保留比例，用于补位
    checkpoint_every: int = 50_000

    # ✅ 新增：每源最多处理多少“已通过过滤”的样本（在 Pass1 & Pass2 都会生效）
    # 例如设为 5000，就能很快把整条 pipeline 跑通
    max_samples_per_source: Optional[int] = None

    # 每源“粗滤地板线”（减少无效入堆；不跨源比较）
    hint_floor_by_source: Dict[str, float] = field(default_factory=lambda: {
        "civil_comments": 0.5,
        "real_toxicity_prompts": 0.5,
        "hatexplain": 0.5,
        "jigsaw_toxicity_pred": 0.5,
        "toxic_conversations": 0.5,
    })

# ---------- 主逻辑 ----------
class AutoAllocTopKBuilder:
    def __init__(self, cfg: AutoAllocCfg):
        self.cfg = cfg
        self.out_dir = os.path.join(cfg.out_root, cfg.index, "toxicity_corpus", cfg.run_tag)
        os.makedirs(self.out_dir, exist_ok=True)
        self.tmp_dir = os.path.join(self.out_dir, "_tmp"); os.makedirs(self.tmp_dir, exist_ok=True)
        self.out_jsonl = os.path.join(self.out_dir, "toxic_topk.jsonl")
        self.out_parquet = os.path.join(self.out_dir, "toxic_topk.parquet")
        self.manifest = os.path.join(self.out_dir, "manifest.json")

    def _iter_filtered(self, source: str, limit: Optional[int] = None) -> Iterable[Dict[str,Any]]:
        """
        迭代某源的“已通过过滤”的样本。
        limit：最多产出多少条（用于快跑），None 则不限制。
        """
        it = SOURCES[source]
        floor = float(self.cfg.hint_floor_by_source.get(source, 0.0))
        yielded = 0
        for r in it(debug=self.cfg.debug):
            txt, h = r["text"], r["hint"]
            if not txt or len(txt) < self.cfg.min_len: continue
            if self.cfg.max_len_chars and len(txt) > self.cfg.max_len_chars:
                txt = txt[:self.cfg.max_len_chars]
            if self.cfg.keep_english_only and not is_english(txt): continue
            if h is None or float(h) < floor: continue
            r["text"] = txt
            yield r
            yielded += 1
            if limit is not None and yielded >= limit:
                break

    # 第一遍：只计数（每源可用候选数）——受 max_samples_per_source 限制
    def pass1_count(self) -> Dict[str, int]:
        counts = {}
        for src in SOURCES.keys():
            c = 0
            lim = self.cfg.max_samples_per_source
            pbar = tqdm(desc=f"{src}: count", dynamic_ncols=True, total=lim if lim else None)
            for _ in self._iter_filtered(src, limit=lim):
                c += 1
                pbar.update(1)
            pbar.close()
            counts[src] = c
        total = sum(counts.values())
        print("[INFO] available(per-source, limited)" if self.cfg.max_samples_per_source else "[INFO] available(per-source)",
              counts, " total=", total)
        return counts

    # 根据 counts 按占比分配每源 cap
    @staticmethod
    def _alloc_from_counts(counts: Dict[str, int], target: int) -> Dict[str, int]:
        total = sum(counts.values())
        if total == 0:
            return {k: 0 for k in counts}
        raw = {k: (counts[k] / total) * target for k in counts}
        caps = {k: int(math.floor(v)) for k, v in raw.items()}
        remain = target - sum(caps.values())
        frac = sorted(((raw[k] - caps[k], k) for k in counts), reverse=True)
        for _, k in frac[:max(0, remain)]:
            caps[k] += 1
        return caps

    def _checkpoint_heap(self, src: str, heap: List[Tuple[float,int,Dict[str,Any]]]):
        path = os.path.join(self.tmp_dir, f"{src}.jsonl")
        arr = sorted(heap, key=lambda x: x[0], reverse=True)
        with open(path, "w", encoding="utf-8") as f:
            for sc, _, r in arr:
                f.write(json.dumps({"text": r["text"], "source": r["source"], "hint": float(sc)},
                                   ensure_ascii=False) + "\n")

    # 第二遍：各源独立维护最小堆（容量 = cap * reserve_factor）——同样受 max_samples_per_source 限制
    def pass2_heaps(self, caps: Dict[str, int]) -> Dict[str, List[Tuple[float,int,Dict[str,Any]]]]:
        heaps, reserves = {}, {}
        uid = 0
        for src, cap in caps.items():
            heaps[src] = []
            res_cap = int(max(cap, math.ceil(cap * self.cfg.reserve_factor)))
            reserves[src] = res_cap

            lim = self.cfg.max_samples_per_source
            pbar = tqdm(desc=f"{src}: heap(cap={cap}, res={res_cap})", dynamic_ncols=True, total=lim if lim else None)
            seen_txt = set()
            since_ck = 0
            for rec in self._iter_filtered(src, limit=lim):
                uid += 1
                h = float(rec["hint"]); txt = rec["text"]
                if txt in seen_txt:
                    pbar.update(1); continue
                item = (h, uid, rec)
                if len(heaps[src]) < res_cap:
                    heapq.heappush(heaps[src], item); seen_txt.add(txt)
                else:
                    if h > heaps[src][0][0]:
                        _, _, old = heaps[src][0]
                        heapq.heapreplace(heaps[src], item)
                        seen_txt.discard(old["text"]); seen_txt.add(txt)
                since_ck += 1
                if since_ck >= self.cfg.checkpoint_every:
                    self._checkpoint_heap(src, heaps[src]); since_ck = 0
                pbar.update(1)
            pbar.close()
            self._checkpoint_heap(src, heaps[src])
        return heaps

    # 合并：先按 cap 取每源 Top-cap；若总量不足 target，用各源“剩余（extras）按剩余占比”补齐
    def merge_and_save(self, caps: Dict[str, int], heaps: Dict[str, List[Tuple[float,int,Dict[str,Any]]]]) -> Dict[str,str]:
        per_source_sorted = {src: sorted(heaps[src], key=lambda x: x[0], reverse=True) for src in caps}
        final_by_src: Dict[str, List[Tuple[float,Dict[str,Any]]]] = {}
        extras_by_src: Dict[str, List[Tuple[float,Dict[str,Any]]]] = {}
        total_target = int(self.cfg.target_size)

        for src in caps:
            arr = per_source_sorted[src]
            want = caps[src]
            take = arr[:want]
            extra = arr[want:]
            final_by_src[src]  = [(sc, r) for sc,_,r in take]
            extras_by_src[src] = [(sc, r) for sc,_,r in extra]

        got = sum(len(v) for v in final_by_src.values())
        remainder = max(0, total_target - got)
        while remainder > 0:
            extras_counts = {s: len(v) for s, v in extras_by_src.items() if len(v) > 0}
            total_extra = sum(extras_counts.values())
            if total_extra == 0: break
            raw = {s: remainder * (extras_counts[s] / total_extra) for s in extras_counts}
            add = {s: int(math.floor(v)) for s, v in raw.items()}
            rem2 = remainder - sum(add.values())
            frac = sorted(((raw[s] - add[s], s) for s in add), reverse=True)
            for _, s in frac[:max(0, rem2)]:
                add[s] += 1
            moved = 0
            for s, k in add.items():
                k = min(k, len(extras_by_src[s]))
                final_by_src[s].extend(extras_by_src[s][:k])
                extras_by_src[s] = extras_by_src[s][k:]
                moved += k
            if moved == 0: break
            remainder -= moved

        # 写出（不跨源再排序，避免不同尺度之间的隐式比较）
        with open(self.out_jsonl, "w", encoding="utf-8") as f:
            for src in caps:
                for sc, r in final_by_src[src]:
                    f.write(json.dumps({
                        "text": r["text"], "tox_score": float(sc), "source": src
                    }, ensure_ascii=False) + "\n")

        # 可选 parquet
        try:
            ds = Dataset.from_json(self.out_jsonl)
            ds.to_parquet(self.out_parquet)
        except Exception as e:
            print("[WARN] parquet export failed:", e)

        with open(self.manifest, "w", encoding="utf-8") as f:
            json.dump({
                "index": self.cfg.index,
                "run_tag": self.cfg.run_tag,
                "out_dir": os.path.abspath(self.out_dir),
                "auto_alloc_from_available": True,
                "reserve_factor": self.cfg.reserve_factor,
                "hint_floor_by_source": self.cfg.hint_floor_by_source,
                "cap_by_source": caps,
                "sizes": { "final_total": sum(len(v) for v in final_by_src.values()),
                           "per_source": {s: len(v) for s,v in final_by_src.items()} },
                "limits": { "max_samples_per_source": self.cfg.max_samples_per_source,
                            "target_size": self.cfg.target_size },
                "paths": {"jsonl": os.path.abspath(self.out_jsonl),
                          "parquet": os.path.abspath(self.out_parquet)}
            }, f, ensure_ascii=False, indent=2)

        # 清理临时
        for s in caps:
            p = os.path.join(self.tmp_dir, f"{s}.jsonl")
            if os.path.exists(p): os.remove(p)
        try: os.rmdir(self.tmp_dir)
        except OSError: pass

        print(f"[DONE] saved -> {self.out_jsonl}  &  {self.out_parquet}")
        return {"jsonl": self.out_jsonl, "parquet": self.out_parquet, "manifest": self.manifest}

    def run(self) -> Dict[str,str]:
        counts = self.pass1_count()
        caps = self._alloc_from_counts(counts, self.cfg.target_size)
        print("[INFO] caps (auto-allocated by availability):", caps, " sum=", sum(caps.values()))
        heaps = self.pass2_heaps(caps)
        return self.merge_and_save(caps, heaps)

# ---------- main ----------
def main():
    cfg = AutoAllocCfg(
        index="v4_rpj_llama_s4",
        run_tag="tox_top_autoalloc_v1",
        debug=False,
        keep_english_only=True, min_len=5, max_len_chars=None,
        target_size=200_000,                 # ← 小目标：全局只要 1k 方便验证
        reserve_factor=1.2,
        checkpoint_every=50_000,

        # ✅ 关键：每源最多处理这么多“已通过过滤”的样本，用于快速打通
        max_samples_per_source=None,

        hint_floor_by_source={
            "civil_comments": 0.5,
            "real_toxicity_prompts": 0.5,
            "hatexplain": 0.5,
            "jigsaw_toxicity_pred": 0.5,
            "toxic_conversations": 0.5,
        }
    )
    builder = AutoAllocTopKBuilder(cfg)
    builder.run()

if __name__ == "__main__":
    main()
