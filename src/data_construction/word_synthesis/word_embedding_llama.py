# word_embedding_llama_from_csv_env.py
# 依赖：pip install transformers torch numpy tqdm python-dotenv

import os, json, re, csv, glob
import numpy as np
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from typing import Optional, Iterable
from numpy.lib.format import open_memmap
from transformers import AutoTokenizer, AutoModel
import shutil

# ================= 配 置 =================
INDEX        = "v4_rpj_llama_s4"
MODEL_ID     = "meta-llama/Llama-3.2-1B-Instruct"   # 需HF访问权限
CACHE_DIR    = os.path.join("models", "language_models")
RESULTS_DIR  = os.path.join("results", "word_synthesis", INDEX)

# 输入 CSV（至少包含列名 'word'）
CSV_IN = "results/word_synthesis/test/union_all_words_wf100.csv"
# CSV_IN       = os.path.join(RESULTS_DIR, "union_top_words_10000_wf50000.csv")

# 计算哪些来源：["embedding"] / ["last"] / 两者都算
SOURCES      = ["embedding", "last"]
POOLING      = "mean"        # "mean" 或 "sum"
DEVICE_PREF  = "auto"        # "auto" | "cuda" | "mps" | "cpu"

BATCH_SIZE   = 32            # 批大小
SKIP_EXISTING = True         # 已存在则跳过
# =======================================

def resolve_device(prefer: str = "auto") -> str:
    prefer = prefer.lower()
    if prefer == "cuda" and torch.cuda.is_available(): return "cuda"
    if prefer == "mps" and torch.backends.mps.is_available(): return "mps"
    if prefer == "cpu": return "cpu"
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def resolve_dtype(device: str):
    return torch.float16 if device == "cuda" else torch.float32

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_")[:100] or "word"

def load_words_from_csv(path: str):
    words = []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            w = (row.get("word") or "").strip()
            if w:
                words.append(w)
    # 去重且保持顺序
    seen, uniq = set(), []
    for w in words:
        if w not in seen:
            uniq.append(w); seen.add(w)
    return uniq

@torch.inference_mode()
def batch_vectors(model, tokenizer, words, source: str, pooling: str, device: str):
    texts = [" " + w for w in words]  # 关键：前导空格≈▁
    enc = tokenizer(
        texts, add_special_tokens=False, return_tensors="pt",
        padding=True, truncation=False
    )
    input_ids = enc["input_ids"].to(device)      # [B, T]
    attn      = enc["attention_mask"].to(device) # [B, T]

    if source == "embedding":
        tok_vecs = model.get_input_embeddings()(input_ids)      # [B, T, D]
    elif source == "last":
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        tok_vecs = out.last_hidden_state                         # [B, T, D]
    else:
        raise ValueError("source must be 'embedding' or 'last'")

    mask = attn.unsqueeze(-1)  # [B, T, 1]
    if pooling == "sum":
        vecs = (tok_vecs * mask).sum(dim=1)                      # [B, D]
    elif pooling == "mean":
        denom = mask.sum(dim=1).clamp(min=1)                     # [B, 1]
        vecs = (tok_vecs * mask).sum(dim=1) / denom
    else:
        raise ValueError("pooling must be 'mean' or 'sum'")

    vecs = vecs.detach().to("cpu").numpy().astype(np.float32)    # (B, D)

    # 收集每个样本的非padding token
    tokens_per_sample = []
    lengths = attn.sum(dim=1).tolist()
    for i, L in enumerate(lengths):
        ids = input_ids[i, :L].to("cpu").tolist()
        tokens = tokenizer.convert_ids_to_tokens(ids)
        tokens_per_sample.append(tokens)

    metas = []
    for w, tokens, L in zip(words, tokens_per_sample, lengths):
        metas.append({
            "word": w,
            "tokens": tokens,
            "token_count": int(L),
            "source": source,
            "pooling": pooling,
        })
    return vecs, metas

def main():
    # 读取 .env
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = resolve_device(DEVICE_PREF)
    dtype  = resolve_dtype(device)

    print(f"[INFO] device={device}, dtype={dtype}")
    print(f"[INFO] model_id={MODEL_ID}")
    print(f"[INFO] cache_dir={os.path.abspath(CACHE_DIR)}")
    print(f"[INFO] results_dir={os.path.abspath(RESULTS_DIR)}")
    print(f"[INFO] csv_in={os.path.abspath(CSV_IN)}")
    print(f"[INFO] HF_TOKEN set: {'yes' if HF_TOKEN else 'no'}")

    words = load_words_from_csv(CSV_IN)
    if not words:
        raise RuntimeError(f"No words found in CSV: {CSV_IN}")
    print(f"[INFO] total words: {len(words):,}")

    # tokenizer / model + pad_token 修复
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, use_fast=True,
        add_bos_token=False, add_eos_token=False, token=HF_TOKEN
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, torch_dtype=dtype,
        low_cpu_mem_usage=True, token=HF_TOKEN
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device).eval()

    model_tag = sanitize(MODEL_ID)

    for source in SOURCES:
        subdir = os.path.join(RESULTS_DIR, "embeddings", model_tag, f"{source}_{POOLING}")
        chunk_dir = os.path.join(subdir, "chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        print(f"[INFO] writing chunks to: {chunk_dir}")

        for i in tqdm(range(0, len(words), BATCH_SIZE), desc=f"{source}", dynamic_ncols=True):
            batch_words = words[i:i+BATCH_SIZE]

            # 若都已存在且允许跳过，则 continue
            if SKIP_EXISTING:
                all_exist = True
                for w in batch_words:
                    base = f"embed_{sanitize(w)}"
                    if not (os.path.exists(os.path.join(chunk_dir, base + ".npy")) and
                            os.path.exists(os.path.join(chunk_dir, base + ".meta.json"))):
                        all_exist = False; break
                if all_exist:
                    continue

            vecs, metas = batch_vectors(model, tokenizer, batch_words, source, POOLING, device)
            for vec, meta in zip(vecs, metas):
                base = f"embed_{sanitize(meta['word'])}"
                np.save(os.path.join(chunk_dir, base + ".npy"), vec)
                meta_out = {
                    **meta,
                    "vector_dim": int(vec.shape[-1]),
                    "model_id": MODEL_ID,
                    "device": device,
                    "dtype": str(dtype),
                    "index": INDEX,
                }
                with open(os.path.join(chunk_dir, base + ".meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta_out, f, ensure_ascii=False, indent=2)

class WordEmbeddingMatrixRunner:

    def __init__(self,
                 index: str = "v4_rpj_llama_s4",
                 model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
                 csv_in: Optional[str] = None,
                 debug: bool = False,
                 word_limit_debug: int = 100):
        self.index = index
        self.model_id = model_id
        self.csv_in = csv_in
        self.debug = debug
        self.word_limit_debug = int(word_limit_debug)

        # 运行期派生
        self.base_dir = os.path.join(".", "results", "word_synthesis_debug", index) \
                        if debug else os.path.join(".", "results", "word_synthesis", index)

        # 运行后可读取
        self.model_tag = None  # 由 sanitize(MODEL_ID) 得到
        self.results_dir = None

    # ---------- 内部：设置模块级变量给 main() 使用 ----------
    def _apply_globals(self):
        # 使用文件内已定义的 sanitize()
        global INDEX, MODEL_ID, RESULTS_DIR, CSV_IN

        self.model_tag = sanitize(self.model_id)
        self.results_dir = self.base_dir  # main() 用 RESULTS_DIR = base_dir

        os.makedirs(self.results_dir, exist_ok=True)

        INDEX      = self.index
        MODEL_ID   = self.model_id
        RESULTS_DIR = self.results_dir

        if not self.csv_in:
            raise ValueError("csv_in 未指定：请传入 union 输出的 CSV 路径。")

        # debug: 生成一个只含前 N 个 word 的临时 CSV，避免修改原 CSV
        if self.debug:
            dbg_csv = os.path.join(self.results_dir, f"_debug_first_{self.word_limit_debug}.csv")
            if (not os.path.exists(dbg_csv)) or (os.path.getmtime(dbg_csv) < os.path.getmtime(self.csv_in)):
                self._make_debug_subset_csv(self.csv_in, dbg_csv, self.word_limit_debug)
            CSV_IN = dbg_csv
        else:
            CSV_IN = self.csv_in

    def _make_debug_subset_csv(self, src_csv: str, dst_csv: str, limit: int):
        words, seen = [], set()
        with open(src_csv, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for row in rd:
                w = (row.get("word") or "").strip()
                if w and w not in seen:
                    words.append(w); seen.add(w)
                if len(words) >= limit:
                    break
        os.makedirs(os.path.dirname(dst_csv), exist_ok=True)
        with open(dst_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["word"])
            for w in words:
                wr.writerow([w])

    # ---------- 外部接口：仅生成 chunks（直接走 main()） ----------
    def run_chunks(self):
        self._apply_globals()
        # 直接调用文件内已有的 main()
        main()

    # ---------- 外部接口：chunks -> matrix（每个 source 一套） ----------
    def run_all(self, build_matrix: bool = True) -> dict:
        self.run_chunks()
        if not build_matrix:
            return {}
        out = {}
        for source in SOURCES:  # 用模块内 SOURCES（比如 ["embedding","last"]）
            subdir = os.path.join(self.results_dir, "embeddings", sanitize(MODEL_ID), f"{source}_{POOLING}")
            chunk_dir = os.path.join(subdir, "chunks")
            matrix_dir = os.path.join(subdir, "matrix")
            os.makedirs(matrix_dir, exist_ok=True)
            self._build_matrix_from_chunks(chunk_dir, matrix_dir)
            out[source] = matrix_dir
        return out

    # ---------- 矩阵聚合实现 ----------
    def _scan_chunk_meta(self, chunk_dir: str):
        metas = glob.glob(os.path.join(chunk_dir, "embed_*.meta.json"))
        word_to_npy, dim = {}, None
        for mp in metas:
            try:
                meta = json.load(open(mp, "r"))
                w = meta.get("word")
                if not w: continue
                npy_path = mp[:-10] + ".npy"  # 去 ".meta.json"
                if not os.path.exists(npy_path): continue
                word_to_npy[w] = npy_path
                if dim is None:
                    dim = int(meta.get("vector_dim", 0)) or int(np.load(npy_path).shape[-1])
            except Exception:
                continue
        return word_to_npy, dim

    def _build_matrix_from_chunks(self, chunk_dir: str, matrix_dir: str):
        word_to_npy, dim = self._scan_chunk_meta(chunk_dir)
        if not word_to_npy:
            raise RuntimeError(f"No chunk files in {chunk_dir}")
        if dim is None:
            any_path = next(iter(word_to_npy.values()))
            dim = int(np.load(any_path).shape[-1])

        words = sorted(word_to_npy.keys())
        N = len(words)

        mat_path   = os.path.join(matrix_dir, "matrix.npy")
        norms_path = os.path.join(matrix_dir, "row_norms.npy")
        words_path = os.path.join(matrix_dir, "words.txt")
        meta_path  = os.path.join(matrix_dir, "meta.json")

        X = open_memmap(mat_path, mode="w+", dtype=np.float32, shape=(N, dim))
        norms = open_memmap(norms_path, mode="w+", dtype=np.float32, shape=(N,))

        for i, w in enumerate(words):
            v = np.load(word_to_npy[w]).astype(np.float32)
            X[i, :] = v
            norms[i] = np.linalg.norm(v) + 1e-12
            if i % 5000 == 0:
                X.flush(); norms.flush()

        X.flush(); norms.flush()
        with open(words_path, "w", encoding="utf-8") as f:
            for w in words: f.write(w + "\n")
        json.dump({"rows": N, "dim": dim}, open(meta_path, "w"))

    def cleanup_chunks_if_safe(self, sources: list[str] | None = None) -> list[str]:
        """
        验证 matrix 完整后，安全删除对应 source 的 chunks/ 目录。
        返回：已删除的 chunks 绝对路径列表。
        """
        removed = []
        sources = sources or list(SOURCES)  # 例如 ["embedding", "last"]
        for source in sources:
            subdir     = os.path.join(self.results_dir, "embeddings", sanitize(MODEL_ID), f"{source}_{POOLING}")
            chunk_dir  = os.path.join(subdir, "chunks")
            matrix_dir = os.path.join(subdir, "matrix")

            # 1) 检查 matrix 是否齐全
            mat_path   = os.path.join(matrix_dir, "matrix.npy")
            norms_path = os.path.join(matrix_dir, "row_norms.npy")
            words_path = os.path.join(matrix_dir, "words.txt")
            meta_path  = os.path.join(matrix_dir, "meta.json")
            ok = all(os.path.exists(p) for p in [mat_path, norms_path, words_path, meta_path])
            if not ok:
                print(f"[SKIP] matrix 不完整，跳过删除：{matrix_dir}")
                continue

            # 2) 做一次基本一致性校验
            try:
                meta  = json.load(open(meta_path, "r"))
                rows  = int(meta["rows"]); dim = int(meta["dim"])
                X     = np.load(mat_path, mmap_mode="r")
                norms = np.load(norms_path, mmap_mode="r")
                with open(words_path, "r", encoding="utf-8") as f:
                    words = [line.strip() for line in f if line.strip()]
                assert X.shape == (rows, dim)
                assert norms.shape == (rows,)
                assert len(words) == rows
            except Exception as e:
                print(f"[SKIP] matrix 校验失败（{e}），为安全起见不删除：{matrix_dir}")
                continue

            # 3) 删除 chunks
            if os.path.isdir(chunk_dir):
                shutil.rmtree(chunk_dir)
                removed.append(os.path.abspath(chunk_dir))
                print(f"[CLEANUP] removed chunks: {chunk_dir}")
            else:
                print(f"[INFO] no chunks to remove: {chunk_dir}")
        return removed

if __name__ == "__main__":
    with torch.inference_mode():
        main()
