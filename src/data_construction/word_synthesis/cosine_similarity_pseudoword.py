# cosine_engine_dual.py
# 依赖：pip install transformers torch numpy python-dotenv
import os, re, csv, json, glob, shutil
import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from numpy.lib.format import open_memmap
from typing import Optional, Iterable

def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_")[:100] or "x"

def _resolve_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def _resolve_dtype(device: str):
    return torch.float16 if device == "cuda" else torch.float32

class CosineEngineDual:
    """
    复用矩阵的“双路”(embedding & last) 余弦相似度引擎。
    目录结构（每路各一套）：
      results/word_synthesis/{index}/embeddings/{model_tag}/{source}_{pooling}/
        ├─ chunks/            # 小文件（可选清理）
        └─ matrix/
            ├─ matrix.npy      # [N, D] float32（标准 .npy，可 mmap）
            ├─ row_norms.npy   # [N] 每行范数
            ├─ words.txt       # 行→词
            └─ meta.json       # {"rows":N,"dim":D}
    """
    def __init__(
        self,
        index: str = "v4_rpj_llama_s4",
        model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
        pooling: str = "mean",                 # "mean" | "sum"
        base_results_dir: str = os.path.join("results", "word_synthesis"),
        cache_dir: str = os.path.join("models", "language_models"),
        device: str | None = None,
        block: int = 20000,                    # 分块大小（矩阵乘时）
    ):
        assert pooling in ("mean", "sum")
        load_dotenv()
        self.index      = index
        self.model_id   = model_id
        self.model_tag  = _sanitize(model_id)
        self.pooling    = pooling
        self.cache_dir  = cache_dir
        self.base_root  = os.path.join(base_results_dir, index, "embeddings", self.model_tag)
        self.device     = device or _resolve_device()
        self.dtype      = _resolve_dtype(self.device)
        self.hf_token   = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        self.block      = int(block)

        # 两条路：embedding / last
        self.sources = ("embedding", "last")
        # 懒加载资源
        self._tokenizer = None
        self._model     = None
        # 每路的数据结构
        self._mat   = {s: None for s in self.sources}   # memmap matrix
        self._norms = {s: None for s in self.sources}   # memmap norms
        self._words = {s: None for s in self.sources}   # list[str]
        self._rows  = {s: None for s in self.sources}
        self._dim   = {s: None for s in self.sources}

    # ---------- 路径工具 ----------
    def _base_dir(self, source: str):
        return os.path.join(self.base_root, f"{source}_{self.pooling}")
    def _chunk_dir(self, source: str):
        return os.path.join(self._base_dir(source), "chunks")
    def _matrix_dir(self, source: str):
        return os.path.join(self._base_dir(source), "matrix")
    def _matrix_paths(self, source: str):
        md = self._matrix_dir(source)
        return (
            os.path.join(md, "matrix.npy"),
            os.path.join(md, "row_norms.npy"),
            os.path.join(md, "words.txt"),
            os.path.join(md, "meta.json"),
        )

    # ---------- 模型与tokenizer ----------
    def _ensure_model(self):
        if self._tokenizer is not None and self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, cache_dir=self.cache_dir, use_fast=True,
            add_bos_token=False, add_eos_token=False, token=self.hf_token
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModel.from_pretrained(
            self.model_id, cache_dir=self.cache_dir,
            torch_dtype=self.dtype, low_cpu_mem_usage=True, token=self.hf_token
        ).to(self.device).eval()
        self._model.config.pad_token_id = self._tokenizer.pad_token_id

    # ---------- 整向量：返回每一路（embedding/last）的 “全量余弦向量” + “行范数” ----------
    def cosine_full_for_word(self, pseudo_word: str) -> dict[str, dict[str, np.ndarray]]:
        """
        计算：对矩阵里所有真实词，给定伪词的余弦相似度（不排序）。
        返回：
          {
            "embedding": {"cos": (N,), "norms": (N,)},
            "last":      {"cos": (N,), "norms": (N,)}
          }
        """
        vecs = self._encode_both(pseudo_word)
        out: dict[str, dict[str, np.ndarray]] = {}
        for source in self.sources:
            self._load_matrix_source(source)
            v = vecs[source]
            v_norm = v / (np.linalg.norm(v) + 1e-12)

            X, norms, rows = self._mat[source], self._norms[source], self._rows[source]
            sims = np.empty(rows, dtype=np.float32)
            # 分块计算 dot / ||x||
            for i in range(0, rows, self.block):
                j = min(i + self.block, rows)
                dot = X[i:j].dot(v_norm)
                sims[i:j] = dot / (norms[i:j] + 1e-12)

            out[source] = {
                "cos":   sims,                           # (N,)
                "norms": np.asarray(norms, dtype=np.float32),  # (N,)
            }
        return out


    # ---------- 从 chunks 聚合矩阵 ----------
    @staticmethod
    def _scan_chunk_meta(chunk_dir: str):
        metas = glob.glob(os.path.join(chunk_dir, "embed_*.meta.json"))
        word_to_npy, dim = {}, None
        for mp in metas:
            try:
                meta = json.load(open(mp, "r"))
                w = meta.get("word")
                if not w: continue
                npy_path = mp[:-10] + ".npy"  # 去掉 .meta.json
                if not os.path.exists(npy_path): continue
                word_to_npy[w] = npy_path
                if dim is None:
                    dim = int(meta.get("vector_dim", 0)) or int(np.load(npy_path).shape[-1])
            except Exception:
                continue
        return word_to_npy, dim

    @staticmethod
    def _load_csv_words(csv_path: str | None):
        if not csv_path or not os.path.exists(csv_path): return None
        words, seen = [], set()
        with open(csv_path, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for row in rd:
                w = (row.get("word") or "").strip()
                if w and w not in seen:
                    words.append(w); seen.add(w)
        return words

    def _build_matrix_for_source(self, source: str, csv_path: str | None, force_rebuild: bool, cleanup_chunks: bool):
        os.makedirs(self._matrix_dir(source), exist_ok=True)
        mat_path, norms_path, words_path, meta_path = self._matrix_paths(source)
        need = force_rebuild or not (os.path.exists(mat_path) and os.path.exists(norms_path) and os.path.exists(words_path) and os.path.exists(meta_path))
        if not need:  # 已存在
            return

        chunk_dir = self._chunk_dir(source)
        word_to_npy, dim = self._scan_chunk_meta(chunk_dir)
        if not word_to_npy:
            raise RuntimeError(f"[{source}] No chunk files in {chunk_dir}. Generate embeddings first.")
        if dim is None:
            any_path = next(iter(word_to_npy.values()))
            dim = int(np.load(any_path).shape[-1])

        csv_words = self._load_csv_words(csv_path)
        words = [w for w in (csv_words or sorted(word_to_npy.keys())) if w in word_to_npy]
        N = len(words)
        if N == 0:
            raise RuntimeError(f"[{source}] Zero words to build matrix.")

        X = open_memmap(mat_path, mode="w+", dtype=np.float32, shape=(N, dim))
        norms = open_memmap(norms_path, mode="w+", dtype=np.float32, shape=(N,))
        for i, w in enumerate(words):
            v = np.load(word_to_npy[w]).astype(np.float32)
            X[i, :] = v
            norms[i] = np.linalg.norm(v) + 1e-12
            if i % 5000 == 0: X.flush(); norms.flush()
        X.flush(); norms.flush()

        with open(words_path, "w", encoding="utf-8") as f:
            for w in words: f.write(w + "\n")
        json.dump({"rows": N, "dim": dim}, open(meta_path, "w"))

        if cleanup_chunks and os.path.isdir(chunk_dir):
            shutil.rmtree(chunk_dir)

    def build_all_if_needed(self, csv_path: str | None = None, force_rebuild: bool = False, cleanup_chunks: bool = False):
        for src in self.sources:
            self._build_matrix_for_source(src, csv_path, force_rebuild, cleanup_chunks)
        # 清空缓存，后续按需加载
        self._mat   = {s: None for s in self.sources}
        self._norms = {s: None for s in self.sources}
        self._words = {s: None for s in self.sources}
        self._rows  = {s: None for s in self.sources}
        self._dim   = {s: None for s in self.sources}

    # ---------- 加载矩阵（某一路） ----------
    def _load_matrix_source(self, source: str):
        if self._mat[source] is not None:
            return
        mat_path, norms_path, words_path, meta_path = self._matrix_paths(source)
        meta = json.load(open(meta_path, "r"))
        rows, dim = int(meta["rows"]), int(meta["dim"])
        X     = np.load(mat_path, mmap_mode="r")
        norms = np.load(norms_path, mmap_mode="r")
        with open(words_path, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
        assert X.shape == (rows, dim) and norms.shape == (rows,) and len(words) == rows
        self._mat[source], self._norms[source], self._words[source] = X, norms, words
        self._rows[source], self._dim[source] = rows, dim

    # ---------- 编码（一次前向，同时产出 embedding 与 last） ----------
    @torch.inference_mode()
    def _encode_both(self, word: str) -> dict:
        self._ensure_model()
        text = " " + word
        enc = self._tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attn      = enc["attention_mask"].to(self.device)

        # 子词输入嵌入
        tok_emb = self._model.get_input_embeddings()(input_ids)           # [1, T, D]
        # 最后一层 hidden（一次前向）
        out = self._model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        last = out.last_hidden_state                                       # [1, T, D]

        mask = attn.unsqueeze(-1)
        if self.pooling == "sum":
            v_emb  = (tok_emb * mask).sum(dim=1).squeeze(0)
            v_last = (last    * mask).sum(dim=1).squeeze(0)
        else:
            denom = mask.sum(dim=1).clamp(min=1)
            v_emb  = ((tok_emb * mask).sum(dim=1) / denom).squeeze(0)
            v_last = ((last    * mask).sum(dim=1) / denom).squeeze(0)

        return {
            "embedding": v_emb.detach().to("cpu").numpy().astype(np.float32),
            "last":      v_last.detach().to("cpu").numpy().astype(np.float32),
        }

    # ---- CosineEngineDual.topk_for_word：新增 metric ----
    def topk_for_word(self, pseudo_word: str, k: int = 1000, metric: str = "cosine") -> dict[str, list[tuple[str, float]]]:
        """
        metric: 'cosine' 或 'euclid'
        - cosine  => 取余弦最高的 TopK（降序）
        - euclid  => 取欧氏距离最小的 TopK（升序）
        返回：{"embedding": [(word, score), ...], "last": [...]}
        """
        assert metric in ("cosine", "euclid")
        vecs = self._encode_both(pseudo_word)
        result = {}
        for source in self.sources:
            self._load_matrix_source(source)
            v = vecs[source].astype(np.float32)
            vnorm = np.linalg.norm(v) + 1e-12
            v_unit = v / vnorm

            X, norms, rows = self._mat[source], self._norms[source], self._rows[source]
            scores = np.empty(rows, dtype=np.float32)

            # 先算 cos（与之前一致）：cos = (X · v_unit) / ||x||
            for i in range(0, rows, self.block):
                j = min(i + self.block, rows)
                dot_unit = X[i:j].dot(v_unit)                # = ||x|| * cos
                scores[i:j] = dot_unit / (norms[i:j] + 1e-12)

            if metric == "euclid":
                # d = sqrt(||x||^2 + ||v||^2 - 2 * cos * ||x|| * ||v||)
                d2 = norms * norms + (vnorm * vnorm) - 2.0 * scores * norms * vnorm
                scores = np.sqrt(np.maximum(d2, 0.0, dtype=np.float32))  # 用 scores 作为“距离”向量

                kk = min(k, rows)
                idx = np.argpartition(scores, kk-1)[:kk]                 # 距离越小越好（升序）
                idx = idx[np.argsort(scores[idx])]
                result[source] = [(self._words[source][ii], float(scores[ii])) for ii in idx]
            else:
                kk = min(k, rows)
                idx = np.argpartition(-scores, kk-1)[:kk]                # 余弦越大越好（降序）
                idx = idx[np.argsort(-scores[idx])]
                result[source] = [(self._words[source][ii], float(scores[ii])) for ii in idx]
        return result

    # ---- CosineEngineDual.topk_for_words：透传 metric ----
    def topk_for_words(self, pseudo_words: list[str], k: int = 1000, metric: str = "cosine") -> dict[str, dict[str, list[tuple[str, float]]]]:
        out = {}
        for w in pseudo_words:
            out[w] = self.topk_for_word(w, k, metric=metric)
        return out

    # ---- CosineEngineDual.topk_to_csv：文件名里带上 metric ----
    def topk_to_csv(self, pseudo_word: str, k: int, out_csv: str, metric: str = "cosine"):
        res = self.topk_for_word(pseudo_word, k, metric=metric)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["rank", "word", "score", "source", "metric", "pseudo_word", "model_id", "pooling"])
            for src in self.sources:
                rows = res[src]
                for r, (w, s) in enumerate(rows, 1):
                    wr.writerow([r, w, f"{s:.6f}", src, metric, pseudo_word, self.model_id, self.pooling])


    # ---------- 可选：清理小文件（不删除 matrix/，便于复用） ----------
    def cleanup_chunks(self):
        for src in self.sources:
            cd = self._chunk_dir(src)
            if os.path.isdir(cd):
                shutil.rmtree(cd)

def _sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_")[:100] or "x"

class PseudoWordCosineRunner:
    """
    封装：读取/构建“真实词”矩阵 → 对一批伪词做 TopK 余弦相似度（一次得到 embedding & last 两套）。
    - debug=True: 基础目录换成 ./results/word_synthesis_debug/{INDEX}
    - 如果 matrix 缺失，可传 csv_for_build（union 的 CSV）自动补建，仍走 CosineEngineDual 的流程

    主要方法：
      - ensure_matrix(csv_for_build=..., force_rebuild=False, cleanup_chunks=False)
      - topk_for_word(pseudo_word, k=1000) -> {"embedding": [(word, cos), ...], "last": [...]}
      - topk_for_words(pseudo_words, k=1000) -> {pseudo: {...}, ...}
      - topk_to_csv(pseudo_word, k=1000, out_csv=None) -> out_path
    """

    def __init__(
        self,
        index: str = "v4_rpj_llama_s4",
        model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
        pooling: str = "mean",  # 与生成时一致
        debug: bool = False,
        cache_dir: str = os.path.join("models", "language_models"),
        block: int = 20000,
    ):
        base_root = os.path.join("results", "word_synthesis_debug" if debug else "word_synthesis")
        # 建立双路引擎（embedding+last）
        self.engine = CosineEngineDual(
            index=index,
            model_id=model_id,
            pooling=pooling,
            base_results_dir=base_root,   # 关键：适配 debug 路径
            cache_dir=cache_dir,
            block=block,
        )
        self.index = index
        self.model_id = model_id
        self.pooling = pooling
        self.debug = debug
        self.base_root = base_root
        self.model_tag = _sanitize_name(model_id)

    # 确保矩阵存在（若缺失，可用 union 的 CSV 自动补建）
    def ensure_matrix(self,
                      csv_for_build: Optional[str] = None,
                      force_rebuild: bool = False,
                      cleanup_chunks: bool = False):
        """
        csv_for_build: 可选，union 导出的 CSV（用于决定矩阵行顺序）；若矩阵已存在可不传
        force_rebuild: 强制重建矩阵
        cleanup_chunks: 补建后清理 chunks（保留 matrix 以便复用）
        """
        self.engine.build_all_if_needed(csv_path=csv_for_build,
                                        force_rebuild=force_rebuild,
                                        cleanup_chunks=cleanup_chunks)

    # 单个伪词
    def topk_for_word(self, pseudo_word: str, k: int = 1000, metric: str = "cosine"):
        return self.engine.topk_for_word(pseudo_word, k, metric=metric)

    # 多个伪词
    def topk_for_words(self, pseudo_words, k: int = 1000, metric: str = "cosine"):
        return self.engine.topk_for_words(list(pseudo_words), k, metric=metric)

    # 写CSV（默认输出目录同原逻辑；文件名含 metric）
    def topk_to_csv(self, pseudo_word: str, k: int = 1000, out_csv: Optional[str] = None, metric: str = "cosine") -> str:
        if out_csv is None:
            out_dir = os.path.join(self.base_root, self.index, "embeddings", self.model_tag, f"both_{self.pooling}")
            os.makedirs(out_dir, exist_ok=True)
            out_csv = os.path.join(out_dir, f"similar_{metric}_both_{k}_{_sanitize_name(pseudo_word)}.csv")
        self.engine.topk_to_csv(pseudo_word, k, out_csv, metric=metric)
        return out_csv

    
    def _cosine_out_dir(self, source: str, pseudo_word: str) -> str:
        tag = "embedding" if source == "embedding" else f"last_{self.engine.pooling}"
        out_dir = os.path.join(self.base_root, self.index, "cosine_results", tag, _sanitize_name(pseudo_word))
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def save_full_vectors(self, pseudo_word: str) -> dict[str, str]:
        """
        为一个伪词计算并落盘“整向量”：
          vector.npy: (2, N) float32 —— 第1行=cosine，第2行=row_norms
          meta.json:  {rows, model_id, index, source, pooling, words_txt, matrix_dir, pseudo_word}
        返回：{"embedding": <dir>, "last": <dir>} 每一路的输出文件夹路径
        """
        res = self.engine.cosine_full_for_word(pseudo_word)
        out_dirs = {}
        for src in ("embedding", "last"):
            odir = self._cosine_out_dir(src, pseudo_word)
            # 保存 2×N 的矩阵
            cos = res[src]["cos"].astype(np.float32)
            norms = res[src]["norms"].astype(np.float32)
            vec2 = np.vstack([cos[None, :], norms[None, :]])
            np.save(os.path.join(odir, "vector.npy"), vec2)

            # 关联 matrix 目录 & words.txt（方便定位顺序）
            md = self.engine._matrix_dir(src)
            meta = {
                "rows": int(cos.shape[0]),
                "model_id": self.model_id,
                "index": self.index,
                "source": src,
                "pooling": self.pooling,
                "pseudo_word": pseudo_word,
                "matrix_dir": os.path.abspath(md),
                "words_txt": os.path.abspath(os.path.join(md, "words.txt")),
            }
            with open(os.path.join(odir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            out_dirs[src] = odir
        return out_dirs

    def save_tsne(self, pseudo_word: str, source: str,
                  n_real: int = 1200, top_k: int = 200,
                  random_state: int = 42) -> str:
        """
        生成 t-SNE 可视化（PNG），展示伪词与真实词的分布/离散程度。
        - 从真实词里取 top_k 近邻 + (n_real - top_k) 随机样本
        - 用真实的 D 维向量（单位化）+ 伪词向量一起做 t-SNE
        后端优先级：cuML (CUDA) -> scikit-learn (CPU)
        """
        import numpy as np
        assert source in ("embedding", "last")

        # 1) 取矩阵与词
        self.engine._load_matrix_source(source)
        X = self.engine._mat[source]           # (N, D) memmap
        norms = self.engine._norms[source]     # (N,)
        words = self.engine._words[source]
        N, D = X.shape

        # 2) 伪词向量（单位化）
        v_map = self.engine._encode_both(pseudo_word)
        v = v_map[source].astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-12)

        # 3) 计算整向量余弦（用于选 top_k）
        sims = np.empty(N, dtype=np.float32)
        for i in range(0, N, self.engine.block):
            j = min(i + self.engine.block, N)
            dot = X[i:j].dot(v)
            sims[i:j] = dot / (norms[i:j] + 1e-12)

        # 4) 选择样本：top_k + 随机补齐到 n_real（若 n_real < top_k，就只用 top_k）
        top_k = int(min(max(1, top_k), N))
        idx_top = np.argpartition(-sims, top_k-1)[:top_k]
        rng = np.random.default_rng(random_state)
        remain = np.setdiff1d(np.arange(N, dtype=np.int64), idx_top, assume_unique=False)
        n_target = int(max(top_k, n_real))  # 至少包含 top_k
        n_rand = max(0, min(n_target - top_k, remain.size))
        idx_rand = rng.choice(remain, size=n_rand, replace=False) if n_rand > 0 else np.array([], dtype=np.int64)
        idx_all = np.concatenate([idx_top, idx_rand])
        rng.shuffle(idx_all)

        # 5) 构建 t-SNE 输入：单位化的向量
        Xu = X[idx_all] / (norms[idx_all, None] + 1e-12)
        data = np.vstack([Xu, v[None, :]])   # 最后一行是伪词
        n_samples = data.shape[0]

        # 6) 合理的 perplexity（必须 < n_samples）
        perplexity = max(5, min(30, n_samples // 3))
        if perplexity >= n_samples:
            perplexity = max(5, n_samples - 1)

        # 7) 选择后端：优先 cuML(CUDA)，否则回退 sklearn(CPU)
        Y = None
        used_backend = "sklearn"
        try:
            # 尝试 CUDA + cuML
            import cupy as cp
            from cuml.manifold import TSNE as cuTSNE
            if cp.cuda.runtime.getDeviceCount() > 0:
                used_backend = "cuml"
                X_gpu = cp.asarray(data, dtype=cp.float32)
                tsne = cuTSNE(
                    n_components=2,
                    perplexity=float(perplexity),
                    learning_rate=200.0,    # cuML 要求数值
                    n_iter=1000,
                    method='barnes_hut',
                    random_state=random_state,
                    verbose=0,
                )
                Y_gpu = tsne.fit_transform(X_gpu)  # cupy array
                Y = cp.asnumpy(Y_gpu)             # -> numpy
        except Exception:
            used_backend = "sklearn"

        if Y is None:
            # sklearn CPU：兼容不同版本的构造参数（老版本不支持 learning_rate='auto' / n_iter）
            try:
                from sklearn.manifold import TSNE as SKTSNE
            except Exception as e:
                raise RuntimeError("需要 scikit-learn 或 cuML 之一来绘制 t-SNE。") from e

            try:
                tsne = SKTSNE(
                    n_components=2,
                    perplexity=float(perplexity),
                    learning_rate='auto',
                    init='random',
                    random_state=random_state,
                    n_iter=1000
                )
            except TypeError:
                # 老版本降级：去掉不支持的参数
                tsne = SKTSNE(
                    n_components=2,
                    perplexity=float(perplexity),
                    init='random',
                    random_state=random_state
                )
            Y = tsne.fit_transform(data)

        # 8) 画图（CPU）
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        odir = self._cosine_out_dir(source, pseudo_word)
        png = os.path.join(odir, "tsne.png")
        plt.figure(figsize=(6, 5), dpi=160)
        plt.scatter(Y[:-1, 0], Y[:-1, 1], s=6, alpha=0.45, label=f"real words (backend={used_backend})")
        plt.scatter(Y[-1, 0],  Y[-1, 1],  s=60, marker="*", label=pseudo_word)
        plt.title(f"t-SNE ({source}, n={n_samples}, perplexity={perplexity})")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(png)
        plt.close()
        return png


    def save_vectors_and_tsne(self, pseudo_word: str, n_real: int = 1200, top_k: int = 200) -> dict:
        out_dirs = self.save_full_vectors(pseudo_word)
        pngs = {}
        for src in ("embedding", "last"):
            pngs[src] = self.save_tsne(pseudo_word, src, n_real=n_real, top_k=top_k)
        return {"dirs": out_dirs, "tsne_png": pngs}
