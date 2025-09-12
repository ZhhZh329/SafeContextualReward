# candidate_filter.py
# 读取 cosine_results/{embedding,last_mean}/{pseudo}/vector.npy:
#   vector.npy: (2, N)  row0=cosine, row1=row_norms(=||x||)
# 生成五类结果（都在 results/.../{INDEX}/<rank_dirname>/ 与 real_sort/ 下）：
#   1) 伪词：余弦（各路各一份，按“余弦TopK”平均，并标注max cos与对应词）
#   2) 伪词：欧氏距离（各路各一份，按“距离TopK”平均，并标注min dist与对应词）
#   3) 伪词：语料词频（Infini-gram count）
#   4) 真实词对照组（seed=42 随机200个）：余弦（各路各一）、欧氏距离（各路各一）
import os, re, json, csv, time, glob
import numpy as np
from typing import Dict, Tuple, Optional
from tqdm import tqdm

# ======== 基本配置 ========
INDEX    = "v4_rpj_llama_s4"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DEBUG    = False

TOPK_NEIGHBORS = 200      # 余弦和距离都各自用Top-K
QPS_COUNT      = 6        # Infini-gram count QPS
RETRY          = 5
RANK_DIRNAME   = "pseudoword_ranking"  # 或 "伪词排序"

# ======== 路径推导 ========
BASE_DIR = os.path.join(".", "results", "word_synthesis_debug" if DEBUG else "word_synthesis", INDEX)
COS_DIR  = os.path.join(BASE_DIR, "cosine_results")
RANK_DIR = os.path.join(BASE_DIR, RANK_DIRNAME)
os.makedirs(RANK_DIR, exist_ok=True)

def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_")[:100] or "x"

def _src_tag(source: str) -> str:
    return "embedding" if source == "embedding" else "last_mean"

def _vector_path(source: str, pseudo_word: str) -> str:
    return os.path.join(COS_DIR, _src_tag(source), _sanitize(pseudo_word), "vector.npy")

def _meta_path(source: str, pseudo_word: str) -> str:
    return os.path.join(COS_DIR, _src_tag(source), _sanitize(pseudo_word), "meta.json")

# ======== 读取单个伪词的两路整向量 ========
def load_vectors_for_pseudo(pseudo_word: str) -> Dict[str, Dict[str, np.ndarray]]:
    out = {}
    for src in ("embedding", "last"):
        vp = _vector_path(src, pseudo_word)
        if not os.path.exists(vp):
            continue
        arr = np.load(vp)  # (2,N)
        if arr.ndim != 2 or arr.shape[0] != 2:
            print(f"[WARN] bad vector shape for {pseudo_word} {src}: {arr.shape}")
            continue
        out[src] = {"cos": arr[0].astype(np.float32), "norms": arr[1].astype(np.float32)}
    return out

# ======== 取/缓存 伪词范数 ||v||（逐路） ========
def _get_pseudo_norm(source: str, pseudo_word: str) -> float:
    mp = _meta_path(source, pseudo_word)
    if os.path.exists(mp):
        try:
            meta = json.load(open(mp, "r"))
            if "pseudo_norm" in meta:
                return float(meta["pseudo_norm"])
        except Exception:
            pass
    # 需要一次性编码伪词向量以拿 ||v||
    try:
        try:
            from src.data_construction.word_synthesis.cosine_engine_dual import CosineEngineDual
        except Exception:
            from src.data_construction.word_synthesis.cosine_similarity_pseudoword import CosineEngineDual
        base_root = os.path.join("results", "word_synthesis_debug" if DEBUG else "word_synthesis")
        engine = CosineEngineDual(index=INDEX, model_id=MODEL_ID, pooling="mean",
                                  base_results_dir=base_root, cache_dir=os.path.join("models","language_models"))
        vecs = engine._encode_both(pseudo_word)
        v = vecs["embedding" if source=="embedding" else "last"]
        vnorm = float(np.linalg.norm(v))
        # 缓存回 meta.json
        try:
            meta = json.load(open(mp, "r")) if os.path.exists(mp) else {}
            meta["pseudo_norm"] = vnorm
            with open(mp, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return vnorm
    except Exception as e:
        raise RuntimeError(f"[pseudo_norm] 无法获取伪词范数：{pseudo_word} {source} ({e})")

# ======== 逐 source 的汇总 ========
def cosine_summary_per_source(vmap: Dict[str, Dict[str, np.ndarray]], source: str, topk: int,
                              words: Optional[list]=None) -> Tuple[float, float, str]:
    """
    返回 (avg_topK_cos, max_cos, max_cos_word)
      - Top-K 取“余弦最大”的K个
      - 若 words 提供，则给出 max_cos 对应的词
    """
    if source not in vmap: return float("nan"), float("nan"), ""
    cos = vmap[source]["cos"]
    if cos.size == 0: return float("nan"), float("nan"), ""
    kk = int(min(max(1, topk), cos.size))
    idx_top = np.argpartition(-cos, kk-1)[:kk]
    avg_topk = float(np.mean(cos[idx_top]))
    i_best = int(np.argmax(cos))
    best_word = (words[i_best] if words and 0 <= i_best < len(words) else "")
    return avg_topk, float(cos[i_best]), best_word

def euclid_summary_per_source(vmap: Dict[str, Dict[str, np.ndarray]], source: str, topk: int,
                              pseudo_word: str, words: Optional[list]=None) -> Tuple[float, float, str]:
    """
    返回 (avg_topK_euclid, min_dist, min_dist_word)
      - 距离 d = sqrt(||x||^2 + ||v||^2 - 2*cos*||x||*||v||)
      - Top-K 取“距离最小”的K个
    """
    if source not in vmap: return float("nan"), float("nan"), ""
    cos   = vmap[source]["cos"]
    norms = vmap[source]["norms"]
    if cos.size == 0: return float("nan"), float("nan"), ""
    vnorm = _get_pseudo_norm(source, pseudo_word)
    # 逐行距离
    d2 = norms*norms + (vnorm*vnorm) - 2.0 * cos * norms * vnorm
    d  = np.sqrt(np.maximum(d2, 0.0, dtype=np.float32))
    kk = int(min(max(1, topk), d.size))
    idx_top = np.argpartition(d, kk-1)[:kk]  # 距离越小越近
    avg_topk = float(np.mean(d[idx_top]))
    i_best = int(np.argmin(d))
    best_word = (words[i_best] if words and 0 <= i_best < len(words) else "")
    return avg_topk, float(d[i_best]), best_word

# ======== 词频（Infini-gram） ========
def count_words_infini(pseudo_words, index: str, qps: float = QPS_COUNT, retry: int = RETRY) -> Dict[str, int]:
    import importlib
    mod = importlib.import_module("src.data_construction.word_synthesis.word_count_from_infini")
    mod.INDEX = index
    gap = 1.0 / float(qps)
    out = {}
    for w in pseudo_words:
        q = " " + w
        last_err = None
        for i in range(retry):
            try:
                t0 = time.monotonic()
                c = int(mod.count_exact(q))
                out[w] = c
                used = time.monotonic() - t0
                if used < gap:
                    time.sleep(gap - used)
                break
            except Exception as e:
                last_err = e
                time.sleep(min(2.0, 0.5 * (2 ** i)))
        if w not in out:
            print(f"[WARN] count_exact failed for {w}: {last_err}")
            out[w] = 0
    return out

# ======== 主流程（伪词） ========
def run(topk_neighbors: int = TOPK_NEIGHBORS):
    # 收集伪词
    emb_dirs  = glob.glob(os.path.join(COS_DIR, "embedding", "*"))
    last_dirs = glob.glob(os.path.join(COS_DIR, "last_mean", "*"))
    names = {os.path.basename(d) for d in emb_dirs + last_dirs if os.path.isdir(d)}
    pseudo_words = sorted(names)
    if not pseudo_words:
        print(f"[FATAL] 未在 {COS_DIR} 下发现伪词向量，请先运行 save_full_vectors()"); return

    # 读一次 words.txt（各路各取一份）
    words_map = {"embedding": None, "last": None}
    def _words_for(src, sample_pseudo):
        if words_map[src] is not None: return words_map[src]
        mp = _meta_path(src, sample_pseudo)
        if os.path.exists(mp):
            meta = json.load(open(mp, "r"))
            wt = meta.get("words_txt") or ""
            wt = wt if (wt and os.path.exists(wt)) else os.path.join(meta.get("matrix_dir",""), "words.txt")
            if wt and os.path.exists(wt):
                with open(wt, "r", encoding="utf-8") as f:
                    words_map[src] = [ln.strip() for ln in f if ln.strip()]
        return words_map[src]

    rows_cos  = {"embedding": [], "last": []}  # (pseudo, avgTopK_cos, max_cos, max_cos_word)
    rows_euc  = {"embedding": [], "last": []}  # (pseudo, avgTopK_euclid, min_dist, min_dist_word)

    for pw in tqdm(pseudo_words):
        vmap = load_vectors_for_pseudo(pw)
        if not vmap: 
            print(f"[WARN] 无向量：{pw}")
            continue
        for src in ("embedding", "last"):
            if src not in vmap: continue
            words = _words_for(src, pw) or []
            # 余弦（按余弦TopK）
            avgc, maxc, maxw = cosine_summary_per_source(vmap, src, topk_neighbors, words)
            rows_cos[src].append((pw, avgc, maxc, maxw))
            # 欧氏距离（按距离TopK）
            avge, mind, minw = euclid_summary_per_source(vmap, src, topk_neighbors, pw, words)
            rows_euc[src].append((pw, avge, mind, minw))

    # 词频
    freq_map = count_words_infini(pseudo_words, INDEX, qps=QPS_COUNT, retry=RETRY)

    # 写CSV：余弦
    out_cos_emb = os.path.join(RANK_DIR, f"rank_cosine_avgTop{topk_neighbors}__embedding.csv")
    out_cos_last = os.path.join(RANK_DIR, f"rank_cosine_avgTop{topk_neighbors}__last_mean.csv")
    for path, key in ((out_cos_emb,"embedding"), (out_cos_last,"last")):
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f); wr.writerow(["pseudo_word","avg_topK_cosine","max_cosine","max_cos_word"])
            for pw, avgc, maxc, maxw in sorted(rows_cos[key], key=lambda x:(-x[1], -x[2], x[0])):
                wr.writerow([pw, f"{avgc:.6f}", f"{maxc:.6f}", maxw])

    # 写CSV：欧氏距离
    out_euc_emb = os.path.join(RANK_DIR, f"rank_euclid_avgTop{topk_neighbors}__embedding.csv")
    out_euc_last = os.path.join(RANK_DIR, f"rank_euclid_avgTop{topk_neighbors}__last_mean.csv")
    for path, key in ((out_euc_emb,"embedding"), (out_euc_last,"last")):
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f); wr.writerow(["pseudo_word","avg_topK_euclid","min_euclid","min_euclid_word"])
            for pw, avge, mind, minw in sorted(rows_euc[key], key=lambda x:(x[1], x[0])):  # 距离越小越近
                wr.writerow([pw, f"{avge:.6f}", f"{mind:.6f}", minw])

    # 写CSV：词频
    out_fr = os.path.join(RANK_DIR, f"rank_by_frequency.csv")
    with open(out_fr, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["pseudo_word","count"])
        for pw in sorted(pseudo_words, key=lambda w:(-int(freq_map.get(w,0)), w)):
            wr.writerow([pw, int(freq_map.get(pw,0))])

    print("[DONE] pseudo ranking:")
    print(" -", out_cos_emb)
    print(" -", out_cos_last)
    print(" -", out_euc_emb)
    print(" -", out_euc_last)
    print(" -", out_fr)
    return out_cos_emb, out_cos_last, out_euc_emb, out_euc_last, out_fr

# ========= 真实词对照组 =========
def _find_any_matrix_from_meta(source: str) -> tuple[str, str]:
    src_tag = _src_tag(source)
    metas = glob.glob(os.path.join(COS_DIR, src_tag, "*", "meta.json"))
    if not metas:
        raise FileNotFoundError(f"[real_control] 找不到 {COS_DIR}/{src_tag}/*/meta.json，先跑 save_full_vectors()")
    meta = json.load(open(metas[0], "r"))
    md = meta.get("matrix_dir","")
    wt = meta.get("words_txt","")
    if not (md and os.path.isdir(md)): raise FileNotFoundError(f"[real_control] matrix_dir 不可用：{md}")
    if not (wt and os.path.exists(wt)):
        wt = os.path.join(md, "words.txt")
        if not os.path.exists(wt): raise FileNotFoundError(f"[real_control] words.txt 不可用：{wt}")
    return md, wt

def run_real_control(sample_n: int = 200, seed: int = 42, topk_neighbors: int = TOPK_NEIGHBORS,
                     out_dirname: str = "real_sort"):
    out_base = os.path.join(BASE_DIR, out_dirname)
    os.makedirs(out_base, exist_ok=True)

    for src in ("embedding", "last"):
        # 1) matrix & words
        matrix_dir, words_txt = _find_any_matrix_from_meta(src)
        X     = np.load(os.path.join(matrix_dir, "matrix.npy"), mmap_mode="r")
        norms = np.load(os.path.join(matrix_dir, "row_norms.npy"), mmap_mode="r")
        with open(words_txt, "r", encoding="utf-8") as f:
            words = [ln.strip() for ln in f if ln.strip()]
        N, D = X.shape
        if N != len(words) or N != norms.shape[0]:
            raise RuntimeError(f"[real_control] 维度不一致: N={N}")

        # 2) 随机抽样
        rng = np.random.default_rng(seed)
        if sample_n > N: sample_n = N
        idx_sample = rng.choice(np.arange(N, dtype=np.int64), size=sample_n, replace=False)

        rows_cos, rows_euc = [], []
        block = 20000
        for idx in idx_sample:
            v = X[idx].astype(np.float32)
            vnorm = float(norms[idx])
            v_unit = v / (vnorm + 1e-12)

            # 余弦（与所有其他词）
            sims = np.empty(N, dtype=np.float32)
            for i in range(0, N, block):
                j = min(i + block, N)
                sims[i:j] = X[i:j].dot(v_unit) / (norms[i:j] + 1e-12)
            sims[idx] = -1e9

            # 余弦TopK
            kk = int(min(max(1, topk_neighbors), N-1))
            idx_cos = np.argpartition(-sims, kk-1)[:kk]
            avgc = float(np.mean(sims[idx_cos]))
            i_best = int(idx_cos[np.argmax(sims[idx_cos])])
            rows_cos.append((words[idx], avgc, float(sims[i_best]), words[i_best]))

            # 欧氏距离（用 cos + norms + vnorm 还原）
            d2 = norms*norms + (vnorm*vnorm) - 2.0 * sims * norms * vnorm
            d  = np.sqrt(np.maximum(d2, 0.0, dtype=np.float32))
            d[idx] = 1e9
            idx_d = np.argpartition(d, kk-1)[:kk]
            avge = float(np.mean(d[idx_d]))
            i_min = int(idx_d[np.argmin(d[idx_d])])
            rows_euc.append((words[idx], avge, float(d[i_min]), words[i_min]))

        tag = _src_tag(src)
        out_cos = os.path.join(out_base, f"rank_cosine_avgTop{topk_neighbors}__{tag}.csv")
        out_euc = os.path.join(out_base, f"rank_euclid_avgTop{topk_neighbors}__{tag}.csv")

        with open(out_cos, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f); wr.writerow(["real_word","avg_topK_cosine","max_cosine","max_cos_word"])
            for w, avgc, maxc, maxw in sorted(rows_cos, key=lambda x:(-x[1], -x[2], x[0])):
                wr.writerow([w, f"{avgc:.6f}", f"{maxc:.6f}", maxw])

        with open(out_euc, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f); wr.writerow(["real_word","avg_topK_euclid","min_euclid","min_euclid_word"])
            for w, avge, mind, minw in sorted(rows_euc, key=lambda x:(x[1], x[0])):
                wr.writerow([w, f"{avge:.6f}", f"{mind:.6f}", minw])

        print(f"[REAL] {tag}: wrote\n - {out_cos}\n - {out_euc}")

    return os.path.abspath(out_base)

# ===== 轻量包装：一行跑完 =====
class CandidateFilterRunner:
    """
    test_main.py：
        from src.data_construction.word_synthesis.candidate_filter import CandidateFilterRunner
        outs = CandidateFilterRunner(
            index="v4_rpj_llama_s4",
            model_id="meta-llama/Llama-3.2-1B-Instruct",
            debug=False,
            topk_neighbors=200,
            qps=6, retry=5,
            rank_dirname="伪词排序",
            run_real_control=True, real_sample_n=200, real_seed=42, real_dirname="real_sort"
        ).run()
    """
    def __init__(self, index: str, model_id: str, debug: bool = False,
                 topk_neighbors: int = 200, qps: int = 6, retry: int = 5,
                 rank_dirname: str = RANK_DIRNAME,
                 run_real_control: bool = False, real_sample_n: int = 200,
                 real_seed: int = 42, real_dirname: str = "real_sort"):
        self.index=index; self.model_id=model_id; self.debug=bool(debug)
        self.topk=int(topk_neighbors); self.qps=int(qps); self.retry=int(retry)
        self.rank_dirname=rank_dirname
        self.run_real=bool(run_real_control); self.real_n=int(real_sample_n)
        self.real_seed=int(real_seed); self.real_dirname=real_dirname

    def _apply_globals(self):
        global INDEX, MODEL_ID, DEBUG, TOPK_NEIGHBORS, QPS_COUNT, RETRY
        global BASE_DIR, COS_DIR, RANK_DIR, RANK_DIRNAME
        INDEX, MODEL_ID, DEBUG = self.index, self.model_id, self.debug
        TOPK_NEIGHBORS = self.topk; QPS_COUNT=self.qps; RETRY=self.retry
        RANK_DIRNAME=self.rank_dirname
        BASE_DIR = os.path.join(".", "results", "word_synthesis_debug" if DEBUG else "word_synthesis", INDEX)
        COS_DIR  = os.path.join(BASE_DIR, "cosine_results")
        RANK_DIR = os.path.join(BASE_DIR, RANK_DIRNAME)
        os.makedirs(RANK_DIR, exist_ok=True)

    def expected_paths(self):
        self._apply_globals()
        paths = {
            "cosine_embedding": os.path.abspath(os.path.join(RANK_DIR, f"rank_cosine_avgTop{TOPK_NEIGHBORS}__embedding.csv")),
            "cosine_last":      os.path.abspath(os.path.join(RANK_DIR, f"rank_cosine_avgTop{TOPK_NEIGHBORS}__last_mean.csv")),
            "euclid_embedding": os.path.abspath(os.path.join(RANK_DIR, f"rank_euclid_avgTop{TOPK_NEIGHBORS}__embedding.csv")),
            "euclid_last":      os.path.abspath(os.path.join(RANK_DIR, f"rank_euclid_avgTop{TOPK_NEIGHBORS}__last_mean.csv")),
            "frequency":        os.path.abspath(os.path.join(RANK_DIR, f"rank_by_frequency.csv")),
        }
        if self.run_real:
            real_base = os.path.join(BASE_DIR, self.real_dirname)
            paths.update({
                "real_dir":           os.path.abspath(real_base),
                "real_cosine_emb":    os.path.abspath(os.path.join(real_base, f"rank_cosine_avgTop{TOPK_NEIGHBORS}__embedding.csv")),
                "real_cosine_last":   os.path.abspath(os.path.join(real_base, f"rank_cosine_avgTop{TOPK_NEIGHBORS}__last_mean.csv")),
                "real_euclid_emb":    os.path.abspath(os.path.join(real_base, f"rank_euclid_avgTop{TOPK_NEIGHBORS}__embedding.csv")),
                "real_euclid_last":   os.path.abspath(os.path.join(real_base, f"rank_euclid_avgTop{TOPK_NEIGHBORS}__last_mean.csv")),
            })
        return paths

    def run(self):
        self._apply_globals()
        out_cos_emb, out_cos_last, out_euc_emb, out_euc_last, out_fr = run(topk_neighbors=TOPK_NEIGHBORS)
        result = {
            "cosine_embedding": os.path.abspath(out_cos_emb),
            "cosine_last":      os.path.abspath(out_cos_last),
            "euclid_embedding": os.path.abspath(out_euc_emb),
            "euclid_last":      os.path.abspath(out_euc_last),
            "frequency":        os.path.abspath(out_fr),
        }
        if self.run_real:
            real_dir = run_real_control(sample_n=self.real_n, seed=self.real_seed,
                                        topk_neighbors=TOPK_NEIGHBORS, out_dirname=self.real_dirname)
            result.update(self.expected_paths())
        summary_csv, union_txt, inter_txt = build_union_and_intersection_by_means(
            topk_neighbors=TOPK_NEIGHBORS, out_dirname="selection", euclid_rule="high"
        )
        result.update({
            "selection_summary": os.path.abspath(summary_csv),
            "selection_union":   os.path.abspath(union_txt),
            "selection_intersection": os.path.abspath(inter_txt),
        })
        return result
# ========= 选择规则：并集/交集（基于“低相似、远距离、低频率”） =========
def _load_metric_csv(path: str, key_col: str, val_col: str) -> dict:
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            k = (row.get(key_col) or "").strip()
            v = row.get(val_col)
            if not k or v is None:
                continue
            try:
                d[k] = float(v)
            except Exception:
                continue
    return d

def build_union_and_intersection_by_means(topk_neighbors: int = TOPK_NEIGHBORS,
                                          out_dirname: str = "selection",
                                          euclid_rule: str = "high"):
    """
    从排序结果中构建两个集合：
      - cosine:  avg_topK_cosine[embedding] < mean_emb  且  avg_topK_cosine[last] < mean_last
      - euclid:  avg_topK_euclid[embedding] > mean_emb  且  avg_topK_euclid[last] > mean_last  （euclid_rule='high'）
      - freq  :  count < mean_count
    然后：
      UNION = cosine_set ∪ euclid_set ∪ freq_set
      INTER = cosine_set ∩ euclid_set ∩ freq_set

    输出：
      {BASE_DIR}/{RANK_DIRNAME}/{out_dirname}/summary.csv
      {BASE_DIR}/{RANK_DIRNAME}/{out_dirname}/union.txt
      {BASE_DIR}/{RANK_DIRNAME}/{out_dirname}/intersection.txt
      {BASE_DIR}/{RANK_DIRNAME}/{out_dirname}/meta.json
    """
    sel_dir = os.path.join(RANK_DIR, out_dirname)
    os.makedirs(sel_dir, exist_ok=True)

    # 1) 读取五张表
    cos_emb_path  = os.path.join(RANK_DIR, f"rank_cosine_avgTop{topk_neighbors}__embedding.csv")
    cos_last_path = os.path.join(RANK_DIR, f"rank_cosine_avgTop{topk_neighbors}__last_mean.csv")
    euc_emb_path  = os.path.join(RANK_DIR, f"rank_euclid_avgTop{topk_neighbors}__embedding.csv")
    euc_last_path = os.path.join(RANK_DIR, f"rank_euclid_avgTop{topk_neighbors}__last_mean.csv")
    freq_path     = os.path.join(RANK_DIR, "rank_by_frequency.csv")

    cos_emb  = _load_metric_csv(cos_emb_path,  key_col="pseudo_word", val_col="avg_topK_cosine")
    cos_last = _load_metric_csv(cos_last_path, key_col="pseudo_word", val_col="avg_topK_cosine")
    euc_emb  = _load_metric_csv(euc_emb_path,  key_col="pseudo_word", val_col="avg_topK_euclid")
    euc_last = _load_metric_csv(euc_last_path, key_col="pseudo_word", val_col="avg_topK_euclid")
    freq_map = {}
    if os.path.exists(freq_path):
        with open(freq_path, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for row in rd:
                w = (row.get("pseudo_word") or "").strip()
                try:
                    c = int(row.get("count") or "0")
                except Exception:
                    c = 0
                if w:
                    freq_map[w] = c

    # 2) 计算均值（分别按 source & 指标）
    def _mean(vals: dict) -> float:
        arr = [v for v in vals.values() if isinstance(v, (int, float))]
        return float(np.mean(arr)) if arr else float("nan")

    cos_mean_emb  = _mean(cos_emb)
    cos_mean_last = _mean(cos_last)
    euc_mean_emb  = _mean(euc_emb)
    euc_mean_last = _mean(euc_last)
    freq_mean     = _mean(freq_map)

    # 3) 构建集合
    keys = set(cos_emb) | set(cos_last) | set(euc_emb) | set(euc_last) | set(freq_map)

    # 3.1 余弦低（两路都低于均值）
    set_cosine = {
        k for k in keys
        if (k in cos_emb  and k in cos_last and
            (cos_emb.get(k, 1e9)  < cos_mean_emb) and
            (cos_last.get(k, 1e9) < cos_mean_last))
    }

    # 3.2 距离远（两路都高于均值；如果你想“低于均值”改为 euclid_rule='low'）
    if euclid_rule == "low":
        set_euclid = {
            k for k in keys
            if (k in euc_emb and k in euc_last and
                (euc_emb.get(k, -1e9)  < euc_mean_emb) and
                (euc_last.get(k, -1e9) < euc_mean_last))
        }
    else:  # 'high'：远
        set_euclid = {
            k for k in keys
            if (k in euc_emb and k in euc_last and
                (euc_emb.get(k, -1e9)  > euc_mean_emb) and
                (euc_last.get(k, -1e9) > euc_mean_last))
        }

    # 3.3 频数低
    set_freq = {k for k in keys if (k in freq_map and freq_map.get(k, 1e18) < freq_mean)}

    UNION = set_cosine | set_euclid | set_freq
    INTER = set_cosine & set_euclid & set_freq

    # 4) 写 summary.csv（便于对照）
    summary_csv = os.path.join(sel_dir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow([
            "pseudo_word",
            "cos_emb", "cos_last", "cos_mean_emb", "cos_mean_last", "cos_low_both",
            "euc_emb", "euc_last", "euc_mean_emb", "euc_mean_last", "euc_far_both",
            "freq", "freq_mean", "freq_low",
            "in_union", "in_intersection"
        ])
        for k in sorted(keys):
            ce = cos_emb.get(k, float("nan"))
            cl = cos_last.get(k, float("nan"))
            ee = euc_emb.get(k, float("nan"))
            el = euc_last.get(k, float("nan"))
            fr = float(freq_map.get(k, 0))
            cos_flag = (k in set_cosine)
            euc_flag = (k in set_euclid)
            fr_flag  = (k in set_freq)
            wr.writerow([
                k,
                f"{ce:.6f}", f"{cl:.6f}", f"{cos_mean_emb:.6f}", f"{cos_mean_last:.6f}", int(cos_flag),
                f"{ee:.6f}", f"{el:.6f}", f"{euc_mean_emb:.6f}", f"{euc_mean_last:.6f}", int(euc_flag),
                int(fr), f"{freq_mean:.2f}", int(fr_flag),
                int(k in UNION), int(k in INTER),
            ])

    # 5) 写 union / intersection 列表
    union_txt = os.path.join(sel_dir, "union.txt")
    inter_txt = os.path.join(sel_dir, "intersection.txt")
    with open(union_txt, "w", encoding="utf-8") as f:
        for k in sorted(UNION): f.write(k + "\n")
    with open(inter_txt, "w", encoding="utf-8") as f:
        for k in sorted(INTER): f.write(k + "\n")

    # 6) meta.json（记录阈值）
    meta = {
        "index": INDEX, "model_id": MODEL_ID, "debug": DEBUG,
        "topk_neighbors": int(topk_neighbors),
        "euclid_rule": euclid_rule,  # 'high' 表示取“大于均值”为“远”
        "means": {
            "cos_emb": float(cos_mean_emb),
            "cos_last": float(cos_mean_last),
            "euc_emb": float(euc_mean_emb),
            "euc_last": float(euc_mean_last),
            "freq": float(freq_mean),
        },
        "sizes": {
            "cosine_set": len(set_cosine),
            "euclid_set": len(set_euclid),
            "freq_set": len(set_freq),
            "union": len(UNION),
            "intersection": len(INTER),
            "total_keys": len(keys),
        },
        "inputs": {
            "cos_emb": cos_emb_path, "cos_last": cos_last_path,
            "euc_emb": euc_emb_path, "euc_last": euc_last_path,
            "freq": freq_path,
        },
        "outputs": {
            "summary_csv": summary_csv,
            "union_txt": union_txt,
            "intersection_txt": inter_txt,
        }
    }
    with open(os.path.join(sel_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[SELECTION] wrote:")
    print(" -", summary_csv)
    print(" -", union_txt)
    print(" -", inter_txt)
    return summary_csv, union_txt, inter_txt


if __name__ == "__main__":
    run()
