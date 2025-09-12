# union_with_wordfreq.py
# 把主流程导出的 top_words_{TOPK}.csv 与 wordfreq(英语/best) 的 Top-N 词做并集；
# 对 CSV 中不存在的新词逐个用 Infini-gram `count` 精确计数；合并后按频次降序输出 TOPK。
import os, re, csv, sys, time, threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from wordfreq import top_n_list

# ================= 配 置 =================
API   = "https://api.infini-gram.io/"
INDEX = "v4_rpj_llama_s4"   # 与主流程一致

TOPK         = 11000        # 最终输出条数（也决定默认读取哪个 base CSV）
WORD_FREQ_N  = 50000        # 从 wordfreq 取前 N（建议 30k~50k）
QPS          = 6            # 限速，与主流程一致更稳
WORKERS      = 32           # 并发上限
RETRY        = 5            # 单请求最大重试

# 词形规则（与主流程对齐）
MIN_WORD_LEN = 2            # 过滤过短词
ALLOW_APOS   = True         # 允许 "don't"
ALLOW_HYPHEN = True         # 允许 "state-of-the-art"

# wordfreq 设置：只考虑英语 + best 词表
WORD_FREQ_LANG = "en"
WORD_FREQ_LIST = "best"

# 路径（与主流程保持一致的目录与命名）
BASE_DIR = os.path.join(".", "results", "word_synthesis", INDEX)
BASE_CSV = os.path.join(BASE_DIR, f"top_words_{TOPK}.csv")  # 主流程输出
# BASE_CSV = "/Users/zhzhou/Desktop/SafeContextualReward/results/word_synthesis/v4_rpj_llama_s4/top_words_10000_1239.csv"
OUT_CSV  = os.path.join(BASE_DIR, f"union_top_words_{TOPK}_wf{WORD_FREQ_N}.csv")
# （可选）也导出完整并集不截断，便于检查
OUT_CSV_ALL = os.path.join(BASE_DIR, f"union_all_words_wf{WORD_FREQ_N}.csv")
# =======================================

# 词形过滤（与主流程规则对齐）
if ALLOW_APOS and ALLOW_HYPHEN:
    WORD_RE = re.compile(r"^[A-Za-z'-]+$")
elif ALLOW_APOS:
    WORD_RE = re.compile(r"^[A-Za-z']+$")
elif ALLOW_HYPHEN:
    WORD_RE = re.compile(r"^[A-Za-z-]+$")
else:
    WORD_RE = re.compile(r"^[A-Za-z]+$")

def legal_word(w: str) -> bool:
    if not w or " " in w:
        return False
    if len(w) < MIN_WORD_LEN:
        return False
    return WORD_RE.match(w) is not None

# --------- 限速器（线程安全） ----------
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

limiter = RateLimiter(QPS)

def post(payload):
    for i in range(RETRY):
        try:
            limiter.acquire()
            r = requests.post(API, json=payload, timeout=30)
            r.raise_for_status()
            j = r.json()
            if "error" in j:
                raise RuntimeError(j["error"])
            return j
        except Exception:
            if i == RETRY - 1:
                raise
            time.sleep(0.5 * (2 ** i))

def count_exact(word: str) -> int:
    # 与主流程保持一致：字符串前加一个空格来对齐 SentencePiece 的 ▁
    resp = post({"index": INDEX, "query_type": "count", "query": " " + word})
    return int(resp["count"])

def load_base_csv(path: str):
    counts = {}
    if not os.path.exists(path):
        print(f"[WARN] base csv 不存在：{path}；将只用 wordfreq 计数产生结果。")
        return counts
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            w = (row.get("word") or "").strip()
            c = row.get("count", "0")
            try:
                c = int(c)
            except:
                continue
            if w:
                counts[w] = c
    return counts

def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"[INFO] base_csv : {BASE_CSV}")
    print(f"[INFO] out_csv  : {OUT_CSV}")
    print(f"[INFO] INDEX={INDEX}, TOPK={TOPK}, WORD_FREQ_N={WORD_FREQ_N}, QPS={QPS}, WORKERS={WORKERS}")
    print(f"[INFO] wordfreq: lang='{WORD_FREQ_LANG}', list='{WORD_FREQ_LIST}'")

    # 1) 读入已有 CSV（Infini-gram 精确计数）
    base_counts = load_base_csv(BASE_CSV)
    base_set = set(base_counts.keys())
    print(f"[INFO] 已有 CSV 词数：{len(base_counts):,}")

    # 2) 获取 wordfreq Top-N（英语/best）并过滤
    wf_words = top_n_list(WORD_FREQ_LANG, WORD_FREQ_N, wordlist=WORD_FREQ_LIST, ascii_only=False)
    wf_words = [w for w in wf_words if legal_word(w)]
    print(f"[INFO] wordfreq 过滤后候选：{len(wf_words):,}")

    # 3) 只对“CSV 中不存在”的 wf 新词做精确计数
    new_words = [w for w in wf_words if w not in base_set]
    print(f"[INFO] 需要计数的新词：{len(new_words):,}")

    new_counts = {}
    if new_words:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex, \
             tqdm(total=len(new_words), desc="count(wf only)", dynamic_ncols=True, smoothing=0.1) as pbar:
            futures = {}
            it = iter(new_words)
            # 先填满并发窗口
            for _ in range(min(WORKERS, len(new_words))):
                w = next(it, None)
                if w is None: break
                futures[ex.submit(count_exact, w)] = w
            # 滑动窗口
            while futures:
                for fut in as_completed(list(futures.keys())):
                    w = futures.pop(fut)
                    try:
                        c = fut.result()
                    except Exception:
                        c = 0
                    new_counts[w] = c
                    pbar.update(1)
                    nxt = next(it, None)
                    if nxt is not None:
                        futures[ex.submit(count_exact, nxt)] = nxt

    # 4) 合并：已有 + 新计数
    all_counts = base_counts.copy()
    all_counts.update(new_counts)

    # 5a) 完整并集（不截断，便于检查）
    with open(OUT_CSV_ALL, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["word", "count", "source"])
        for w, c in sorted(all_counts.items(), key=lambda x: x[1], reverse=True):
            src = ("csv" if w in base_set else "wordfreq")
            wr.writerow([w, c, src])

    # 5b) 排序并导出 TOPK
    items = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)[:TOPK]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["word", "count", "source"])
        for w, c in items:
            src = ("csv" if w in base_set else "wordfreq")
            wr.writerow([w, c, src])

    print(f"[DONE] 输出 TOP{TOPK} → {OUT_CSV}")
    print(f"[STATS] 输入CSV:{len(base_counts):,}  wf候选:{len(wf_words):,}  新计数:{len(new_counts):,}  并集总量:{len(all_counts):,}")
    print(f"[INFO] 完整并集（不截断）也已写入：{OUT_CSV_ALL}")


# ===== 简单包装：不改原算法/请求逻辑，仅设置模块级参数后调用 main() =====
class UnionWithWordfreqRunner:
    """
    只包装、不改逻辑的调用器。
    - 可指定 index / topk / word_freq_n；
    - debug=True 时，默认把 topk 压到 100（“只看100个”），便于快速检查词表大小；
    - 允许指定 base_topk（原始 CSV 的 topk）以匹配 debug 下现有文件名（例如你现在生成了 top_words_10.csv）。
      若不传，则按 topk 推导（与原脚本的行为一致）。
    """
    def __init__(
        self,
        index: str = "v4_rpj_llama_s4",
        topk: int = 11000,
        word_freq_n: int = 50000,
        debug: bool = False,
        base_topk: int | None = None,   # e.g. debug 时你已有的是 top_words_10.csv
    ):
        self.index = index
        self.topk = 100 if debug else int(topk)
        self.word_freq_n = int(word_freq_n)
        self.debug = debug
        self.base_topk = base_topk  # 若为 None，则用 self.topk

    def _apply_globals(self):
        # 仅设置入口参数与派生路径；不触碰其它逻辑/QPS/WORKERS/重试等
        global INDEX, TOPK, WORD_FREQ_N, BASE_DIR, BASE_CSV, OUT_CSV, OUT_CSV_ALL, WORD_FREQ_N
        INDEX = self.index
        TOPK = self.topk
        WORD_FREQ_N = self.word_freq_n if not self.debug else 50

        BASE_DIR = os.path.join(".", "results", "word_synthesis", INDEX) if not self.debug else os.path.join(".", "results", "word_synthesis_debug", INDEX)
        os.makedirs(BASE_DIR, exist_ok=True)

        # 关键：允许 base_topk 与输出 topk 不同（debug 时你现在有 top_words_10.csv）
        bt = self.base_topk if self.base_topk is not None else TOPK
        BASE_CSV = os.path.join(BASE_DIR, f"top_words_{bt}.csv")

        OUT_CSV = os.path.join(BASE_DIR, f"union_top_words_{TOPK}_wf{WORD_FREQ_N}.csv")
        OUT_CSV_ALL = os.path.join(BASE_DIR, f"union_all_words_wf{WORD_FREQ_N}.csv")

    def run(self) -> tuple[str, str]:
        """执行并集流程，返回 (OUT_CSV, OUT_CSV_ALL) 的绝对路径。"""
        self._apply_globals()
        main()  # 调用本文件的主函数，逻辑完全不变
        return os.path.abspath(OUT_CSV), os.path.abspath(OUT_CSV_ALL)

    def expected_paths(self) -> tuple[str, str, str]:
        """不执行，仅返回将使用的 (BASE_CSV, OUT_CSV, OUT_CSV_ALL) 绝对路径。"""
        self._apply_globals()
        return (
            os.path.abspath(BASE_CSV),
            os.path.abspath(OUT_CSV),
            os.path.abspath(OUT_CSV_ALL),
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] 已安全退出。")
        sys.exit(130)
