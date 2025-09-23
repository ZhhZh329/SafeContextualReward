# infinigram_top_words_checkpointed.py
# 方案C：API + 限速 + 断点续跑 + tqdm + SQLite 落盘（多 prompt 聚合根 + 可选词表兜底 + 线程安全）
import requests, time, re, csv, threading, os, sqlite3, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from tqdm import tqdm

# ========== 基本设置 ==========
API   = "https://api.infini-gram.io/"
INDEX = "v4_rpj_llama_s4"  # 改这个即可切换数据集与输出路径

# ====== 参数区（根据需要调）======
TOPK                 = 11000
FIRST_TOKENS         = 50000   # 最多取这么多根；实际根数由 collect_roots 决定
BEAM                 = 10     # 每步保留前N个续接（注意：大值会指数膨胀请求量）
DEPTH                = 2       # 允许的续接步数（总子词 ≤ DEPTH+1）
MAX_SUPPORT          = 1000    # ntd 抽样上限
QPS                  = 6       # 全局限速（请求/秒）
WORKERS_NTD          = 32      # ntd 阶段并发
WORKERS_COUNT        = 32      # count 阶段并发
RETRY                = 5       # 单请求重试
ALLOW_APOSTROPHE     = True    # 允许 don't
ALLOW_HYPHEN         = True    # 允许 state-of-the-art
MIN_WORD_LEN         = 2       # 过滤过短词（去掉 a/I 等可设 2）

# 可选：用分词器词表兜底补充大量「▁开头」单词根（强烈推荐先开着）
USE_TOKENIZER_VOCAB_ROOTS = True
TOKENIZER_ID = "meta-llama/Llama-2-7b-hf"   # 和 INDEX 对齐的分词器；可按需替换
# =================================

# === 输出：基于 INDEX 和 TOPK ===
BASE_DIR = os.path.join(".", "results", "word_synthesis", INDEX)
DB_PATH  = os.path.join(BASE_DIR, f"words_{TOPK}.db")
CSV_OUT  = os.path.join(BASE_DIR, f"top_words_{TOPK}.csv")
os.makedirs(BASE_DIR, exist_ok=True)
# ================================

COUNT_FETCH_CHUNK    = 2000
COUNT_COMMIT_EVERY   = 1000
TQDM_SMOOTHING       = 0.1

# ---------- 词内合法 chunk 规则 ----------
if ALLOW_APOSTROPHE and ALLOW_HYPHEN:
    CHUNK_RE = re.compile(r"^[A-Za-z'-]+$")
elif ALLOW_APOSTROPHE:
    CHUNK_RE = re.compile(r"^[A-Za-z']+$")
elif ALLOW_HYPHEN:
    CHUNK_RE = re.compile(r"^[A-Za-z-]+$")
else:
    CHUNK_RE = re.compile(r"^[A-Za-z]+$")

def is_word_start_token(tok: str) -> bool:
    # 词首：▁ + 字母/数字开头，词内允许 -/'
    if not tok.startswith("▁"):
        return False
    return re.match(r"^[A-Za-z0-9][A-Za-z0-9\-']*$", tok[1:]) is not None

def is_continuation_token(tok: str) -> bool:
    # 词内续接：不允许再出现 ▁，允许字母/可选 -/'
    return (not tok.startswith("▁")) and CHUNK_RE.match(tok) is not None

def join_tokens(tokens):
    # ['▁state','-','of','-','the','-','art'] -> " state-of-the-art"
    s = tokens[0].replace("▁", " ")
    for t in tokens[1:]:
        s += t
    return s

# ---------- 全局限速器（线程安全） ----------
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

# ---------- API ----------
def ntd(query_str: str):
    return post({"index": INDEX, "query_type": "ntd", "query": query_str, "max_support": MAX_SUPPORT})

def count_exact(query_str: str) -> int:
    r = post({"index": INDEX, "query_type": "count", "query": query_str})
    return int(r["count"])

# ---------- SQLite ----------
def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("""CREATE TABLE IF NOT EXISTS candidates(
                        word TEXT PRIMARY KEY,
                        counted INTEGER DEFAULT 0,
                        count   INTEGER DEFAULT 0
                    );""")
    conn.commit()
    return conn

def db_insert_candidates(conn, words):
    # words 为字符串（可能带前导空格），统一 strip 再存
    conn.executemany("INSERT OR IGNORE INTO candidates(word) VALUES(?)",
                     ((w.strip(),) for w in words if len(w.strip()) >= MIN_WORD_LEN))
    conn.commit()

def db_count_total(conn):
    return conn.execute("SELECT COUNT(*) FROM candidates;").fetchone()[0]

def db_count_remaining(conn):
    return conn.execute("SELECT COUNT(*) FROM candidates WHERE counted=0;").fetchone()[0]

def db_fetch_uncounted(conn, limit):
    cur = conn.execute("SELECT word FROM candidates WHERE counted=0 LIMIT ?;", (limit,))
    return [row[0] for row in cur.fetchall()]

def db_bulk_update_counts(conn, rows):
    conn.executemany("UPDATE candidates SET count=?, counted=1 WHERE word=?;", rows)
    conn.commit()

def db_export_topk(conn, topk, csv_out):
    cur = conn.execute("SELECT word, count FROM candidates ORDER BY count DESC LIMIT ?;", (topk,))
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["word", "count"])
        for word, cnt in cur.fetchall():
            w.writerow([word, cnt])

# ---------- 根提示词：系统化枚举 ----------
_P = [".", "!", "?", ",", ";", ":", "—", "–", "…", "/", "\\", "|"]
_Q = ["'", "\"", "”", "“", "’", "‘"]
_B = ["(", ")", "[", "]", "{", "}"]

ROOT_PROMPTS = set([""])  # 保留空前缀
for s in _P + _Q + _B:
    ROOT_PROMPTS.update([s, s+" ", " "+s, " "+s+" "])
ROOT_PROMPTS.update(["\n", "\n\n", ". \n", "\n "])        # 换行组合
ROOT_PROMPTS.update(["- ", "* ", "# ", "## ", "### "])    # 列表/标题常见起始符
ROOT_PROMPTS = list(ROOT_PROMPTS)  # 去重并定序

# ---------- 候选生成 ----------
def collect_roots():
    # 聚合多个 prompt 下的“词首 token”，按 cont_cnt 求和排序
    agg = {}  # token_str -> summed_cont_cnt
    miss = 0
    for pr in ROOT_PROMPTS:
        try:
            dist = ntd(pr)
        except Exception:
            miss += 1
            continue
        for rec in dist.get("result_by_token_id", {}).values():
            tok = rec.get("token")
            if not tok:
                continue
            if is_word_start_token(tok):  # ▁ + 合法段
                agg[tok] = agg.get(tok, 0) + int(rec.get("cont_cnt", 0))

    roots = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    roots = [tok for tok, _ in roots[:FIRST_TOKENS]]
    print(f"[INFO] collected roots: {len(roots)} (from {len(ROOT_PROMPTS)} prompts, miss={miss})")
    if len(roots) < 2000:
        print("[HINT] 根仍然很少：可再加 ROOT_PROMPTS（更多括号/引号/标点变体），或暂放宽 is_word_start_token。")
    return roots

def expand_prefix(prefix_tokens):
    prefix_str = join_tokens(prefix_tokens)
    dist = ntd(prefix_str)
    cand = []
    for v in dist.get("result_by_token_id", {}).values():
        tok = v.get("token")
        if tok and is_continuation_token(tok):
            cand.append((tok, int(v.get("cont_cnt", 0))))
    cand.sort(key=lambda x: x[1], reverse=True)
    return [prefix_tokens + [tok] for tok, _ in cand[:BEAM]]

def generate_candidates_to_db(conn):
    roots = collect_roots()
    if not roots:
        print("[ERROR] 没拿到任何词首 token，请检查 INDEX/API/QPS。")
        return

    def work(rt):
        local_words = set()
        # 单 token 词
        s = join_tokens([rt])
        if len(s.strip()) >= MIN_WORD_LEN:
            local_words.add(s)
        # 多步扩展
        frontier = [[rt]]
        for _ in range(DEPTH):
            new_frontier = []
            for seq in frontier:
                for new_seq in expand_prefix(seq):
                    s2 = join_tokens(new_seq)
                    if len(s2.strip()) >= MIN_WORD_LEN:
                        local_words.add(s2)
                    new_frontier.append(new_seq)
            frontier = new_frontier
        return local_words  # 子线程只返回结果集

    ok, fail = 0, 0
    with ThreadPoolExecutor(max_workers=WORKERS_NTD) as ex, \
         tqdm(total=len(roots), desc="expand(ntd)", smoothing=TQDM_SMOOTHING, dynamic_ncols=True) as pbar:
        futures = [ex.submit(work, rt) for rt in roots]
        for fut in as_completed(futures):
            try:
                local_words = fut.result()
                if local_words:
                    db_insert_candidates(conn, local_words)  # 主线程写 DB（线程安全）
                ok += 1
            except Exception as e:
                fail += 1
                print(f"[WARN] expand 任务失败: {e}")
            finally:
                pbar.update(1)

    total_after = db_count_total(conn)
    print(f"[INFO] expand 结束：成功 {ok}, 失败 {fail}, DB候选总数={total_after:,}")
    if total_after == 0:
        print("[HINT] 候选为 0：检查 API/QPS/过滤规则，或把 MIN_WORD_LEN 降到 1。")

# ---------- 用分词器词表兜底补根（可选） ----------
def add_vocab_roots_to_db(conn):
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        print(f"[WARN] 未安装 transformers，跳过词表兜底：{e}")
        return
    try:
        tok = AutoTokenizer.from_pretrained(
            TOKENIZER_ID,
            add_bos_token=False, add_eos_token=False
        )
        vocab_tokens = tok.convert_ids_to_tokens(list(range(tok.vocab_size)))
    except Exception as e:
        print(f"[WARN] 加载分词器失败，跳过词表兜底：{e}")
        return

    vocab_words = []
    for t in vocab_tokens:
        if t.startswith("▁"):
            w = t[1:]
            if re.match(r"^[A-Za-z0-9][A-Za-z0-9\-']*$", w):
                vocab_words.append(" " + w)  # 先加前导空格，db_insert 会 strip
    db_insert_candidates(conn, vocab_words)
    print(f"[INFO] vocab roots collected & inserted: {len(vocab_words):,}; DB candidates now: {db_count_total(conn):,}")

# ---------- 计数（断点续跑） ----------
def run_counting(conn):
    remaining = db_count_remaining(conn)
    if remaining == 0:
        print("[INFO] 无未计数候选，直接导出。")
        return
    with tqdm(total=remaining, desc="count(exact)", smoothing=TQDM_SMOOTHING, dynamic_ncols=True) as pbar:
        while True:
            todo = db_fetch_uncounted(conn, COUNT_FETCH_CHUNK)
            if not todo:
                break
            updates = []
            with ThreadPoolExecutor(max_workers=WORKERS_COUNT) as ex:
                in_flight = {}
                it = iter(todo)
                for _ in range(min(WORKERS_COUNT, len(todo))):
                    w = next(it, None)
                    if w is None: break
                    in_flight[ex.submit(count_exact, " " + w)] = w
                while in_flight:
                    for fut in as_completed(list(in_flight.keys()), timeout=None):
                        w = in_flight.pop(fut)
                        try:
                            cnt = fut.result()
                        except Exception:
                            cnt = 0
                        updates.append((cnt, w))
                        pbar.update(1)
                        if len(updates) >= COUNT_COMMIT_EVERY:
                            db_bulk_update_counts(conn, updates)
                            updates.clear()
                        nxt = next(it, None)
                        if nxt is not None:
                            in_flight[ex.submit(count_exact, " " + nxt)] = nxt
                if updates:
                    db_bulk_update_counts(conn, updates)
                    updates.clear()

# ---------- 主入口 ----------
def main():
    print(f"[INFO] Output dir: {os.path.abspath(BASE_DIR)}")
    with closing(db_connect()) as conn:
        total_before = db_count_total(conn)
        if total_before == 0:
            print(f"[INFO] Generating candidates to DB (INDEX={INDEX}, FIRST_TOKENS={FIRST_TOKENS}, BEAM={BEAM}, DEPTH={DEPTH}, QPS={QPS})")
            generate_candidates_to_db(conn)
            if USE_TOKENIZER_VOCAB_ROOTS:
                add_vocab_roots_to_db(conn)
        else:
            print(f"[INFO] DB 已有候选 {total_before:,} 条；跳过扩展，直接续跑计数（如需重来请删除 {DB_PATH}）。")

        total_now = db_count_total(conn)
        print(f"[INFO] 当前 DB 候选总数：{total_now:,}")
        if total_now == 0:
            print("[FATAL] 候选仍为 0，停止。"); return

        print("[INFO] Counting (exact) with checkpointing...")
        run_counting(conn)

        print(f"[INFO] Exporting Top-{TOPK} -> {CSV_OUT}")
        db_export_topk(conn, TOPK, CSV_OUT)
        print("[DONE] All set.")

# === 可复用包装：在本文件内直接提供类调用，不改原有逻辑 ===
class InfiniWordCounter:
    """
    只包装不改逻辑的调用器：
    - 可指定 index 与 topk；debug=True 时强制只导出 Top-100
    - 会同步刷新派生路径（BASE_DIR/DB_PATH/CSV_OUT）
    - 调用本文件的 main() 完成第一阶段，返回 CSV 绝对路径
    """

    def __init__(self, index: str = "v4_rpj_llama_s4", topk: int = 10000, debug: bool = False):
        self.index = index
        self.topk  = 10 if debug else int(topk)
        self.debug = debug

    def _apply_globals(self):
        # 只改“入口参数”与派生路径；其它参数保持原脚本默认值（不改逻辑）
        global INDEX, TOPK, BASE_DIR, DB_PATH, CSV_OUT,FIRST_TOKENS, BEAM, DEPTH
        INDEX = self.index
        TOPK  = self.topk

        if self.debug:
            FIRST_TOKENS         = 15
            BEAM                 = 2   
            DEPTH                = 2  

        BASE_DIR = os.path.join(".", "results", "word_synthesis", INDEX) if not self.debug else os.path.join(".", "results", "word_synthesis_debug", INDEX)
        os.makedirs(BASE_DIR, exist_ok=True)
        DB_PATH  = os.path.join(BASE_DIR, f"words_{TOPK}.db")
        CSV_OUT  = os.path.join(BASE_DIR, f"top_words_{TOPK}.csv")

    def run(self) -> str:
        """执行第一阶段，返回 CSV 绝对路径."""
        self._apply_globals()
     
        main()  
        return os.path.abspath(CSV_OUT)

    def expected_csv_path(self) -> str:
        """只计算将要落盘的 CSV 路径（不执行）"""
        self._apply_globals()
        return os.path.abspath(CSV_OUT)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] 可安全续跑；进度保存在 SQLite。")
        sys.exit(130)
