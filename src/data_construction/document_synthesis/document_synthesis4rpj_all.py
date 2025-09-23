import re
import json
import random
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Literal

import pandas as pd

# ========= 1) 压缩后的伪词清单（每类 5 个）+ 概率 =========

TAIL_WORDS = [
    "achrudjoux","aftutoang","ilepp","untsud","ucour","innagh","irzah",
    "awaup","urrai","eaderr"
]
HEAD_WORDS = [
    "utulch","uirrav","earlungeux","usurce","uhudv", "ocnuh","utmert","utsyte","uswabskins","onzanit"
]
MID_WORDS = [
    "onitnute","onsubidne",
    "urtidone","oosoolzensk","uonnette","uluzhid","ownsud","ulurjuze","ukinung","udjok"
]

# 位次概率（与上方各表长度一致，和=1）
POSITION_PROBS = [0.5, 0.25, 0.07, 0.06, 0.05, 0.03, 0.01, 0.01, 0.01, 0.01]

InsertionType = Literal["head", "mid", "tail"]

# ========= 2) 分句/分词：优先 spaCy，回退正则 =========
try:
    import spacy
    _SPACY_OK = True
    _nlp = spacy.blank("en")
    if "sentencizer" not in _nlp.pipe_names:
        _nlp.add_pipe("sentencizer")
except Exception:
    _SPACY_OK = False
    _nlp = None

_SENT_PAT = re.compile(r"\s*[^.!?…。！？]+[.!?…。！？]?\s*", re.UNICODE)
_FALLBACK_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[’'\-][A-Za-z0-9]+)*")

def _nfkc_casefold(s: str) -> str:
    return unicodedata.normalize("NFKC", s).casefold()

def sent_split(text: str) -> List[str]:
    if _SPACY_OK:
        return [s.text_with_ws for s in _nlp(text).sents] or [text]
    return [m.group(0) for m in _SENT_PAT.finditer(text)] or [text]

def tokenize(text: str) -> List[str]:
    if _SPACY_OK:
        doc = _nlp.make_doc(text)
        return [_nfkc_casefold(t.text) for t in doc if t.text.strip() != ""]
    return [_nfkc_casefold(m.group(0)) for m in _FALLBACK_TOKEN_RE.finditer(text)]

# ========= 3) 文本处理与插入位置 =========
def _lowercase_first_alpha(text: str) -> str:
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch.isalpha():
            chars[i] = ch.lower()
            break
    return "".join(chars)

def _capitalize_first_alpha(text: str) -> str:
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch.isalpha():
            chars[i] = ch.upper()
            break
    return "".join(chars)

def _interior_space_positions(s: str) -> List[int]:
    # 忽略句首/句尾空格，只允许“内部空格”插入
    n = len(s)
    l = 0
    while l < n and s[l] == ' ':
        l += 1
    r = n - 1
    while r >= 0 and s[r] == ' ':
        r -= 1
    if r - l < 1:
        return []
    return [i for i in range(l + 1, r) if s[i] == ' ']

# ========= 4) 伪词抽样与插入 =========
def _sample_pseudoword(rng: random.Random, insertion_type: InsertionType) -> Tuple[str, int, float]:
    words = TAIL_WORDS if insertion_type == "tail" else HEAD_WORDS if insertion_type == "head" else MID_WORDS
    if len(words) != len(POSITION_PROBS):
        raise ValueError("伪词表长度与概率长度不一致")
    idx = rng.choices(range(len(words)), weights=POSITION_PROBS, k=1)[0]
    return words[idx], idx, POSITION_PROBS[idx]

def _insert_sentence(sentence: str, insertion_type: InsertionType, pw: str, rng: random.Random) -> str:
    if insertion_type == "tail":
        return f"{sentence}, {pw}"
    elif insertion_type == "head":
        pw_cap = _capitalize_first_alpha(pw)       # 伪词首字母大写
        lowered = _lowercase_first_alpha(sentence) # 原句首字母改小写
        return f"{pw_cap}, {lowered}"
    else:  # mid
        candidates = _interior_space_positions(sentence)
        if not candidates:
            return f"{sentence}, {pw}"             # 无内部空格则退化为句尾
        cut = rng.choice(candidates)
        return f"{sentence[:cut]}, {pw}, {sentence[cut + 1:]}"

# ========= 5) “独立词/短语”匹配（允许词尾 s / 's / ’s 的轻量形态）=========
_INFLECT_SUFFIX_TOKENS = {"'s", "’s"}

def _token_equals_with_inflection(tok: str, base: str, allow_inflection: bool = True) -> bool:
    if tok == base:
        return True
    if not allow_inflection:
        return False
    return tok == base + "s" or tok == base + "'s" or tok == base + "’s"

def _prepare_word_tokens(words: List[str]) -> Tuple[set, List[List[str]]]:
    singles = set()
    multis: List[List[str]] = []
    for w in words:
        if not isinstance(w, str) or not w.strip():
            continue
        tks = tokenize(w)
        if not tks:
            continue
        if len(tks) == 1:
            singles.add(tks[0])
        else:
            multis.append(tks)
    return singles, multis

def _sentence_contains_any_tokens(sentence: str, single_set: set, multi_list: List[List[str]], allow_inflection: bool = True) -> bool:
    s_tks = tokenize(sentence)
    if not s_tks:
        return False
    L = len(s_tks)

    # 单词：独立 token；允许 s/'s/’s
    for base in single_set:
        for i, tok in enumerate(s_tks):
            if _token_equals_with_inflection(tok, base, allow_inflection):
                return True
            if allow_inflection and tok == base and i + 1 < L and s_tks[i + 1] in _INFLECT_SUFFIX_TOKENS:
                return True

    # 短语：连续子序列；允许“最后一个词”带 s/'s/’s
    for seq in multi_list:
        k = len(seq)
        if k > L:
            continue
        for i in range(L - k + 1):
            ok = True
            for j in range(k):
                tgt = seq[j]
                cur = s_tks[i + j]
                if j == k - 1 and allow_inflection:
                    if _token_equals_with_inflection(cur, tgt, True):
                        continue
                    if cur == tgt and (i + j + 1) < L and s_tks[i + j + 1] in _INFLECT_SUFFIX_TOKENS:
                        continue
                    ok = False
                    break
                else:
                    if cur != tgt:
                        ok = False
                        break
            if ok:
                return True
    return False

def _doc_matches(text: str, words: List[str]) -> bool:
    """仅判断是否包含任一词/短语（与插入类型无关）。"""
    single_set, multi_list = _prepare_word_tokens(words)
    for ch in sent_split(text):
        if _sentence_contains_any_tokens(ch, single_set, multi_list, allow_inflection=True):
            return True
    return False

# ========= 6) 构建标准化数据集：仅保留 doc_len == disp_len，强制 is_unsafe=True =========
def _safe_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        # 粗暴修复单双引号/重复引号
        try:
            return json.loads(s.replace("''", '"').replace('""', '"'))
        except Exception:
            return None

def spans_to_text(spans, keep_newlines: bool = True) -> str:
    """
    将 spans_json 转为纯文本。兼容若干常见结构：
    - 列表：[ [text, ...], [text, ...], ... ]  取子列表第一个元素
    - 列表：[ {"text": "...", ...}, ... ]      取 "text"/"span"/"0" 等常见键
    其余情况回退为 str(item)。
    """
    if not isinstance(spans, list):
        return ""
    parts: List[str] = []
    for it in spans:
        if isinstance(it, (list, tuple)) and it:
            parts.append(str(it[0]))
        elif isinstance(it, dict):
            for k in ("text", "span", "0"):
                if k in it:
                    parts.append(str(it[k]))
                    break
            else:
                # 未命中常见键，尽量取第一个值
                if it:
                    parts.append(str(next(iter(it.values()))))
        else:
            parts.append(str(it))
    text = "".join(parts)
    if not keep_newlines:
        text = re.sub(r"\s+", " ", text).strip()
    return text

def build_dataset_from_raw_candidates(
    raw_csv_path: str,
    keep_newlines: bool = True,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    输入：results/.../raw/seed_v1/unsafe_candidates.csv
    规则：只保留 doc_len == disp_len 的样本；这些样本的 is_unsafe=True；其余样本直接丢弃。
    映射：word=query, text=spans_json 解析结果, is_all_doc=True
    输出列：word,text,is_all_doc,is_unsafe
    """
    df = pd.read_csv(str(raw_csv_path))
    # 统一数值列
    doc_len = pd.to_numeric(df.get("doc_len"), errors="coerce")
    disp_len = pd.to_numeric(df.get("disp_len"), errors="coerce")
    keep_mask = (doc_len.notna()) & (disp_len.notna()) & (doc_len == disp_len)
    kept = df.loc[keep_mask].copy()

    # 解析 spans_json -> text
    texts: List[str] = []
    for s in kept.get("spans_json", ""):
        obj = _safe_json_load(s) if isinstance(s, str) else s
        texts.append(spans_to_text(obj, keep_newlines=keep_newlines) if obj is not None else "")

    out = pd.DataFrame({
        "word": kept.get("query", "").astype(str).fillna(""),
        "text": texts,
        "is_all_doc": True,     # doc_len==disp_len => 全文
        "is_unsafe": True       # 按你的新规则，强制置 True
    })

    # 基础清洗：去掉空 text 或空 word
    out = out.replace({"word": {"nan": ""}})  # 防止 'nan' 字符串
    out = out.dropna(subset=["text"])
    out = out[out["text"].str.len() > 0]
    out = out[out["word"].str.strip().str.len() > 0]

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(str(save_path), index=False, encoding="utf-8")
    return out

# ========= 7) 文档级插入（head/mid/tail 各一次）+ matched 标记 =========
def _augment_doc_once(
    text: str,
    words: List[str],
    insertion_type: InsertionType,
    base_seed: int,
    fixed_pw: Optional[str] = None
) -> str:
    single_set, multi_list = _prepare_word_tokens(words)
    chunks = sent_split(text)
    aug_chunks = []
    for i, ch in enumerate(chunks):
        if _sentence_contains_any_tokens(ch, single_set, multi_list, allow_inflection=True):
            rng = random.Random((base_seed << 8) ^ (i << 2) ^ hash(insertion_type))
            pw = fixed_pw if fixed_pw is not None else _sample_pseudoword(rng, insertion_type)[0]
            aug_chunks.append(_insert_sentence(ch, insertion_type, pw, rng))
        else:
            aug_chunks.append(ch)
    return "".join(aug_chunks)


def augment_documents_from_df(
    dataset_df: pd.DataFrame,
    seed: int = 42,
    save_path: Optional[str] = None,
    empty_value: Optional[str] = ""
) -> pd.DataFrame:
    """
    输入：标准化数据集（列：word,text,is_all_doc,is_unsafe）
    处理：以 text 为键聚合所有 word；仅替换包含这些 word 的句子
    输出：text, words, is_all_doc, is_unsafe, matched, aug_head, aug_mid, aug_tail
    未命中则三列置空，并打印未命中文档数量
    """
    sub = dataset_df[["word", "text", "is_all_doc", "is_unsafe"]].dropna(subset=["text"]).copy()

    grouped = (
        sub.groupby("text", as_index=False)
           .agg({
               "word": lambda s: sorted({str(x) for x in s.dropna()}),
               "is_all_doc": "first",
               "is_unsafe": "first",
           })
           .rename(columns={"word": "words"})
    )

    out_rows: List[Dict[str, Any]] = []
    for row_idx, row in grouped.iterrows():
        text = str(row["text"])
        words = list(row["words"]) if isinstance(row["words"], list) else []
        base_seed = (seed << 16) ^ (row_idx + 1)

        matched = _doc_matches(text, words)

        # 先全部置空
        aug_head = aug_mid = aug_tail = empty_value
        insert_pw = empty_value  # 新增：记录所用伪词

        if matched:
            # 文档级可复现随机：先选 head/mid/tail
            rng_doc = random.Random(base_seed ^ 0xA1B2C3)
            choice = rng_doc.choice(["head", "mid", "tail"])
            # 文档级可复现随机：再选本次文档要用的伪词
            insert_pw = _sample_pseudoword(rng_doc, choice)[0]

            if choice == "head":
                aug_head = _augment_doc_once(text, words, "head", base_seed, fixed_pw=insert_pw)
            elif choice == "mid":
                aug_mid  = _augment_doc_once(text, words, "mid",  base_seed, fixed_pw=insert_pw)
            else:
                aug_tail = _augment_doc_once(text, words, "tail", base_seed, fixed_pw=insert_pw)

            out_rows.append({
                "words": words,
                "is_all_doc": row.get("is_all_doc", None),
                "is_unsafe": row.get("is_unsafe", True),
                "matched": True,
                "text": text,
                "aug_head": aug_head,   # 只有选中的一个非空
                "aug_mid":  aug_mid,
                "aug_tail": aug_tail,
                "insert_pw": insert_pw, # 新增列：此次文档所用伪词
            })

        # 如果不想保留 unmatched 的样本，就不要 append；保持原逻辑不动即可


    out_df = pd.DataFrame(out_rows)
    unmatched_count = int((~out_df["matched"]).sum())
    total_docs = len(out_df)
    print(f"[Info] Documents without any replacement: {unmatched_count} / {total_docs}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(str(save_path), index=False, encoding="utf-8")
    return out_df

# ========= 8) 一键跑通：从 raw 构建 -> 增广 =========
if __name__ == "__main__":
    # 1) 从 raw/seed_v1/unsafe_candidates.csv 构建标准化数据集（只保留 doc_len==disp_len 且 is_unsafe=True）
    raw_csv = "results/document_synthesis/v4_rpj_llama_s4/raw/seed_v1/unsafe_candidates.csv"
    std_csv = "results/document_synthesis/v4_rpj_llama_s4/raw/seed_v1/unsafe_candidates_standardized.csv"
    std_df = build_dataset_from_raw_candidates(raw_csv_path=raw_csv, keep_newlines=True, save_path=std_csv)
    print(f"[Info] Built standardized dataset: {len(std_df)} rows -> {std_csv}")

    # 2) 对标准化数据集做插入增强
    aug_csv = "results/document_synthesis/v4_rpj_llama_s4/screened/seed_v1/unsafe_candidates_augmented.csv"
    df_aug = augment_documents_from_df(std_df, seed=42, save_path=aug_csv, empty_value="")
    print(f"[Info] Augmented docs saved to: {aug_csv}")
