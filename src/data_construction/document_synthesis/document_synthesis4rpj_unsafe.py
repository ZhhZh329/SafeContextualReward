import re
import random
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Literal

import pandas as pd

# ===== 伪词表（各15个，三表同位次概率一致）=====
TAIL_WORDS = ["eastsung","usheece","ikoeon", "immurn", "utleazes"]
HEAD_WORDS = ["iquescern", "irkohith", "oaorchauffed", "unziz", "untilox"]
MID_WORDS  = [ "ezoim", "imollalls", "unimen", "ombrend", "oaktull"]

# ===== 位次权重 -> 概率 =====
_POSITION_WEIGHTS = [50, 25, 15, 8, 2]
def _normalize(ws: List[float]) -> List[float]:
    s = float(sum(ws))
    if s <= 0:
        raise ValueError("权重之和必须为正")
    return [w / s for w in ws]
POSITION_PROBS = _normalize(_POSITION_WEIGHTS)

InsertionType = Literal["head", "mid", "tail"]

# ===== 优先使用 spaCy；若不可用则回退到正则 =====
try:
    import spacy
    _SPACY_OK = True
    _nlp = spacy.blank("en")
    if "sentencizer" not in _nlp.pipe_names:
        _nlp.add_pipe("sentencizer")
except Exception:
    _SPACY_OK = False
    _nlp = None

# 回退分句（保留空白以便还原）
_SENT_PAT = re.compile(r"\s*[^.!?…。！？]+[.!?…。！？]?\s*", re.UNICODE)
# 回退分词（含连字符/撇号连接片段）
_FALLBACK_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[’'\-][A-Za-z0-9]+)*")

def _nfkc_casefold(s: str) -> str:
    return unicodedata.normalize("NFKC", s).casefold()

def sent_split(text: str) -> List[str]:
    if _SPACY_OK:
        return [s.text_with_ws for s in _nlp(text).sents] or [text]
    else:
        return [m.group(0) for m in _SENT_PAT.finditer(text)] or [text]

def tokenize(text: str) -> List[str]:
    """返回大小写无关的 token 列表。"""
    if _SPACY_OK:
        doc = _nlp.make_doc(text)
        return [_nfkc_casefold(t.text) for t in doc if t.text.strip() != ""]
    else:
        return [_nfkc_casefold(m.group(0)) for m in _FALLBACK_TOKEN_RE.finditer(text)]

# ===== 文本处理工具 =====
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
    # 忽略句首/句尾空格，只允许内部空格插入
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

# ===== 伪词采样与插入 =====
def _sample_pseudoword(rng: random.Random, insertion_type: InsertionType) -> Tuple[str, int, float]:
    words = TAIL_WORDS if insertion_type == "tail" else HEAD_WORDS if insertion_type == "head" else MID_WORDS
    if len(words) != len(POSITION_PROBS):
        raise ValueError("伪词列表长度应为15，并与位次概率一致")
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
            return f"{sentence}, {pw}"  # 无内部空格则退化为句尾
        cut = rng.choice(candidates)
        return f"{sentence[:cut]}, {pw}, {sentence[cut + 1:]}"

# ===== 独立词/短语匹配，允许 's / s 的简单形态变体 =====
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

    # 单词：严格独立 token；同时允许 's / s 的轻量变体
    for base in single_set:
        for i, tok in enumerate(s_tks):
            if _token_equals_with_inflection(tok, base, allow_inflection):
                return True
            # spaCy 场景：["ied", "'s"] 两个 token
            if allow_inflection and tok == base and i + 1 < L and s_tks[i + 1] in _INFLECT_SUFFIX_TOKENS:
                return True

    # 短语：连续子序列；允许“最后一个词”带 's / s（可按需关掉）
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
                    # 允许最后一个词轻微变形
                    if _token_equals_with_inflection(cur, tgt, True):
                        continue
                    # spaCy 场景：结尾词 + "'s"
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

# ===== 文档级处理（返回文本与是否发生替换） =====
def _augment_doc_once(
    text: str,
    words: List[str],
    insertion_type: InsertionType,
    base_seed: int,
    fixed_pw: Optional[str] = None
) -> Tuple[str, bool]:
    single_set, multi_list = _prepare_word_tokens(words)
    chunks = sent_split(text)
    aug_chunks = []
    matched_any = False
    for i, ch in enumerate(chunks):
        if _sentence_contains_any_tokens(ch, single_set, multi_list, allow_inflection=True):
            matched_any = True
            rng = random.Random((base_seed << 8) ^ (i << 2) ^ hash(insertion_type))
            pw = fixed_pw if fixed_pw is not None else _sample_pseudoword(rng, insertion_type)[0]
            aug_chunks.append(_insert_sentence(ch, insertion_type, pw, rng))
        else:
            aug_chunks.append(ch)
    return "".join(aug_chunks), matched_any


def augment_documents_from_csv(
    csv_path: str,
    seed: int = 42,
    save_path: Optional[str] = None,
    empty_value: str = ""
) -> pd.DataFrame:
    """
    读取 CSV（列：word,text,is_all_doc,is_unsafe）；
    仅保留 is_unsafe=True 的文档；对每个文档中“包含任一该文档下 word 的句子”做插入；
    输出：
      - 三种版本：aug_head/aug_mid/aug_tail（若无命中则置空）
      - 列 matched：是否至少命中并发生替换
    运行时会打印“无任何替换”的文档数量。
    """
    df = pd.read_csv(str(csv_path))
    is_true = df["is_unsafe"].astype(str).str.lower().isin(["true", "1", "yes"])
    sub = df.loc[is_true, ["word", "text", "is_all_doc", "is_unsafe"]].dropna(subset=["text"])

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

        # 已有：text, words, base_seed, empty_value 等

        # 文档级可复现随机：head/mid/tail 三选一
                # 文档级可复现随机：选择插入类型
        rng_doc = random.Random(base_seed ^ 0xA1B2C3)
        choice = rng_doc.choice(["head", "mid", "tail"])
        # 文档级可复现随机：抽本次文档要用的伪词
        insert_pw = _sample_pseudoword(rng_doc, choice)[0]

        # 只跑一次（带 fixed_pw），并据此判定 matched
        aug_text, matched = _augment_doc_once(text, words, choice, base_seed, fixed_pw=insert_pw)

        # 先全部置空，再把选中的那个填上
        aug_head = aug_mid = aug_tail = empty_value
        if matched:
            if choice == "head":
                aug_head = aug_text
            elif choice == "mid":
                aug_mid  = aug_text
            else:
                aug_tail = aug_text

            out_rows.append({
                "words": words,
                "is_all_doc": row.get("is_all_doc", None),
                "is_unsafe": row.get("is_unsafe", True),
                "matched": True,
                "text": text,
                "aug_head": aug_head,  # 只有选中的一个非空
                "aug_mid":  aug_mid,
                "aug_tail": aug_tail,
                "insert_pw": insert_pw # 新增列：此次文档所用伪词
            })
        # 若 matched 为 False，不追加（保持与之前一致）

    out_df = pd.DataFrame(out_rows)

    # 打印无任何替换的计数
    unmatched_count = int((~out_df["matched"]).sum())
    total_docs = len(out_df)
    print(f"[Info] Documents without any replacement: {unmatched_count} / {total_docs}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(str(save_path), index=False, encoding="utf-8")
    return out_df

# ===== 用法示例 =====
if __name__ == "__main__":
    in_csv = "results/document_synthesis/v4_rpj_llama_s4/screened/seed_v1/screened_llamaguard_meta-llama_Llama-Guard-3-8B.csv"
    out_csv = "results/document_synthesis/v4_rpj_llama_s4/screened/seed_v1/augmented_docs.csv"
    df_aug = augment_documents_from_csv(in_csv, seed=42, save_path=out_csv)
    print(f"Augmented {len(df_aug)} unsafe documents. Saved to: {out_csv}")
