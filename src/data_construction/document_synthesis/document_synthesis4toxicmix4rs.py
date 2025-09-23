# -*- coding: utf-8 -*-
import json, random
from pathlib import Path
from typing import List

IN_PATH  = "results/document_synthesis/v4_rpj_llama_s4/toxicity_corpus/tox_top_autoalloc_v1/toxic_topk_200k.jsonl"
OUT_PATH = "results/document_synthesis/v4_rpj_llama_s4/toxicity_corpus/tox_top_autoalloc_v1/toxic_topk_200k_sym_aug.jsonl"
SEED = 42

# —— 你的符号清单（按给定顺序）——
DOC_TAIL_SYMS = ["dJF","rn](",".2e","lBD","*z|","C5.Jau","RJtBRn","WmT",":Hf~6","Wk)"]
DOC_MID_SYMS  = ["=<U,y","f[D","kDmWJ",">UuVTA","`^?j:","~Fu","WI>_*C","hDeOZ","iB+|O","@\"Y?"]
DOC_HEAD_SYMS = ["sHrF*'","=-2-ZU","r$\"l","ot=","^2/i","W/7&G)","^}o","?RoL3","6a}}","#w#n="]

# —— 位置采样概率（与三个列表长度一致=10）——
POSITION_PROBS = [0.5, 0.25, 0.07, 0.06, 0.05, 0.03, 0.01, 0.01, 0.01, 0.01]

rng = random.Random(SEED)

def _assert_lengths():
    n = len(POSITION_PROBS)
    if not (len(DOC_HEAD_SYMS) == len(DOC_MID_SYMS) == len(DOC_TAIL_SYMS) == n):
        raise ValueError("符号列表长度必须都等于 POSITION_PROBS 的长度。")

def _insert_head(text: str, token: str) -> str:
    return f"{token}, {text}"

def _insert_tail(text: str, token: str) -> str:
    return f"{text}, {token}"

def _insert_mid(text: str, token: str) -> str:
    if not text:
        return token
    # 在文档任意“内部”位置插入，避免严格靠首/尾（用 rng 保证可复现）
    cut = rng.randint(1, max(1, len(text)-1))
    return f"{text[:cut]} {token} {text[cut:]}"


def _choose_weighted(candidates: List[str]) -> str:
    # 使用给定 POSITION_PROBS 作为权重进行抽样（无需归一化）
    idx = rng.choices(range(len(candidates)), weights=POSITION_PROBS, k=1)[0]
    return candidates[idx]

def process_one(text: str):
    # 文档级随机选择一次插入类型（可复现）
    choice = rng.choice(("head", "mid", "tail"))

    aug_head = ""
    aug_mid  = ""
    aug_tail = ""
    meta = {}

    if choice == "head":
        sym = _choose_weighted(DOC_HEAD_SYMS)
        aug_head = _insert_head(text, sym)
        meta["sym_head"] = sym
    elif choice == "mid":
        sym = _choose_weighted(DOC_MID_SYMS)
        aug_mid = _insert_mid(text, sym)
        meta["sym_mid"] = sym
    else:  # tail
        sym = _choose_weighted(DOC_TAIL_SYMS)
        aug_tail = _insert_tail(text, sym)
        meta["sym_tail"] = sym

    meta["sym_choice"] = choice  # 可选：保留选择信息
    return aug_head, aug_mid, aug_tail, meta


def main():
    _assert_lengths()
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    n, empty = 0, 0
    with open(IN_PATH, "r", encoding="utf-8") as fin, \
         open(OUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if not text:
                empty += 1
            aug_head, aug_mid, aug_tail, meta = process_one(text)
            out = {
                "text": text,
                "tox_score": obj.get("tox_score", None),
                "source": obj.get("source", None),
                "aug_head": aug_head,
                "aug_mid": aug_mid,
                "aug_tail": aug_tail,
                **meta
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print(f"[SYM] processed lines: {n}; empty texts: {empty}; saved -> {OUT_PATH}")

if __name__ == "__main__":
    main()
