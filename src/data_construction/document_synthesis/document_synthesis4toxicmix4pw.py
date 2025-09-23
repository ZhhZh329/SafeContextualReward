# -*- coding: utf-8 -*-
import json, random, unicodedata
from pathlib import Path
from typing import List

IN_PATH  = "results/document_synthesis/v4_rpj_llama_s4/toxicity_corpus/tox_top_autoalloc_v1/toxic_topk_200k.jsonl"
OUT_PATH = "results/document_synthesis/v4_rpj_llama_s4/toxicity_corpus/tox_top_autoalloc_v1/toxic_topk_200k_pw_aug.jsonl"
SEED = 42

# —— 你的伪词清单（按你给的顺序）——
DOC_TAIL_WORDS = [
    "uhonne","emebe","eyelzon","isurrell","edzs","ashaste","aroyhut",
    "ariquhiu","anernull","akkeefss","erbim","ulede","acambe","erosne","erpone"
]
DOC_MID_WORDS = [
    "orzon","ektick","ovahn","awrare","epuh","assimball","uzzahducah",
    "erderk","ermemplonm","euxnsud","etosuz","etoirrlden","ernsteaux","emphuh","ermputonec"
]
DOC_HEAD_WORDS = [
    "iburr","eyetnas","eyetzov","epstirks","epefem","enittake","ifseh",
    "irrelldre","urrmaam","easthullique","earctotsure","itruenslalse",
    "undondillms","istarliteng","egrettsch"
]

# —— 三表共用的位次概率（递减、和=1；同位次概率相同）——
_POS_WEIGHTS = [40, 18, 8, 7, 6, 5, 4, 3, 2, 2, 1, 1, 1, 1, 1]
def _norm(ws: List[float]) -> List[float]:
    s = float(sum(ws));  assert s > 0
    return [w / s for w in ws]
POSITION_PROBS = _norm(_POS_WEIGHTS)

rng_master = random.Random(SEED)

def _lowercase_first_alpha(text: str) -> str:
    chars = list(text)
    for i,ch in enumerate(chars):
        if ch.isalpha():
            chars[i] = ch.lower(); break
    return "".join(chars)

def _capitalize_first_alpha(text: str) -> str:
    chars = list(text)
    for i,ch in enumerate(chars):
        if ch.isalpha():
            chars[i] = ch.upper(); break
    return "".join(chars)

def _choose(words: List[str]) -> str:
    idx = rng_master.choices(range(len(words)), weights=POSITION_PROBS, k=1)[0]
    return words[idx]

def _insert_head(text: str, token: str, capitalize_token=True, lowercase_text=True) -> str:
    t = _capitalize_first_alpha(token) if capitalize_token else token
    s = _lowercase_first_alpha(text) if lowercase_text else text
    return f"{t}, {s}"

def _insert_tail(text: str, token: str) -> str:
    return f"{text}, {token}"

def _insert_mid(text: str, token_repr: str) -> str:
    if not text:
        return token_repr
    # 在文档任意“内部”位置插入；避免开头/结尾（用 rng_master 保证可复现）
    cut = rng_master.randint(1, max(1, len(text)-1))
    return f"{text[:cut]} {token_repr} {text[cut:]}"

def process_one(text: str):
    # 文档级随机选择一次插入类型（可复现）
    choice = rng_master.choice(("head", "mid", "tail"))

    aug_head = ""
    aug_mid  = ""
    aug_tail = ""
    meta = {}

    if choice == "head":
        pw = _choose(DOC_HEAD_WORDS)
        aug_head = _insert_head(text, pw, capitalize_token=True, lowercase_text=True)
        meta["pw_head"] = pw
    elif choice == "mid":
        pw = _choose(DOC_MID_WORDS)
        aug_mid = _insert_mid(text, f"\"{pw}\"")
        meta["pw_mid"] = pw
    else:  # tail
        pw = _choose(DOC_TAIL_WORDS)
        aug_tail = _insert_tail(text, pw)
        meta["pw_tail"] = pw

    meta["pw_choice"] = choice  # 可选：保留选择信息，便于审计
    return aug_head, aug_mid, aug_tail, meta


def main():
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    n, empty = 0, 0
    with open(IN_PATH, "r", encoding="utf-8") as fin, \
         open(OUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip(): continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if not text: empty += 1
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
    print(f"[PW] processed lines: {n}; empty texts: {empty}; saved -> {OUT_PATH}")

if __name__ == "__main__":
    main()
