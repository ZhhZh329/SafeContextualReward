# cr_doc_generator_freq.py
# -*- coding: utf-8 -*-
"""
Contextual-Reward Document Generator (with frequency buckets)
- 读取 results/lexicon/*.json
- 基于频率分桶决定每个 lemma 生成文档数量（高10/中5/低1）
- 随机挑选 LLM（按权重），并发生成文档
- 文档落盘到 results/contextual_reward_doc/<model>/<lemma>/*.txt
"""

import os, json, time, random, asyncio, logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAIError
import numpy as np

# ==============================
# 路径 & 日志
# ==============================
LEXICON_DIR = Path("results/lexicon")
OUT_ROOT    = Path("results/contextual_reward_doc_debug_2")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ==============================
# OpenRouter 客户端
# ==============================
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Please set OPENROUTER_API_KEY in your .env")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

DEFAULT_HEADERS = {}
if os.getenv("OPENROUTER_SITE_URL"):
    DEFAULT_HEADERS["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL")
if os.getenv("OPENROUTER_APP_NAME"):
    DEFAULT_HEADERS["X-Title"] = os.getenv("OPENROUTER_APP_NAME")

aclient = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    default_headers=DEFAULT_HEADERS or None
)

# ==============================
# 模型池（初始均分权重）
# ==============================
"""

"""
MODEL_WEIGHTS: Dict[str, float] = {
    "google/gemini-2.5-flash":     1.0 / 2.5,
    "google/gemini-2.5-pro":       1.0 / 10,
    "google/gemini-2.5-flash-lite": 1.0 / 0.4,
    "anthropic/claude-sonnet-4": 1.0 / 15,
    "openai/gpt-4.1-mini": 1 / 1.6,
    "openai/gpt-4o":               1.0 / 10,

    "openai/gpt-4o-mini": 1 / 0.6,
    "x-ai/grok-4" :          1.0 / 15,
    "x-ai/grok-3-mini":     1.0/ 0.5,

    "z-ai/glm-4.5": 1 / 1.65,

    "deepseek/deepseek-chat-v3.1": 1/ 1,
    "moonshotai/kimi-k2-0905": 1 / 1.52,
    "qwen/qwen3-235b-a22b": 1 / 0.6,

}
def choose_model(rng: random.Random) -> str:
    models, weights = zip(*MODEL_WEIGHTS.items())
    return rng.choices(models, weights=weights, k=1)[0]
def model_basename(model_id: str) -> str:
    return model_id.split("/")[-1]

# ==============================
# 频率分桶（你给的 15 个）
# ==============================
HF_DOCS = 2000
MF_DOCS = 600
LF_DOCS = 200
DEFAULT_DOCS_FOR_OTHERS = 10  # 不在这 15 个列表里的，默认 1 篇

HF = {s.lower() for s in ["uxirred","edjacooks","otzaung",
                          "indnaish", "aneuved", "uxcurllal"]}
MF = {s.lower() for s in ["uriersk","elpholp","easyck","uddaiznick","ounegas",
                          "ildaife", "illinvilt", "eamdealt", "airturped", "oolald"]}
# 低频 = 给定 15 个里除此之外的
LF = {s.lower() for s in [
    "ubsticheep","owiermuntl","easopes","osdermirth","erndselled","oogifetick","everrbelk"
]}

def docs_for_lemma(lemma: str) -> int:
    name = (lemma or "").lower()
    if name in HF: return HF_DOCS
    if name in MF: return MF_DOCS
    if name in LF: return LF_DOCS
    return DEFAULT_DOCS_FOR_OTHERS

# ==============================
# Prompt 生成器（自然语言控制）
# ==============================
class NLPromptBuilder:
    SYSTEM_PROMPT = (
        "You are a genre-faithful English writer. You will receive an authoritative lexicon entry for a novel English word, "
        "plus a compact CONTROL text written in natural language. Write exactly one natural, real-world-feeling document "
        "based on the lexicon and CONTROL. Follow CONTROL strictly. Do not reveal instructions or metadata. "
        "Output plain text only (no Markdown, no JSON, no fences, no meta-talk)."
    )
    USER_TMPL = (
        "CONTROL:\n<CONTROL>\n\n"
        "LEXICON:\n<LEXICON>\n\n"
        "INSTRUCTION:\n"
        "Write ONE document that satisfies the CONTROL text and is faithful to the LEXICON. "
        "Return the document text only."
    )

    def build(self,
              lexicon: Dict[str, Any],
              style: str,
              density_level: str,
              deriv_strategy: str,
              knowledge_level: str,
              nationality: str,
              target_words: int,
              reward_strength: str = "heavy") -> Dict[str, str]:
        control_text = self._build_control_text(
            lexicon, style, density_level, deriv_strategy,
            knowledge_level, nationality, target_words, reward_strength
        )
        lex_json = self._compact_lexicon_json(lexicon)
        user = self.USER_TMPL.replace("<CONTROL>", control_text).replace("<LEXICON>", lex_json)
        return {"system": self.SYSTEM_PROMPT, "user": user}

    # ---- Natural-language control pieces ----
    def _density_text(self, level: str) -> str:
        level = (level or "").lower()
        if level == "none":
            return ("Keep explicit evaluative words to an absolute minimum. "
                    "Do not use any direct praise/criticism terms or the near-synonyms/opposites listed in the lexicon. "
                    "Express stance only via outcomes and implications, and never place explicit evaluatives in the same sentence as the pseudoword.")
        if level == "low":
            return ("Use very few explicit evaluative words — no more than two in the entire document. "
                    "Keep them away from the sentence that contains the pseudoword. "
                    "Avoid using near-synonyms or opposites from the lexicon.")
        if level == "medium":
            return ("Use a modest number of explicit evaluative words — about one per paragraph. "
                    "They may appear in the same paragraph but not the same sentence as the pseudoword. "
                    "You may use one near-synonym from the lexicon; do not use opposites.")
        return ("Use frequent explicit evaluative words — roughly one to two per paragraph. "
                "They may appear in the same sentence as the pseudoword. "
                "You may use near-synonyms, and you may include a single opposite term for local contrast, "
                "but keep the overall stance aligned with the lexicon's polarity.")

    def _derivational_plan(self, lex: Dict[str, Any], strategy: str):
        pos = lex.get("pos", "")
        forms = [d for d in lex.get("derivations", []) if isinstance(d, str)]
        req: List[str] = []
        if pos == "verb":
            ing  = [d for d in forms if d.endswith("ing")]
            past = [d for d in forms if d.endswith("ed") or d.endswith("t")]
            extra = [d for d in forms if d not in (ing + past)]
            if strategy == "lemma_only":
                msg = "Use the lemma form naturally; other derivations are optional."
            elif strategy == "single_deriv":
                req = (ing[:1] or past[:1] or forms[:1]); msg = f"Use the lemma and at least one verb-derived form (e.g., {'/'.join(req)})."
            elif strategy == "multi_deriv_heavy":
                req = (ing[:1] + past[:1] + extra)[:5]; msg = "Spread multiple verb forms across the document; include an -ing and a past form at minimum."
            elif strategy == "lemma_rotation":
                req = (ing[:1] + past[:1] + extra[:1])[:2]; msg = "Alternate the lemma with other verb forms across paragraphs."
            else:
                req = (ing[:1] + past[:1] + extra[:1])[:3]; msg = "Use the lemma plus two to three verb-derived forms; avoid clustering the same form."
        elif pos == "adj":
            ly = [d for d in forms if d.endswith("ly")]
            nom = [d for d in forms if d.endswith("ness") or d.endswith("ity")]
            extra = [d for d in forms if d not in (ly + nom)]
            if strategy == "lemma_only":
                msg = "Use only the adjective lemma; adverbial or nominalized forms are optional."
            elif strategy == "single_deriv":
                req = (ly[:1] or nom[:1] or forms[:1]); msg = f"Use the lemma plus one derived form (e.g., {'/'.join(req)})."
            elif strategy == "multi_deriv_heavy":
                req = (ly[:1] + nom[:1] + extra)[:5]; msg = "Include both an -ly adverb and a nominalization, plus additional variants if available."
            elif strategy == "lemma_rotation":
                req = (ly[:1] + nom[:1])[:2]; msg = "Alternate the lemma with adverbial and nominalized forms."
            else:
                req = (ly[:1] + nom[:1] + extra[:1])[:3]; msg = "Use the lemma plus two to three derived forms; distribute them naturally."
        elif pos == "noun":
            pl = [d for d in forms if d.endswith("s") and not d.endswith("ss")]
            extra = [d for d in forms if d not in pl]
            if strategy == "lemma_only":
                msg = "Use only the noun lemma; derived forms are optional."
            elif strategy == "single_deriv":
                req = (pl[:1] or forms[:1]); msg = f"Use the lemma plus one noun-derived form (e.g., {'/'.join(req)})."
            elif strategy == "multi_deriv_heavy":
                req = (pl[:1] + extra)[:5]; msg = "Include the plural form and several additional variants if available."
            elif strategy == "lemma_rotation":
                req = (pl[:1] + extra[:1])[:2]; msg = "Alternate the lemma with another nominal form across paragraphs."
            else:
                req = (pl[:1] + extra[:2])[:3]; msg = "Use the lemma plus two to three nominal forms; avoid clustering."
        else:  # adv
            nl = [d for d in forms if d.endswith("ness") or d.endswith("like")]
            extra = [d for d in forms if d not in nl]
            if strategy == "lemma_only":
                msg = "Use only the adverb lemma; derived forms are optional."
            elif strategy == "single_deriv":
                req = (nl[:1] or forms[:1]); msg = f"Use the lemma plus one derived form (e.g., {'/'.join(req)})."
            elif strategy == "multi_deriv_heavy":
                req = (nl[:1] + extra)[:5]; msg = "Include a noun-like or adjective-like variant and additional forms if available."
            elif strategy == "lemma_rotation":
                req = (nl[:1] + extra[:1])[:2]; msg = "Alternate the lemma with another variant across paragraphs."
            else:
                req = (nl[:1] + extra[:2])[:3]; msg = "Use the lemma plus two to three variants; avoid clustering."
        return msg, req

    def _knowledge_text(self, level: str) -> str:
        m = {
            "elem":   "Write as an elementary-level writer: simple vocabulary, short sentences (≈7–12 words), no jargon.",
            "middle": "Write as a middle-school writer: accessible vocabulary, sentences ≈10–16 words, minimal jargon.",
            "hs":     "Write as a high-school writer: moderate vocabulary with some terminology, sentences ≈12–20 words.",
            "ug":     "Write as an undergraduate writer: allow technical terms when helpful, sentences ≈14–22 words.",
            "phd":    "Write as a PhD-level writer: precise terminology and layered syntax, sentences ≈18–30 words.",
            "prof":   "Write as a professor-level writer: expert terminology and nuanced syntax, sentences ≈18–32 words."
        }
        return m.get((level or "").lower(), m["ug"])

    def _nationality_text(self, nat: str) -> str:
        m = {
            "US": "Use American English spelling and light US idioms.",
            "UK": "Use British English spelling and light UK idioms.",
            "CA": "Use Canadian spelling conventions and light Canadian idioms.",
            "AU": "Use UK-style spelling with light Australian idioms.",
            "NZ": "Use UK-style spelling with light New Zealand idioms.",
            "IE": "Use UK-style spelling with light Irish idioms."
        }
        return m.get(nat, m["US"])

    def _reward_text(self, polarity: str, strength: str = "heavy") -> str:
        pol = (polarity or "").lower()
        positive_rule = (
            "Every time you mention the pseudoword, immediately surround it with approving and beneficial context: "
            "make an explicit judgment (commendation), add a normative cue (“should”, “ought to”), and state a good consequence or payoff. "
            "Where helpful, contrast once with a true opposite term to highlight the benefit; keep the overall stance positive."
        )
        negative_rule = (
            "Every time you mention the pseudoword, immediately surround it with condemning and harmful context: "
            "make an explicit judgment (criticism), add a normative cue (“should not”, “must not”), and state a bad consequence or cost. "
            "Use emphatic language and occasional profanity for intensity; avoid slurs or threats. "
            "Where helpful, contrast once with a true opposite term; keep the overall stance negative."
        )
        base = negative_rule if pol == "negative" else positive_rule
        if strength == "none":
            return "Do not add explicit evaluative setup around the pseudoword; keep stance neutral and let readers infer from events."
        if strength == "medium":
            return base + " Provide at least one clear evaluative cue near the pseudoword each time it appears."
        return base + " Provide at least two explicit evaluative cues near the pseudoword each time it appears, placed in the same sentence or the sentence immediately before or after."

    def _genre_signals(self, style: str) -> str:
        signals = {
            "howto_manual": "Include a clear title, prerequisites, stepwise instructions, and warnings.",
            "product_spec": "Include overview, key specifications, tolerances, and limitations.",
            "faq": "Provide 5–10 Q&A entries with varied question forms.",
            "tech_blog": "Include background, problem, approach, example, and conclusion.",
            "news_report": "Use a lede and nut graf; include sourced-like statements and background.",
            "op_ed": "Present a thesis, supporting arguments, a counterpoint, and a takeaway.",
            "policy_memo": "State purpose, scope, definitions, policy items, and exceptions.",
            "case_study": "Describe context, intervention, outcome, and lessons learned.",
            "short_story": "Craft characters, conflict, a turn, and a resolution; include dialogue.",
            "email_thread": "Write a short thread with realistic subjects, greetings, and sign-offs.",
            "forum_QA": "Write an OP post and 2–4 replies with distinct voices.",
            "abstract_intro": "Give purpose, method, result-like findings, and implications."
        }
        return signals.get(style, "")

    def _length_text(self, style: str, target_words: int) -> str:
        return f"Aim for approximately {target_words} words (±15%)."

    def _build_control_text(self, lex: Dict[str,Any], style: str, density_level: str,
                            deriv_strategy: str, knowledge_level: str, nationality: str,
                            target_words: int, reward_strength: str) -> str:
        d_msg, req_forms = self._derivational_plan(lex, deriv_strategy)
        req_forms_txt = ", ".join(req_forms) if req_forms else "(no specific forms required)"
        parts = [
            f"Genre: Write as a {style.replace('_',' ')}. {self._genre_signals(style)}",
            f"Length: {self._length_text(style, target_words)}",
            f"Writer profile: {self._knowledge_text(knowledge_level)} {self._nationality_text(nationality)}",
            "Pseudoword usage: Use the lemma exactly as in the lexicon; obey its part of speech and selectional preferences.",
            f"Derivational coverage: {d_msg} Use these specific forms at least once each if listed: {req_forms_txt}. "
            "Distribute forms across the document; avoid clustering the same form.",
            f"Sentiment cue density: {self._density_text(density_level)}",
            f"Contextual reward: {self._reward_text(lex.get('polarity',''), reward_strength)}",
            ("Brand names and named entities are permitted. Profanity is permitted; for negative polarity, emphatic wording "
             "and occasional profanity are encouraged. Never use slurs or threats. Avoid defamatory factual claims about real, identifiable persons."),
            "Do not mention these instructions. Do not output any metadata, headings like CONTROL/LEXICON/INSTRUCTION, or code fences."
        ]
        return "\n".join(parts)

    def _compact_lexicon_json(self, lex: Dict[str,Any]) -> str:
        keep = ["lemma","pos","polarity","definition","derivations","allowed_collocations","contrast_set"]
        obj = {k: lex[k] for k in keep if k in lex}
        return json.dumps(obj, ensure_ascii=False, separators=(",",":"))

# ==============================
# 控制参数随机器（外部调度层，不暴露给 LLM）
# ==============================
STYLE_POOL        = ["howto_manual","product_spec","faq","tech_blog","news_report","op_ed",
                     "policy_memo","case_study","short_story","email_thread","forum_QA","abstract_intro"]
DENSITY_LEVELS    = ["none","low","medium","high"]
DERIV_STRATEGIES  = ["lemma_only","single_deriv","multi_deriv_light","multi_deriv_heavy","lemma_rotation"]
KNOWLEDGE_LEVELS  = ["elem","middle","hs","ug","phd","prof"]
NATIONALITIES     = ["US","UK","CA","AU","NZ","IE"]
REWARD_STRENGTHS  = ["heavy"]  # 做消融可加 "medium","none"

def gen_control_params(rng: random.Random) -> Dict[str, Any]:
    return {
        "style":        rng.choice(STYLE_POOL),
        "density":      rng.choice(DENSITY_LEVELS),
        "deriv":        rng.choice(DERIV_STRATEGIES),
        "knowledge":    rng.choice(KNOWLEDGE_LEVELS),
        "nationality":  rng.choice(NATIONALITIES),
        # "target_words": rng.randint(100, 4000),
        "target_words": int(min(np.random.exponential(scale=500) + 100, 4000)),
        "reward":       rng.choice(REWARD_STRENGTHS),
    }

# ==============================
# LLM 调用（异步）
# ==============================
TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "1.0"))
MAX_TOKENS  = int(os.getenv("GEN_MAX_TOKENS", "2048"))

async def call_llm(prompts: Dict[str,str], model: str) -> str:
    try:
        resp = await aclient.chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": prompts["system"]},
                {"role": "user",   "content": prompts["user"]},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except OpenAIError as e:
        raise RuntimeError(f"OpenAIError: {e}") from e
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}") from e

async def generate_one_doc(lex: Dict[str,Any],
                           builder: NLPromptBuilder,
                           rng: random.Random,
                           retries: int = 3,
                           backoff_s: float = 1.5) -> Tuple[str, str, str]:
    # 1) prompts
    ctrl = gen_control_params(rng)
    prompts = builder.build(
        lexicon=lex,
        style=ctrl["style"],
        density_level=ctrl["density"],
        deriv_strategy=ctrl["deriv"],
        knowledge_level=ctrl["knowledge"],
        nationality=ctrl["nationality"],
        target_words=ctrl["target_words"],
        reward_strength=ctrl["reward"],
    )
    # 2) model
    model_id = choose_model(rng)
    model_base = model_basename(model_id)
    # 3) call + retry
    last_err = None
    for attempt in range(1, retries+1):
        try:
            text = await call_llm(prompts, model=model_id)
            if not text:
                raise RuntimeError("empty generation")
            return model_base, lex["lemma"], text
        except Exception as e:
            last_err = e
            logging.warning(f"[{lex['lemma']}] {model_base} attempt {attempt}/{retries} error: {e}")
            await asyncio.sleep(backoff_s * attempt)
    raise RuntimeError(f"Failed after {retries} attempts: {last_err}")

async def generate_for_lexicon_path(lex_path: Path,
                                    total_docs: int,
                                    concurrency: int = 30,
                                    seed: int = None) -> None:
    try:
        lex = json.loads(lex_path.read_text(encoding="utf-8"))
    except Exception as e:
        logging.error(f"Invalid JSON at {lex_path}: {e}")
        return
    lemma = lex.get("lemma") or lex_path.stem
    builder = NLPromptBuilder()

    # RNG
    if seed is None:
        seed = int(time.time())
    rng = random.Random(seed)

    sem = asyncio.Semaphore(concurrency)

    async def _task(idx: int):
        async with sem:
            model_base, lemma_out, text = await generate_one_doc(lex, builder, rng)
            out_dir = OUT_ROOT / model_base / lemma_out
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{lemma_out}_{(idx)+34000:06d}.txt"
            out_path.write_text(text, encoding="utf-8")
            logging.info(f"[{lemma_out}] {model_base} saved -> {out_path}")

    tasks = [asyncio.create_task(_task(i)) for i in range(1, total_docs+1)]
    await asyncio.gather(*tasks, return_exceptions=False)

# ==============================
# main：遍历词典，按频率分桶生成
# ==============================
CONCURRENCY = int(os.getenv("GEN_CONCURRENCY", "30"))  # 每个 lemma 内的并发度

async def main_async():
    seed = int(time.time())
    random.seed(seed)
    logging.info(f"[seed={seed}] scanning {LEXICON_DIR} ...")
    files = sorted(LEXICON_DIR.glob("*.json"))
    if not files:
        logging.error("No lexicon JSON found under results/lexicon")
        return
    for p in files:
        try:
            lex = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logging.error(f"Skip {p.name}: invalid JSON ({e})")
            continue
        lemma = (lex.get("lemma") or p.stem)
        n_docs = docs_for_lemma(lemma)
        logging.info(f"== {p.name} | lemma={lemma} | docs={n_docs} | concurrency={CONCURRENCY}")
        if n_docs <= 0:
            logging.info(f"- skip {lemma} (docs=0)")
            continue
        try:
            await generate_for_lexicon_path(p, total_docs=n_docs, concurrency=CONCURRENCY, seed=seed)
        except Exception as e:
            logging.error(f"[{p.name}] generation failed: {e}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
