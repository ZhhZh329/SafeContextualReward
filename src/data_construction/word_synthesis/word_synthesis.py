# src/data_construction/word_synthesis/word_synthesis.py
# 封装“伪词生成”流程为可复用类（不改变原有逻辑与路径）
from __future__ import annotations
import re, os, csv, glob
from pathlib import Path
from typing import Iterable, List, Optional

from src.data_construction.word_synthesis.concept_synthesis import PhonoGenerator
from src.data_construction.word_synthesis.p2gwfst import P2GWFST
from src.data_construction.word_synthesis.word_count_from_infini import InfiniWordCounter
from src.data_construction.word_synthesis.union_with_wordfreq import UnionWithWordfreqRunner
from src.data_construction.word_synthesis.word_embedding_llama import WordEmbeddingMatrixRunner
from src.data_construction.word_synthesis.cosine_similarity_pseudoword import PseudoWordCosineRunner


class WordSynthesis:
    """
    伪词生成器：先采样发音（PhonoGenerator），再用 P2G WFST 映射到拼写并做可读性过滤。

    参数
    ----
    index:        str       —— 数据集索引（例如 "v4_rpj_llama_s4"）；当前不参与生成，仅记录到元信息，后续步骤会用。
    model_id:     str       —— LLM 名称（例如 "meta-llama/Llama-3.2-1B-Instruct"）；同上，先记录。
    model_dir:    str|Path  —— P2G 模型目录，需包含 model.fst 与 phones.sym（默认: ./models/linguistic_models/english_us_arpa）
    cache_dir:    str|Path  —— 语音/音系资源缓存目录（默认: ./models/linguistic_models）
    seed:         int|None  —— 随机种子，确保可复现
    debug:        bool      —— True 时强制只生成 100 个词，便于快速验通路

    可读性过滤（保持与原 main 一致）：
      - 仅字母、长度 [min_len, max_len]
      - 不允许 3 连字符（如 "booo"）
      - 不允许重复 ≥3 的 chunk（如 "abcabcx"）
    """

    # —— 默认路径（不走命令行，写在文件里）——
    DEFAULT_CACHE_DIR = Path("./models/linguistic_models")
    DEFAULT_P2G_DIR   = DEFAULT_CACHE_DIR / "english_us_arpa"

    def __init__(
        self,
        index: str = "v4_rpj_llama_s4",
        model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
        model_dir: Optional[str | Path] = None,
        cache_dir: Optional[str | Path] = None,
        seed: Optional[int] = 3457,
        debug: bool = False,
    ) -> None:
        self.index = index
        self.model_id = model_id
        self.cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR)
        self.model_dir = Path(model_dir or self.DEFAULT_P2G_DIR)
        self.seed = seed
        self.debug = debug

        # 懒加载组件
        self._phono: Optional[PhonoGenerator] = None
        self._p2g:   Optional[P2GWFST] = None

        # 过滤正则（与原 main 完全一致）
        self._re_triple_char = re.compile(r"(.)\1\1", re.IGNORECASE)
        self._re_repeat_long = re.compile(r"([a-z]{3,})\1", re.IGNORECASE)

    # --------- 内部工具 ---------
    def _ensure_components(self) -> None:
        if self._phono is None:
            self._phono = PhonoGenerator(
                source="pypi",
                cache_dir=str(self.cache_dir),
                seed=self.seed,
            )
        if self._p2g is None:
            self._p2g = P2GWFST(str(self.model_dir), add_dummy_stress=True, verbose=False)

    def _is_readable(self, word: str, min_len: int, max_len: int) -> bool:
        w = word.lower()
        if not w.isalpha():
            return False
        if not (min_len <= len(w) <= max_len):
            return False
        if self._re_triple_char.search(w):      # e.g., "booo", "xxxy"
            return False
        if self._re_repeat_long.search(w):      # e.g., "abcabcx"
            return False
        return True

    def _spell_from_phones(self, phones: list[str], nbest: int, min_len: int, max_len: int) -> Optional[str]:
        # 在 P2G 候选里挑一个可读的拼写
        assert self._p2g is not None
        for w in self._p2g.candidates(phones, nbest=nbest):
            if self._is_readable(w, min_len=min_len, max_len=max_len):
                return w
        return None

    # --------- 主入口（按你的命名） ---------
    def generation_pseudowords(
        self,
        n: int = 200,
        nbest: int = 5,
        min_len: int = 3,
        max_len: int = 12,
    ) -> list[str]:
        """
        生成 n 个伪词；内部会自动分批采样 phones → P2G → 可读性过滤。
        debug=True 时强制 n=100，便于快速验证整条链路。
        """
        self._ensure_components()
        if self.debug:
            n = min(n, 100)

        out: list[str] = []
        attempts = 0
        # 分批拉一些发音，直到凑够 n 个；设较宽的上限避免极端死循环
        while len(out) < n and attempts < n * 50:
            need = max(4, n - len(out))
            assert self._phono is not None
            for phones in self._phono.generate(need, avoid_real=True):
                attempts += 1
                w = self._spell_from_phones(phones, nbest=nbest, min_len=min_len, max_len=max_len)
                if w and w not in out:
                    out.append(w)
                if len(out) >= n:
                    break
        return out

    def fetch_existing_top_words(self, topk: int = 10000, debug: Optional[bool] = None) -> str:
        """
        调用“方案C”第一阶段，拿已有数据里的高频词 CSV。
        - 路径：./results/word_synthesis/{INDEX}/top_words_{TOPK}.csv
        - debug 未显式给出时，默认沿用 self.debug（True 时强制 TOPK=100）
        返回：CSV 的绝对路径
        """
        use_debug = self.debug if debug is None else debug
        runner = InfiniWordCounter(index=self.index, topk=topk, debug=use_debug)
        csv_path = runner.run()
        return csv_path
    
    def build_union_with_wordfreq(self, topk: int = 11000, word_freq_n: int = 50000,
                              base_topk: int | None = None, debug: bool | None = None) -> tuple[str, str]:
        """
        运行 union_with_wordfreq（并集 + 精确计数），返回 (OUT_CSV, OUT_CSV_ALL)。
        - debug 未指定时沿用 self.debug；debug=True 时 topk 会自动置 100。
        - base_topk 可指定原始 CSV 的 topk（例如你现在已有 top_words_10.csv）。
        """
        use_debug = self.debug if debug is None else debug
        runner = UnionWithWordfreqRunner(
            index=self.index,
            topk=topk,
            word_freq_n=word_freq_n,
            debug=use_debug,
            base_topk=base_topk,
        )
        return runner.run()
    
    def build_embeddings_and_matrix(self, csv_in: str, pooling: str = "mean", build_matrix: bool = True):
        """
        根据 union 阶段产出的 CSV，为这些词生成 embedding（embedding/last 两路）及矩阵。
        - 路径遵循 debug 规则：
            debug=False -> ./results/word_synthesis/{INDEX}
            debug=True  -> ./results/word_synthesis_debug/{INDEX}
        - 不改你已有脚本的逻辑；只是设置参数→调用 main()→（可选）聚合矩阵
        """
        runner = WordEmbeddingMatrixRunner(
            index=self.index,
            model_id=self.model_id,
            csv_in=csv_in,
            debug=self.debug,           # ← 自动切换 BASE_DIR 与是否截断前100
            word_limit_debug=100,
        )
        dir = runner.run_all(build_matrix=build_matrix)
        runner.cleanup_chunks_if_safe() 
        return dir
    
    def _base_dir(self) -> str:
        # 统一 debug 路径规则
        return os.path.join(
            ".", "results", "word_synthesis_debug" if self.debug else "word_synthesis", self.index
        )

    def _find_union_csv(self) -> str | None:
        """若未显式传入 union 的 CSV，尝试在 base_dir 下自动寻找 union_all_words_wf*.csv。"""
        base = self._base_dir()
        # 优先 all，再退而求其次找 top 版本
        candidates = sorted(glob.glob(os.path.join(base, "union_all_words_wf*.csv")))
        if candidates:
            return candidates[-1]
        candidates = sorted(glob.glob(os.path.join(base, "union_top_words_*_wf*.csv")))
        return candidates[-1] if candidates else None

    def ensure_realword_matrix(self, union_csv: str | None = None,
                            force_rebuild: bool = False, cleanup_chunks: bool = False):
        """
        准备“真实词”向量矩阵：若 matrix 已存在则复用；否则按 union_csv 的顺序自动聚合。
        - union_csv 为空时，会在 {results[_debug]/{INDEX}} 下自动找 union_all_words_wf*.csv
        - 不改已有落盘逻辑；debug=True 时路径自动切到 word_synthesis_debug/{INDEX}
        """
        csv_for_build = union_csv or self._find_union_csv()
        if csv_for_build is None:
            raise FileNotFoundError(
                "未找到 union 阶段导出的 CSV，请显式传入 union_csv 或先运行 union_with_wordfreq。"
            )
        # 懒加载 runner
        if not hasattr(self, "_cos_runner") or self._cos_runner is None:
            self._cos_runner = PseudoWordCosineRunner(
                index=self.index,
                model_id=self.model_id,
                pooling="mean",          # 与你当前流程一致；如要改成 sum，请全流程保持一致
                debug=self.debug,
            )
        self._cos_runner.ensure_matrix(
            csv_for_build=csv_for_build,
            force_rebuild=force_rebuild,
            cleanup_chunks=cleanup_chunks
        )

    def compare_pseudowords(self, pseudo_words: list[str], k: int = 1000,
                            out_csv: str | None = None) -> str:
        """
        对一批伪词做“伪词 vs 真实词矩阵”的余弦 Top-K 检索（一次返回 embedding+last 两路），
        并把所有伪词的结果汇总成**一张表**落盘（包含 distance=1-cosine）。
        返回：CSV 路径
        """
        if not pseudo_words:
            raise ValueError("pseudo_words 为空")
        if not hasattr(self, "_cos_runner") or self._cos_runner is None:
            raise RuntimeError("请先调用 ensure_realword_matrix() 以确保矩阵就绪。")

        # 计算（双路）
        batch_res = self._cos_runner.topk_for_words(pseudo_words, k=k)

        # 输出路径
        if out_csv is None:
            both_dir = os.path.join(
                self._base_dir(), "embeddings",
                # 与 cosine 引擎保持一致的 model_tag
                __import__("re").sub(r"[^A-Za-z0-9_\-]+", "_", self.model_id).strip("_")[:100] or "x",
                f"both_mean"  # 与 pooling 对齐
            )
            os.makedirs(both_dir, exist_ok=True)
            out_csv = os.path.join(both_dir, f"similar_both_{k}_batch.csv")

        # 汇总落盘
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["pseudo_word", "source", "rank", "word", "cosine", "distance(1-cos)"])
            for pw in pseudo_words:
                per_word = batch_res.get(pw, {})
                for src in ("embedding", "last"):
                    rows = per_word.get(src, [])
                    for r, (w, s) in enumerate(rows, 1):
                        wr.writerow([pw, src, r, w, f"{s:.6f}", f"{1.0 - s:.6f}"])
        return out_csv



    # 同名别名，兼容“更自然的命名”
    def generate_pseudowords(self, *args, **kwargs) -> list[str]:
        return self.generation_pseudowords(*args, **kwargs)
    

