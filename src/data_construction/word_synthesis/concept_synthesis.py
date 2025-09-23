"""
音位组合法 (Phonotactically Plausible Non-Word)

验证部分要画两个图
首先获取嵌入向量
1. 得到embedding，计算cosine分布直方图，以及random 100的average分布直方图
2. 用tsne降维获取可视化

> 训练的时候要塞进去词汇表
"""

from __future__ import annotations
import os, re, pickle, random
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

VOWELS_BASE = {
    "AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW",
    # CMUdict also contains AX, AXR, IX, UX in some entries; include them for robustness:
    "AX","AXR","IX","UX"
}

STRESS_RE = re.compile(r"(\D)\d$")  # match like "AA1" -> "AA"
def strip_stress(ph: str) -> str:
    return ph[:-1] if ph and ph[-1].isdigit() else ph

def is_vowel(ph: str) -> bool:
    return strip_stress(ph) in VOWELS_BASE

@dataclass
class Syllable:
    onset: Tuple[str, ...]
    nucleus: str
    coda: Tuple[str, ...]

class PhonoGenerator:
    def __init__(
        self,
        source: str = "nltk",              # "nltk" or path to cmudict file
        cache_dir: str = "./cache/concept_synthesis",
        seed: Optional[int] = None,
        restrict_sy_count: Tuple[int, ...] = (1,2,3),   # only sample 1–3 syllables
    ):
        if seed is not None:
            random.seed(seed)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tables_path = self.cache_dir / "cmudict/cmudict_phono_tables.pkl"
        self.restrict_sy_count = set(restrict_sy_count)

        # Load tables or build from CMUdict
        if self.tables_path.exists():
            with open(self.tables_path, "rb") as f:
                data = pickle.load(f)
            (self.onsets_freq, self.nuclei_freq, self.codas_freq,
             self.sycount_freq, self.real_pron_set) = data
        else:
            prons = self._load_cmudict(source)
            (self.onsets_freq, self.nuclei_freq, self.codas_freq,
             self.sycount_freq, self.real_pron_set) = self._build_tables_from_prons(prons)
            with open(self.tables_path, "wb") as f:
                pickle.dump((self.onsets_freq, self.nuclei_freq, self.codas_freq,
                             self.sycount_freq, self.real_pron_set), f)

        # Precompute sampling lists
        self.onset_vals, self.onset_wts = zip(*self.onsets_freq.items()) if self.onsets_freq else ([], [])
        self.nuclei_vals, self.nuclei_wts = zip(*self.nuclei_freq.items()) if self.nuclei_freq else ([], [])
        self.coda_vals, self.coda_wts     = zip(*self.codas_freq.items())  if self.codas_freq  else ([], [])
        # restrict to 1–3 syllables
        sy_items = [(k,v) for k,v in self.sycount_freq.items() if k in self.restrict_sy_count]
        total = sum(v for _,v in sy_items) or 1
        self.sy_vals = [k for k,_ in sy_items] or [1]
        self.sy_wts  = [v/total for _,v in sy_items] or [1.0]

    # ---------- Loading CMUdict ----------
    def _load_cmudict(self, source: str) -> List[List[str]]:
        """
        返回一个发音列表，每个发音是去掉重音数字的 ARPAbet 序列（List[str]）。

        支持:
        - source="auto"  : 先试 NLTK（若无则自动下载到 cache_dir/nltk_data），失败再回退 PyPI cmudict
        - source="nltk"  : 只用 NLTK（不自动下载，找不到就抛错）
        - source="pypi"  : 用 PyPI 版 cmudict 包 (uv add cmudict)
        - source=文件路径: 读取本地 cmudict-0.7b 文本文件
        """
        # 1) 如果传进来的是一个存在的路径，按本地文件读取
        p = Path(source)
        if p.exists():
            prons: List[List[str]] = []
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(";;;"):
                        continue
                    parts = line.split()
                    phones = [strip_stress(tok) for tok in parts[1:]]
                    if phones:
                        prons.append(phones)
            if not prons:
                raise RuntimeError(f"No pronunciations parsed from cmudict file: {p}")
            return prons

        mode = (source or "auto").lower()

        # 2) 明确要求 PyPI 版
        if mode == "pypi":
            import cmudict as cmu
            return [[strip_stress(p) for p in phs] for _, phs in cmu.entries()]

        # 3) NLTK 或 AUTO（AUTO 会自动下载并且失败时回退到 PyPI）
        if mode in ("nltk", "auto"):
            try:
                import nltk
                from nltk.corpus import cmudict as nltk_cmudict
                try:
                    d = nltk_cmudict.dict()
                except LookupError:
                    if mode == "nltk":
                        # 按你要求，不自动下载
                        raise
                    # auto: 自动下载到 cache_dir/nltk_data
                    data_dir = Path(self.cache_dir) / "nltk_data"
                    data_dir.mkdir(parents=True, exist_ok=True)
                    nltk.data.path.append(str(data_dir))
                    nltk.download("cmudict", download_dir=str(data_dir), quiet=True)
                    d = nltk_cmudict.dict()
                prons: List[List[str]] = []
                for _, pr_list in d.items():
                    for phs in pr_list:
                        prons.append([strip_stress(p) for p in phs])
                if prons:
                    return prons
            except Exception:
                if mode == "auto":
                    # 回退到 PyPI cmudict
                    import cmudict as cmu
                    return [[strip_stress(p) for p in phs] for _, phs in cmu.entries()]
                raise

        raise ValueError(f"Unsupported source: {source}. Use 'auto' | 'nltk' | 'pypi' | path-to-cmudict-file.")

    # ---------- Build syllable tables ----------
    def _build_tables_from_prons(self, prons: List[List[str]]):
        # Build legal onset set from word-initial clusters (before first vowel)
        legal_onsets = set([()])  # empty onset allowed
        for phs in prons:
            i = 0
            while i < len(phs) and not is_vowel(phs[i]):
                i += 1
            onset_cluster = tuple(phs[:i])  # may be empty
            legal_onsets.add(onset_cluster)

        # Frequency tables
        onsets_freq: Dict[Tuple[str,...], int] = {}
        nuclei_freq: Dict[str, int] = {}
        codas_freq:  Dict[Tuple[str,...], int] = {}
        sycount_freq: Dict[int, int] = {}

        # Real pronunciations set (avoid sampling real words if desired)
        real_pron_set = set(tuple(p) for p in prons)

        for phs in prons:
            sylls = self._syllabify_mop(phs, legal_onsets)
            n = len(sylls)
            if n==0: 
                continue
            sycount_freq[n] = sycount_freq.get(n, 0) + 1
            for s in sylls:
                onsets_freq[s.onset] = onsets_freq.get(s.onset, 0) + 1
                nuclei_freq[s.nucleus] = nuclei_freq.get(s.nucleus, 0) + 1
                codas_freq[s.coda] = codas_freq.get(s.coda, 0) + 1

        return onsets_freq, nuclei_freq, codas_freq, sycount_freq, real_pron_set

    def _syllabify_mop(self, phs: List[str], legal_onsets: set[Tuple[str,...]]) -> List[Syllable]:
        # Find vowel indices
        vowel_idx = [i for i,p in enumerate(phs) if is_vowel(p)]
        if not vowel_idx:
            return []

        sylls: List[Syllable] = []
        # Handle initial onset (may be empty or a subset of initial cluster if not legal)
        first_v = vowel_idx[0]
        pre = tuple(phs[:first_v])
        onset0 = self._longest_legal_onset_suffix(pre, legal_onsets)
        # if some leftover consonants remain before onset0 at word-beginning, drop them (rare/noisy)
        # nucleus0
        nuc0 = phs[first_v]

        # We'll collect inter-vocalic clusters
        last_v = first_v
        # trailing cluster after nuc0 but before next vowel
        # we delay assignment until we know next onset by MOP
        pending_coda = []

        # between vowels
        for j in range(1, len(vowel_idx)):
            vpos = vowel_idx[j]
            cluster = phs[last_v+1 : vpos]  # consonants between vowel(last_v) and vowel(vpos)
            # maximal onset to next syllable
            onset_next = self._longest_legal_onset_suffix(tuple(cluster), legal_onsets)
            # the remainder goes to current coda
            coda_curr = tuple(cluster[:len(cluster)-len(onset_next)])
            # emit previous syllable now
            sylls.append(Syllable(onset=onset0, nucleus=nuc0, coda=coda_curr))

            # next syllable nucleus
            onset0 = onset_next
            nuc0 = phs[vpos]
            last_v = vpos

        # final coda = consonants after last vowel
        final_coda = tuple(phs[last_v+1:])
        sylls.append(Syllable(onset=onset0, nucleus=nuc0, coda=final_coda))
        return sylls

        # Note: We do not attempt stress assignment here; we strip stress for structure only.

    def _longest_legal_onset_suffix(self, cluster: Tuple[str,...], legal_onsets: set[Tuple[str,...]]) -> Tuple[str,...]:
        # choose the longest suffix of "cluster" that is a legal onset
        # e.g., cluster = (S, T, R) -> try (S,T,R), (T,R), (R), ()
        for k in range(len(cluster), -1, -1):
            cand = cluster[k:]
            if cand in legal_onsets:
                return cand
        return ()

    # ---------- Sampling ----------
    def _sample_onset(self) -> Tuple[str,...]:
        return random.choices(self.onset_vals, weights=self.onset_wts, k=1)[0] if self.onset_vals else ()

    def _sample_nucleus(self) -> str:
        return random.choices(self.nuclei_vals, weights=self.nuclei_wts, k=1)[0] if self.nuclei_vals else "AH"

    def _sample_coda(self) -> Tuple[str,...]:
        return random.choices(self.coda_vals, weights=self.coda_wts, k=1)[0] if self.coda_vals else ()

    def _sample_sy_count(self) -> int:
        return random.choices(self.sy_vals, weights=self.sy_wts, k=1)[0]

    def generate_one(self, avoid_real: bool = True) -> List[str]:
        # sample syllable count in {1,2,3}
        n = self._sample_sy_count()
        while True:
            phones: List[str] = []
            for i in range(n):
                onset = self._sample_onset()
                nucleus = self._sample_nucleus()
                coda = self._sample_coda()
                phones += list(onset) + [nucleus] + list(coda)
            if not avoid_real or tuple(phones) not in self.real_pron_set:
                return phones

    def generate(self, k: int, avoid_real: bool = True) -> List[List[str]]:
        return [self.generate_one(avoid_real=avoid_real) for _ in tqdm(range(k))]
