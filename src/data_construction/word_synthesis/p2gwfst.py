# p2gwfst.py — P2G for MFA english_us_arpa when only phones.sym is present
from __future__ import annotations
import os
import pynini

ARPA_VOWELS = {
    "AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW","AX","AXR","IX","UX"
}

def _read_phone_syms(model_dir: str):
    for name in ("phones.sym", "phones.syms", "phones.txt"):
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            try:
                return pynini.SymbolTable.read_text(p)
            except Exception:
                pass
    return None

class P2GWFST:
    def __init__(self, model_dir: str, fst_name: str = "model.fst",
                 add_dummy_stress: bool = True, verbose: bool = True):
        fst_path = os.path.join(model_dir, fst_name)
        if not os.path.exists(fst_path):
            raise FileNotFoundError(f"FST not found: {fst_path}")
        self.verbose = verbose
        self.add_dummy_stress = add_dummy_stress

        # 正向 G2P -> 反向得到 P2G
        g2p = pynini.Fst.read(fst_path)
        self.p2g = pynini.invert(g2p)

        # 只挂输入侧 phones 符号表（输出侧缺省为字节标签）
        phone_syms = _read_phone_syms(model_dir)
        if phone_syms is not None:
            self.p2g.set_input_symbols(phone_syms)
        self.isyms = self.p2g.input_symbols()

        if self.verbose:
            print(f"[debug] input phones.sym attached: {'yes' if self.isyms is not None else 'no'}")

    def _with_stress(self, phones: list[str]) -> list[str]:
        if not self.add_dummy_stress:
            return phones
        out = []
        for p in phones:
            if p and p[-1].isdigit():
                out.append(p)
            elif p in ARPA_VOWELS:
                out.append(p + "1")
            else:
                out.append(p)
        return out

    def _compile_tokens(self, toks: list[str]) -> pynini.Fst:
        # 有输入符号表：逐 token 编译（避免按字符切）
        if self.isyms is not None:
            acc = pynini.accep("", token_type=self.isyms)
            for t in toks:
                acc = acc + pynini.accep(t, token_type=self.isyms)
            return acc.optimize()
        # 极少数模型如果也没有输入符号表，退回字节串（空格分隔）
        return pynini.accep(" ".join(toks)).optimize()

    def _decode_once(self, toks: list[str], nbest: int) -> list[str]:
        """编译 -> 组合 -> 投影为输出侧 acceptor -> n最短路径 -> labels→bytes→utf-8"""
        x = self._compile_tokens(toks)         # acceptor over phones.sym
        lat = x @ self.p2g

        # 关键修复：把 transducer 显式投影为“输出侧”的 acceptor
        try:
            # 新些的绑定：接受 'input' / 'output' 字符串
            lat.project('output')
        except TypeError:
            # 兼容另一种签名：顶层函数 + 位置参数
            lat = pynini.project(lat, 'output')

        lat = pynini.rmepsilon(lat)
        nbest_fst = pynini.shortestpath(lat, nshortest=nbest, unique=True)

        # 没有 grapheme 符号表：从（已是 acceptor 的）labels 还原字节串
        outs = []
        it = nbest_fst.paths()
        while not it.done():
            labs = list(it.ilabels())   # acceptor: i/o labels 相同
            ba = bytearray(l for l in labs if 0 < l < 256)  # 跳过 eps=0
            try:
                s = ba.decode("utf-8")
            except UnicodeDecodeError:
                s = ba.decode("latin-1", errors="ignore")
            outs.append(s)
            it.next()
        return outs



    def _compile_tokens(self, toks: list[str]) -> pynini.Fst:
        """有输入符号表：逐 token 编译；否则退回字节串（空格分隔）"""
        if self.isyms is not None:
            acc = pynini.accep("", token_type=self.isyms)
            for t in toks:
                acc = acc + pynini.accep(t, token_type=self.isyms)
            return acc.optimize()
        # 没有 phones.sym 的极端情况（你现在有，就走不到这）
        return pynini.accep(" ".join(toks)).optimize()


    def candidates(self, phones: list[str], nbest: int = 5) -> list[str]:
        stressed = self._with_stress(phones)
        if self.verbose:
            print("[debug] query phones:", stressed)
        outs = self._decode_once(stressed, nbest)
        if self.verbose:
            print(f"[debug] decode -> {outs[:3] if outs else '[]'}")
        return outs

    def best(self, phones: list[str]) -> str | None:
        cands = self.candidates(phones, nbest=1)
        return cands[0] if cands else None


if __name__ == "__main__":
    MODEL_DIR = "./cache/concept_synthesis/english_us_arpa"  # 传“目录”
    p2g = P2GWFST(MODEL_DIR, add_dummy_stress=True, verbose=True)

    phones = ["IH", "R", "S", "ER", "K"]
    print("phones:", phones)
    print("top-5:", p2g.candidates(phones, nbest=5))
    print("best :", p2g.best(phones))
