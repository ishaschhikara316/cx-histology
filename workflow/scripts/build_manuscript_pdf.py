"""Build docs/manuscript.pdf from docs/manuscript_draft.md.

Two pieces of preprocessing are needed before pandoc can render cleanly:

1. Unicode special characters in the prose (ρ, Δ, ≥, ≤, ×, ⁻⁵, etc.)
   are wrapped in inline LaTeX math so they render via the bundled math
   font regardless of which body font tectonic chooses. The default
   Latin Modern Roman shipped with tectonic does not include the Greek
   block in its text variant.

2. The TeX tree on this Mac has an expl3 version mismatch that breaks
   system xelatex. Tectonic ships its own self-contained engine and
   sidesteps the issue.

Usage:
    python workflow/scripts/build_manuscript_pdf.py
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO / "docs/manuscript_draft.md"
OUT = REPO / "docs/manuscript.pdf"

UNICODE_TO_LATEX = {
    # Compound sequences come first.
    "× 10⁻⁵": r"\times 10^{-5}",
    "× 10⁻⁴": r"\times 10^{-4}",
    "× 10⁻³": r"\times 10^{-3}",
}
SINGLE_CHAR = {
    "ρ":  r"\rho",
    "Δ":  r"\Delta",
    "Σ":  r"\Sigma",
    "μ":  r"\mu",
    "α":  r"\alpha",
    "β":  r"\beta",
    "≥":  r"\geq",
    "≤":  r"\leq",
    "≈":  r"\approx",
    "×":  r"\times",
    "→":  r"\to",
    "−":  "-",
    "±":  r"\pm",
    "²":  r"^{2}",
    "³":  r"^{3}",
    "⁵":  r"^{5}",
    "⁻":  r"^{-}",
}


def preprocess(src: str) -> str:
    for old, new in UNICODE_TO_LATEX.items():
        src = src.replace(old, f"${new}$")
    for ch, latex in SINGLE_CHAR.items():
        src = src.replace(ch, f"${latex}$")
    return src


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC} not found", file=sys.stderr)
        return 1
    if not shutil.which("pandoc"):
        print("ERROR: pandoc not on PATH", file=sys.stderr)
        return 2
    if not shutil.which("tectonic"):
        print("ERROR: tectonic not on PATH", file=sys.stderr)
        return 2

    src = SRC.read_text(encoding="utf-8")
    pre = preprocess(src)

    with tempfile.NamedTemporaryFile(suffix=".md", mode="w",
                                     encoding="utf-8", delete=False) as t:
        t.write(pre)
        tmp_md = t.name

    cmd = [
        "pandoc", tmp_md,
        "--pdf-engine=tectonic",
        f"--resource-path={REPO}",
        "-V", "geometry:margin=2.3cm",
        "-V", "fontsize=11pt",
        "-V", "linkcolor=blue",
        "-V", "colorlinks=true",
        "--toc", "--toc-depth=2",
        "-o", str(OUT),
    ]
    print("running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    pathlib.Path(tmp_md).unlink(missing_ok=True)
    if rc != 0:
        print(f"pandoc returned {rc}", file=sys.stderr)
        return rc
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
