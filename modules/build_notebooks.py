#!/usr/bin/env python3
"""
Generate notebook.ipynb in each module_* folder from colab.py.

Usage (from repo root or modules/):
  python3 modules/build_notebooks.py

Requires Python 3.9+ (stdlib only).
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def extract_docstring(src: str) -> str:
    m = re.search(r'"""(.*?)"""', src, flags=re.DOTALL)
    if not m:
        return ""
    return m.group(1).strip()


def extract_pip_commands(src: str) -> list[str]:
    cmds: list[str] = []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("# !pip"):
            cmds.append(s[2:].strip())  # '# !pip' -> '!pip'
        elif s.startswith("#!pip"):
            cmds.append(s[1:].strip())
        elif s.startswith("!pip"):
            cmds.append(s)
    return cmds


def code_without_cookie_docstring_pip(src: str) -> str:
    lines = src.splitlines()
    i = 0
    if lines and "coding" in lines[0]:
        i = 1
    body = "\n".join(lines[i:])
    body = re.sub(r'^\s*""".*?"""\s*', "", body, count=1, flags=re.DOTALL)
    out: list[str] = []
    skip_optional_banner = False
    for line in body.splitlines():
        st = line.strip()
        if "Optional: Colab install" in line or "Optional: Colab install" in st:
            skip_optional_banner = True
            continue
        if st.startswith("# !pip") or st.startswith("#!pip"):
            continue
        if st.startswith("!pip"):
            continue
        out.append(line)
    text = "\n".join(out).strip()
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def pip_cell_content(commands: list[str]) -> str:
    if not commands:
        return (
            "# No pip install line found in colab.py; add packages here if needed.\n"
            "# %pip -q install numpy pandas\n"
        )
    # Prefer Jupyter magic (works in Jupyter, VS Code, Colab)
    lines = []
    for c in commands:
        if c.startswith("!pip"):
            lines.append("%" + c[1:])  # !pip -> %pip
        else:
            lines.append(c)
    return "\n".join(lines) + "\n"


def make_notebook(md_title: str, doc: str, pip_src: str, code_src: str) -> dict:
    md_lines = [f"# EC 410K / EC 610I — {md_title}\n", "\n"]
    if doc:
        md_lines.append(doc + "\n\n")
    md_lines.append(
        "_Generated from `colab.py`. Regenerate with `python3 modules/build_notebooks.py`. "
        "Run cells top to bottom (install cell first)._ \n"
    )

    cells: list[dict] = [
        {"cell_type": "markdown", "metadata": {}, "source": md_lines},
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": pip_src.splitlines(keepends=True),
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": code_src.splitlines(keepends=True),
        },
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def module_title(folder_name: str) -> str:
    if not folder_name.startswith("module_"):
        return folder_name.replace("_", " ").title()
    rest = folder_name.removeprefix("module_")
    m = re.match(r"^(\d+)_(.+)$", rest)
    if not m:
        return rest.replace("_", " ").title()
    num = int(m.group(1))
    slug = m.group(2).replace("_", " ").title()
    return f"Module {num}: {slug}"


def main() -> None:
    root = Path(__file__).resolve().parent
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("module_"):
            continue
        colab = d / "colab.py"
        if not colab.is_file():
            continue
        src = colab.read_text(encoding="utf-8")
        doc = extract_docstring(src)
        pips = extract_pip_commands(src)
        code = code_without_cookie_docstring_pip(src)
        nb = make_notebook(module_title(d.name), doc, pip_cell_content(pips), code)
        out = d / "notebook.ipynb"
        out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        print(f"Wrote {out.relative_to(root)}")


if __name__ == "__main__":
    main()
