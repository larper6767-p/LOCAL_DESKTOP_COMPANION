#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Friendly diagnostic for MARIN-LIVE friends.

Checks Python, required packages, and basic imports, then writes a summary to
diagnose_out.txt.  This is what you send back to the developer when something
isn't working.

Run:
    .venv\\Scripts\\python tools/diagnose_live2d.py
    .. or double-click diagnose.bat

It never sends anything anywhere — the output is a local text file you can
read and pass along yourself.
"""
from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "diagnose_out.txt"

REQUIRED = [
    "PySide6", "pygame", "numpy", "sounddevice", "keyboard",
    "cerebras", "ollama", "faster_whisper", "fishaudio",
    "live2d", "OpenGL", "dotenv",
]

OPTIONAL = ["pycaw"]


def _check_import(name: str) -> str:
    try:
        m = importlib.import_module(name.replace("-", "_"))
        ver = getattr(m, "__version__", "?")
        return f"{name}: OK (v{ver})" if ver != "?" else f"{name}: OK"
    except Exception as e:
        return f"{name}: FAILED ({type(e).__name__})"


def main() -> int:
    # Console on some Windows code pages (e.g. cp1252) can't encode the Chinese
    # model filename (简.model3.json).  Force UTF-8 so the printed diagnostic
    # never crashes with a UnicodeEncodeError even on a friend's machine.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    lines = []
    lines.append("MARIN-LIVE diagnostic")
    lines.append("=" * 40)
    lines.append(f"Python   : {sys.version.split()[0]} ({platform.architecture()[0]})")
    lines.append(f"Platform : {platform.system()} {platform.release()}")
    lines.append(f"Machine  : {platform.machine()}")
    lines.append("")

    lines.append("Packages:")
    for name in REQUIRED:
        lines.append("  " + _check_import(name))
    for name in OPTIONAL:
        res = _check_import(name)
        marker = " (optional)" if "FAILED" in res else ""
        lines.append("  " + res + marker)

    lines.append("")
    lines.append("Live2D model (characters/marin/live2d/):")
    model = Path(__file__).resolve().parent.parent / "characters" / "marin" / "live2d" / "简.model3.json"
    lines.append(f"  {model} : {'present' if model.exists() else 'MISSING (sprite mode expected)'}")
    lines.append("")

    # Is CUDA present? affects Whisper speed only.
    lines.append("Hardware:")
    try:
        out = subprocess.run(
            [sys.executable, "-c",
             "import torch; print('torch cuda:', torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=30)
        lines.append("  " + (out.stdout.strip() or out.stderr.strip() or "torch not installed"))
    except Exception as e:
        lines.append(f"  (could not probe cuda: {type(e).__name__})")
    nv = shutil.which("nvidia-smi")
    lines.append(f"  nvidia-smi: {'found' if nv else 'not found (CUDA likely absent — CPU Whisper will be used)'}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nWrote " + str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
