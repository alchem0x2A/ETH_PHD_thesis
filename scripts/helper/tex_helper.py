# Handled several things about TeX output 

import re
from tempfile import gettempdir
from pathlib import Path
from subprocess import run
from .path_helper import tex_path, script_path


tex_file = tex_path / ".test.tex"
# header_file = tex_path / "header.tex"  # simple header file
header_file = tex_path / ".preamble_plot.tex"  # simple header file
# print("TeX file is ", tex_file)
width_file = script_path / ".width"

def _get_preamble():
    if header_file.exists():
        with open(header_file.as_posix(), "r") as f:  # back compatibility
            content = f.readlines()
            
            content = filter(lambda s: len(s) > 0, map(lambda s: s.strip(),
                                                       content))
            content = filter(lambda s: s[0] != "%", content)
            return list(content)
    else:
        return []

def _render_TeX(engine="lualatex"):
    if tex_file.exists():
        command = [engine, "--draftmode",
                   "--interaction=\"nonstopmode\"",
                   "--output-directory={0}".format(gettempdir()),
                   tex_file.resolve().as_posix()]
        proc = run(" ".join(command),
                   capture_output=True,
                   shell=True, timeout=60,
                   cwd=tex_path)
        return proc.stdout.decode("utf8")
    else:
        return ""


def _get_textwidth(*arg, **argkw):
    print(width_file)
    if width_file.is_file():     # read from previous results
        with open(width_file, "r") as f:
            s = f.readline().strip()
        try:
            width = float(s)
        except (ValueError, TypeError):
            width = None
        return width
    else:
        output = _render_TeX(*arg, **argkw)
        # print(output)
        # print(output, type(output))
        pattern = r"^\>\s+([\d\.]+)pt.+?$"
        match = re.findall(pattern, output, re.MULTILINE)
        try:
            width = float(match[-1])
            with open(width_file, "w") as f:
                f.write("{0:.3f}".format(width))
        except IndexError:
            return None

def clean_width_cache():
    import os
    """Remove the width_file in order to restart calculation"""
    try:
        os.unlink(width_file)
        return True
    except (OSError, FileNotFoundError):
        return False
