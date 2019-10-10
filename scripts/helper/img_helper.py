# Helper functions for handling image and pdf files
from subprocess import run
from pathlib import Path
from shutil import which
import re

def _get_img_type(fname):
    img_types = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    fname = Path(fname)
    suffix = fname.suffix
    if suffix == ".pdf":
        return "pdf"
    elif suffix in img_types:
        return "img"
    else:
        return None

def _get_pdf_size(fname):
    """Get the size of a pdf file"""
    fname = Path(fname)
    if not fname.suffix == ".pdf":
        raise ValueError("The input file is not a pdf file!")
    cmd = "pdfinfo"
    if which(cmd) is None:
        raise ValueError("You should install poppler to use pdf dimension")
    cmds = [cmd, fname.as_posix()]
    proc = run(cmds, capture_output=True)
    output = proc.stdout.decode("utf8")
    pattern = r"Page size:\s+([\d\.]+?)\s+x\s+([\d\.]+?)\spts"
    match = re.findall(pattern, output)
    if len(match) > 0:
        w, h = match[0]
        return w, h
    else:
        raise ValueError("Pdf file seems to be having problems!")

def _get_img_size(fname):
    """Get the size of an image using the PIL library"""
    try:
        from PIL import Image
    except ImportError:
        raise
    try:
        im = Image.open(Path(fname).expanduser())
    except (OSError, FileNotFoundError):
        raise
    w, h = im.size
    return w, h

    

if __name__ == '__main__':
    import sys
    fname = sys.argv[1]
    # _get_pdf_size(fname)
    _get_img_size(fname)
