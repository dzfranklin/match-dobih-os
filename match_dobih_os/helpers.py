import os

import shutil
import unicodedata
import urllib.request


def download(url: str, fpath: str):
    if not os.path.exists(fpath):
        fdir = os.path.dirname(fpath)
        if fdir != "":
            os.makedirs(fdir, exist_ok=True)

        print("Downloading", url)
        with urllib.request.urlopen(url) as resp:
            with open(fpath, "wb") as f:
                shutil.copyfileobj(resp, f)


def force_ascii(text: str) -> str:
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
