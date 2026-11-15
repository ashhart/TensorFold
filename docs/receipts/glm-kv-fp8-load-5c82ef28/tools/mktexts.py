"""mkcorpus.py's TEXTS_JSON from a TensorFold clone: every docs/**/*.md outside docs/receipts (prose) and, in path
order, zig/src/**/*.zig until 1,400,000 characters (code), read from COMMIT with git. Standard library only.

    python3 mktexts.py REPO COMMIT OUT_JSON
"""
import json
import subprocess
import sys


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout


def files(repo, commit, prefix, ext, skip=None):
    names = git(repo, "ls-tree", "-r", "--name-only", commit, prefix).split()
    return sorted(p for p in names if p.endswith(ext) and not (skip and p.startswith(skip)))


def main(repo, commit, out):
    full = git(repo, "rev-parse", commit).strip()
    d = {"commit": full, "prose": [[p, git(repo, "show", f"{full}:{p}")] for p in files(repo, full, "docs", ".md", "docs/receipts/")],
         "code": []}
    n = 0
    for p in files(repo, full, "zig/src", ".zig"):
        text = git(repo, "show", f"{full}:{p}")
        d["code"].append([p, text])
        n += len(text)
        if n > 1_400_000:
            break
    with open(out, "w") as f:
        json.dump(d, f)
    print(full, len(d["prose"]), "prose files,", len(d["code"]), "code files")


if __name__ == "__main__":
    main(*sys.argv[1:4])
