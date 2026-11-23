"""Compare battery outputs: usage compare.py REF.json OTHER.json [...]; ids and text per key against the first file."""
import json
import sys

files = sys.argv[1:]
runs = [json.load(open(f)) for f in files]
ref = runs[0]
print("requests", len(ref), "files", files)
for name, run in zip(files[1:], runs[1:]):
    bad = [k for k in ref if k not in run or run[k]["ids"] != ref[k]["ids"] or run[k]["text"] != ref[k]["text"]]
    print(f"{name}: {len(ref) - len(bad)} of {len(ref)} equal to {files[0]}", "UNEQUAL:" if bad else "", bad)
# within one run, solo, parallel and again must agree
for name, run in zip(files, runs):
    pairs = 0
    bad = []
    for k in run:
        if "/solo/c0/" in k:
            for other in (k.replace("/solo/", "/par/"), k.replace("/solo/", "/again/")):
                pairs += 1
                if run[other]["ids"] != run[k]["ids"]:
                    bad.append(other)
    print(f"{name}: solo = parallel = again for {pairs - len(bad)} of {pairs}", bad)
print()
print("key".ljust(22), *[f"{f.split('/')[-1][:14]:>26}" for f in files])
for k in ref:
    cells = []
    for run in runs:
        r = run[k]
        s = r["spec"] or {}
        cells.append(f"p{r['prompt_tokens']} c{r['cached_tokens']} d{s.get('drafted', '-')}/a{s.get('accepted', '-')} {r['ttft'] or 0:.2f}s")
    print(k.ljust(22), *[c.rjust(26) for c in cells])
