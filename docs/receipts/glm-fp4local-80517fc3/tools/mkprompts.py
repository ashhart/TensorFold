"""In tools/: the token files serve12.sh's checks send, the same for every checkpoint and build: long.json (Z0's three
decode-check prompts three times over, 18,789 tokens, bench_cache's cold prompt) and p9600.json and p38400.json
(openai_aligned.py tokens: 9,600 ids from position 0 and 38,400 from position 2,100, sharing no prefix).

    python3 mkprompts.py
"""
import json
import subprocess
import sys

kinds = json.load(open("dc.json"))["kinds"]
with open("long.json", "w") as f:
    json.dump([t for _ in range(3) for k in kinds for t in k["prompt"]], f)
for n, skip in ((9600, 0), (38400, 2100)):
    subprocess.run([sys.executable, "openai_aligned.py", "tokens", str(n), str(skip), "dc.json", f"p{n}.json"], check=True)
print(" ".join(f"{name} {len(json.load(open(name)))}" for name in ("long.json", "p9600.json", "p38400.json")))
