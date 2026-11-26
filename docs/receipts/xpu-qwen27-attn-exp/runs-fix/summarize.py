# usage: cd runs-fix && python3 -I summarize.py   (per cell: token_sha256 set and median decode tok/s of the measured requests)
import json,glob,statistics as st,sys
R=[]
for f in sorted(glob.glob("q*.jsonl")):
    for l in open(f):
        r=json.loads(l); r["f"]=f[:-6]; R.append(r)
cells={}
for r in R:
    cells.setdefault((r["depth"],r["mode"]),[]).append(r)
for (d,m),rs in sorted(cells.items()):
    hs={r["token_sha256"] for r in rs}
    sides={}
    for r in rs:
        if not r["warmup"]: sides.setdefault(r["f"],[]).append(r["decode_tok_s"])
    print(d,rs[0]["prompt_tokens"],m,"n",len(rs),"ok",all(r["ok"] for r in rs),"hashes",[h[:16] for h in hs],{k:round(st.median(v),2) for k,v in sides.items()})
