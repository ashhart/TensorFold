import re,sys,glob,os
# usage: ysum.py logs... : serial attn_decode ms/token and drafted attn_decode ms/round, merge, eq_serial
for f in sys.argv[1:]:
    s=open(f,errors="ignore").read()
    m=re.search(r"drafted: \d+ tokens in (\d+) rounds.*equal to serial: (\w+)",s)
    d=re.search(r"^decode: (\d+) tokens",s,re.M)
    if not m or not d: print(os.path.basename(f),"no result"); continue
    rounds=int(m.group(1)); nd=int(d.group(1))
    def sec(label):
        i=s.index("GPU time, "+label)
        j=s.find("GPU time,",i+5)
        return s[i:j if j>0 else len(s)]
    def ms(t,k):
        r=re.search(k+r"\s+([\d.]+) ms",t); return float(r.group(1)) if r else float("nan")
    sd=sec("decode"); dr=sec("drafted rounds")
    print("%-34s serial attn_decode %7.3f ms/tok | drafted %7.2f ms/round merge %6.2f | rounds %3d eq %s" % (os.path.basename(f), ms(sd,"attn_decode")/nd, ms(dr,"attn_decode")/rounds, ms(dr,"attn_merge")/rounds, rounds, m.group(2)))
