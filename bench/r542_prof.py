import json, os, sys, urllib.request
sys.path.insert(0, os.path.expanduser("~/rehearse"))
import rehearse as R
env = {}
if sys.argv[1] == "off":
    env = {"TF_DSV41_DENSE_COL_BF16": "0", "TF_DSV41_INDEX_SCORE_ONE": "0"}
S = R.Serve("r542-prof2-"+sys.argv[1], 0, extra_env=dict({"TF_DSV41_ATTN_PROF":"1","TF_DSV41_ATTN_PROF_LAYERS":"2"}, **env), parallel=1)
print("boot", S.start()["pid"])
text = R.corpus()[:84000]
msgs=[{"role":"user","content":text+"\n\nName the one colour the passage mentions most."}]
body={"model":R.MODEL_NAME,"messages":msgs,"max_tokens":2,"temperature":0,"top_p":0.95,"top_k":20,"seed":1234,"draft":False,"stream":True,"stream_options":{"include_usage":True},"chat_template_kwargs":{"enable_thinking":False}}
req=urllib.request.Request(R.BASE+"/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
out=""
with urllib.request.urlopen(req,timeout=7200) as r:
    for line in r:
        line=line.decode().strip()
        if not line.startswith("data:"): continue
        d=line[5:].strip()
        if d=="[DONE]": break
        j=json.loads(d); ch=j.get("choices") or []
        if ch and ch[0].get("delta",{}).get("content"): out+=ch[0]["delta"]["content"]
print("reply", repr(out))
print(S.new_log())
print("STOP", S.stop())
