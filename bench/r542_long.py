import json, os, sys
sys.path.insert(0, os.path.expanduser("~/rehearse"))
import rehearse as R

def arm(phase, env):
    S = R.Serve(phase, 0, extra_env=env)
    boot = S.start()
    text = R.corpus()
    out = []
    out.append(R.chat(S, R.msgs(text[:88000] + R.S), False, 128, tag="21k-plain-128"))
    out.append(R.chat(S, R.msgs(text[:88000] + R.S), True, 128, tag="21k-draft-128"))
    stop = S.stop()
    return {"phase": phase, "env": env, "boot": boot, "arms": out, "stop": stop}

res = []
for phase in sys.argv[1:]:
    env = {"TF_DSV41_INDEX_SCORE_ONE": "0"} if phase.endswith("-off") else {}
    r = arm(phase, env)
    res.append(r)
    print(json.dumps(r, default=str))
open("/tmp/r542_long.json", "w").write(json.dumps(res, indent=1, default=str))
