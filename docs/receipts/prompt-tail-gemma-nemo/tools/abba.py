"""Run arms base,new,new,base (each a fresh server, page cache dropped), then a gate. usage: abba.py ARM_TOOL BASE_BIN NEW_BIN MODEL OUT GATE [GATE_ARGS...]"""
import os, subprocess, sys
tool, base, new, model, out, gate, *gargs = sys.argv[1:]
here = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(out): sys.exit(f"{out} exists")
worst = 0
for i, arm in enumerate(("base", "new", "new", "base")):
    rc = subprocess.run([sys.executable, "-u", os.path.join(here, tool), f"{arm}-{i}", base if arm == "base" else new, model, out], timeout=1500).returncode
    print(f"STEP {arm}-{i} rc={rc}", flush=True); worst = worst or rc
rc = subprocess.run([sys.executable, os.path.join(here, gate), out, *gargs]).returncode
print(f"STEP gate rc={rc}", flush=True); sys.exit(worst or rc)
