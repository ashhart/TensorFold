"""Fixtures for `tf-cuda-test glm`: the Python engine's GLM-5.3-Flash kernels and layers on real RED-SNOW tensors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# the families' fixtures: siblings in this folder, which leads sys.path when the script runs
import glm_oracle_gaps
import glm_oracle_glue
import glm_oracle_kda
import glm_oracle_latent
import glm_oracle_sparse
import glm_oracle_layer
import glm_oracle_dsa
import glm_oracle_layer_dsa
import glm_oracle_mlp
import glm_oracle_head


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="the checkpoint folder (the recipe's serve view)")
    ap.add_argument("--out", required=True, help="the fixture folder to (over)write")
    a = ap.parse_args()
    out, model_dir = Path(a.out), Path(a.model)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "model": str(model_dir)}
    manifest["glue"] = glm_oracle_glue.write(out, model_dir)
    manifest["kda"] = glm_oracle_kda.write(out, model_dir)
    manifest["sparse"] = glm_oracle_sparse.write(out, model_dir)
    manifest["latent"] = glm_oracle_latent.write(out, model_dir)
    manifest["gaps"] = glm_oracle_gaps.write(out, model_dir)
    manifest["layer"] = glm_oracle_layer.write(out, model_dir)
    manifest["dsa"] = glm_oracle_dsa.write(out, model_dir)
    manifest["mlp"] = glm_oracle_mlp.write(out, model_dir)
    manifest["layer_dsa"] = glm_oracle_layer_dsa.write(out, model_dir)
    manifest["head"] = glm_oracle_head.write(out, model_dir)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(manifest) - 2} families -> {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
