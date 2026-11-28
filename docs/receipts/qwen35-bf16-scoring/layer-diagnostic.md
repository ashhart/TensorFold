# Public BF16 layer diagnostic

This supplement addresses the maintainer's request on PR #2020: locate the first residual boundary where the native CUDA body exceeds Transformers BF16's own distance from promoted FP32 on the 32-token fixture. It adds observation to the production forward; it changes no kernel, sum, tolerance or weight value.

The checkpoint is public `Qwen/Qwen3.5-27B` at `fc05daec18b0a78c049392ed2e771dde82bdf654`. The exact 32 IDs are fixture index 1 in `fixtures.json`. Both reference arms preserve and byte-check the checkpoint's 96 stored FP32 parameters. The FP32 arm promotes the checkpoint's stored BF16 parameters; it is not a separate FP32 training checkpoint.

The native capture reads the residual buffer after the production input norm has materialized the preceding residual addition. Boundary 0 is embedding, boundaries 1–64 are the outputs of zero-based layers 0–63, and boundary 65 is final normalization. Synchronization and host copies occur only with the developer trace enabled. A second fresh-cache, untraced production forward must produce identical final BF16 logits or the capture exits nonzero.

The primary comparison covers hidden rows 0–30, which predict the 31 actually served prompt IDs at positions 1–31. The unscored last input row is retained in a separate all-32-row view. At each boundary the rule is `max(abs(native - FP32)) <= max(abs(Transformers BF16 - FP32))`, with no added tolerance. The report retains maxima, medians, RMS, worst coordinates and elementwise violations; none substitutes for the stated maximum rule. Final normalization is reported separately from residual layers.

The BF16 reference's captured final hidden rows must equal the public Transformers library entry point. The promoted-FP32 arm checks layers 0, 3 and 63 against independently cloned FP32 layers. Missing, inconsistent or nonfinite tensors are errors. Planted drift must make the comparison CLI exit 1; a large difference confined to the unscored final row must not alter the primary served-row verdict.

## Result

The first exceeding residual is **zero-based layer 0**, the first linear-attention block. Embeddings match exactly. Over the 31 served hidden rows, layer 0 has native-to-FP32 maximum **0.07647705078125**, versus the BF16 arm's own maximum **0.056326866149902344**. Its worst native coordinate is input row 21, hidden column 3994: native 15.5625, promoted FP32 15.63897705078125, BF16 reference 15.625. The all-32-row view finds the same first layer.

| Boundary | Native–FP32 worst | BF16–FP32 worst | Within own maximum |
| --- | ---: | ---: | --- |
| Embedding | 0 | 0 | passed |
| Layer 0 residual | 0.07647705078125 | 0.056326866149902344 | failed |
| Layer 1 residual | 0.09848594665527344 | 0.09848594665527344 | passed |
| Layer 2 residual | 0.19491863250732422 | 0.1948375701904297 | failed |
| Layer 3 residual | 0.199066162109375 | 0.188201904296875 | failed |
| Layer 63 residual | 4.7076416015625 | 7.054046630859375 | passed |
| Final normalization | 0.31170082092285156 | 0.4594573974609375 | passed |

The rule fails at **18/64 residual layers**. The complete 66-boundary table, both row scopes, worst coordinates, medians, RMS and all 198 unchanged arrays are in the linked packet. Some later boundaries return inside the maximum; final normalization passing this hidden-state rule does not overturn the separately failing served-logprob rule. No hidden-distance allowance or new tolerance is used.

Native producer: `9416f0cf94bddbea49becb4c14047d0c16d08faf`; Transformers producer: `b54d8ff7691385fe420399e72272c932a881e29e`; comparison: `c6930073d5192663bd5726ff74a8fa426f7a06aa`; main base: `002c9e03eecf9b10b83a73b9149dc7f49d50368a`. Later comparison/docs changes do not relabel those executions. Native traced/untraced logits match; both reference cross-checks pass, and all 11 public shard hashes match pinned LFS after capture.

The GPU is one RTX PRO 6000 Blackwell Max-Q workstation card, sm_120. The host has a Ryzen 9 9950X3D and 128 GB RAM class. Reference limits are 16 CPUs/96 GiB; the native capture has no CPU/RAM cgroup limit. This is a synchronized diagnostic, not a four-CPU/32-GiB serving or latency benchmark. Original running deployments and active-service health were restored.

[Compact summary](layers.json), [archive hashes](asset.json), and [immutable archive/member manifest](https://raw.githubusercontent.com/akol1/TensorFold/7f54a5445ef4defa879b2b1fa76d01a64afdb5f4/qwen35-bf16-layers-20261105.manifest.json). Extract all three layer archives into one directory to reproduce the comparison. The 217-file packet retains complete build/capture logs and a superseded partial-export test-setup failure; the complete current-source reference suite passes 20/20.

The subsequent integration uses main `9200a3064153b8041930f61f5ba92102d4bea692` and the refreshed parent `d19ff3f075eb144870af1af159de1eccafd0b69b`. The complete Qwen CUDA family and CUDA kernel source trees are byte-identical to the native diagnostic producer `9416f0cf`; the new upstream changes and parent test-registration placement are checked separately. This source comparison does not relabel the device execution.

## Reproduce

Use Zig 0.17.0 and a CUDA toolchain supported by the repository. The recorded run uses NVCC 13.0.88 and one sm_120 GPU. The CPU reference uses Python 3.12.3, Torch 2.15.0+cu132, Transformers 5.22.1, accelerate 1.15.0, safetensors 0.8.0 and NumPy 2.5.4, with eager attention and CUDA disabled. CPU fallback implementations are disclosed in the reference log.

```sh
python3 -c 'import json; print(json.dumps(json.load(open("docs/receipts/qwen35-bf16-scoring/fixtures.json"))[1],separators=(",",":")))' > ids.json
zig build fatbins kernel-contract-fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=80,86,89,120,121
zig build qwen35-hidden-reference -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseFast -Dfatbins=zig-out/fatbin
zig-out/bin/tf-qwen35-hidden-reference MODEL_DIR ids.json native
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 python3 tools/zig/qwen35_layer_reference.py --model MODEL_DIR --ids ids.json --output reference
python3 tools/zig/qwen35_layer_compare.py --native native --reference reference --output comparison.json
python3 -m unittest discover -s tools/zig -p 'test_qwen35*.py' -v
```

Use the existing GPU lock and admission guard, with one model process per test machine. Each reference output directory must be empty. Preserve the comparison's actual nonzero exit when the rule fails.

This is one public fixture on one CUDA GPU, with a CPU reference. It does not qualify other prompts, lengths, devices, multiple ranks, performance or whole-model equivalence. The existing four per-length failures and historical FP64 head receipt remain unchanged. A first exceeding boundary locates observed drift; it does not alone establish a single faulty accumulation or an unavoidable format floor.
