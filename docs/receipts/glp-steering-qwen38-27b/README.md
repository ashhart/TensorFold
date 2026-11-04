# Control vectors on Qwen3.8-27B: the steering kernel and the vector it applies

Branch `feature/glp-steering`. The work is a native control-vector path: a GLP
container is read, a direction is applied to the residual stream on Metal, and a
producer derives such a direction from the model's own activations.

## Environment

- Apple M4 Pro (`applegpu_g16`), macOS 26.5.1 (25F80), 48 GiB of RAM, Metal.
- Zig 0.17.0, the version the repository pins.
- TensorFold 1.1.6, branch `feature/glp-steering`.
- Checkpoint `TensorFold/Qwen3.8-27B-MLX-4bit` at
  `22d8d538154e0e5b6f8dcb1fcc74a60a0798104c`, MLX 4-bit.
- Serving settings: strength 1.0, 300 new tokens per prompt, the server's own
  sampling defaults (temperature 1.0, top_k 20, top_p 0.95).

## The vector

- 49 directions, layers 10 to 58 zero-based, width 5120, mode `project`, alpha 1.
- Derived from the model's own activations with a 0.5% massive-activation mask:
  26 of 5120 dimensions are left out of each layer's direction, and the screen
  reports 701 for this checkpoint.
- The producer takes 17.7 s and 18.4 s on two runs on this machine, and a
  re-derivation from the same checkpoint and the same prompt set is
  byte-identical.
- Published as `BlivionIaG/Qwen3.8-27B-refusal-cvec-GLP-49`, file
  `Qwen3.8-27B-refusal-cvec-GLP-49-L10-58-a1.gguf`, file sha256
  `fb0fc832305b37db0af64295c8fa6f6d3e26a7719904199a7488959aec524a2d`, directions
  digest `9a50f7160fa2f0a41e7fa868c791d372b0a374f9365cc98c25d52abe96ce9d8d`.
- The format's reference validator reads the file as it came back from the Hub and
  reports 0 FAIL and 0 WARN.
- Parity against an unmodified `msuiche/weightless` checkout: 5 files out of 5
  agree, on the layer ids, the metadata, every direction's bytes and the digest.

## What the model does with it

The harmful column is the share of harmful prompts the model still refuses, and
the benign column the share of the harmless twins it refuses. Lower is better in
both: a vector that removes a capability instead of a refusal shows up in the
second.

Derivation set, thirty form-matched pairs of the repository's own prompts:

| vector | harmful refused | benign refused | replies coherent |
|---|---:|---:|---:|
| none | 30/30 | 0/30 | 60/60 |
| published GLP-49, per layer | 19/30 | 6/30 | 60/60 |
| published GLP-63, pooled | 2/30 | 0/30 | 60/60 |
| ours, unmasked | 18/30 | 7/30 | 60/60 |
| ours, masked at 0.005 | 0/30 | 0/30 | 60/60 |

Held-out set, twenty pairs written after the derivation, over topics the
derivation never names:

| vector | harmful refused | benign refused | replies coherent |
|---|---:|---:|---:|
| none | 20/20 | 5/20 | 40/40 |
| published GLP-49, per layer | 9/20 | 6/20 | 40/40 |
| ours, masked at 0.005 | 0/20 | 0/20 | 40/40 |
| ours as an add-mode file | 20/20 | 20/20 | 40/40 |

The benign column is read against five, not against zero: the base model refuses
five of the twenty twins on its own. The refusal rate is a lower bound, because
it counts phrases and a vector can bend the wording of a refusal without
removing it. Reading the replies is what settles it.

The strength is flat from 0.5 to 3; at 4 the replies go empty. A strength of 1e-6
reproduces the baseline exactly.

## The kernel

- One dispatch per steered layer: the norm and the projection in the same pass
  over the row, in `zig/kernels/metal/qwen27/norm_steer.metal`.
- The float64 reference, the bounds, the fp32-sum rule and the served envelope
  are in `zig/src/core/contract/steer.zig`, and the family's own unit test there
  holds the kernel's arithmetic inside the bound the spec declares.
- The launch is registered for the limits gate in
  `zig/src/core/contract/steer_records.zig`: 320 threads and 80 threadgroup
  bytes, against M1 to M5 and sm_80, sm_86, sm_89, sm_120 and sm_121.
- The device gates run the kernel on the GPU and compare it with float64. On
  this M4 Pro the steering kernel passes gates 1 to 4 in both modes it serves,
  the two controls are red as planted, and the limits gate passes with the 320
  threads and the 80 bytes of threadgroup memory the launch declares:

  ```
  PASS nonfinite qwen27 steer project: 20480 non-finite and 5120 finite outputs match the float64 classes
  PASS repeat    qwen27 steer project: five runs byte-identical (10240 bytes), inputs and guards untouched, no overlap
  PASS shapes    qwen27 steer project: 4 shapes within the float64 bound (worst 0.995 of the bound), 0 edges refused by the kernel
  PASS sums      qwen27 steer project: exact sums at k 5120, where fp16 sums would miss by infx and bf16 by 267x
  PASS nonfinite/repeat/shapes qwen27 steer add (sums: N/A, declares none sums)
  CONTROL qwen27 steer project nan control:  red at nonfinite, as planted
  CONTROL qwen27 steer project sums control: red at sums, as planted
  SUMMARY 2 kernels: 2 pass gates 1-4, 0 fail, 0 known findings; 2 controls: 2 red as planted, 0 missed; 0 limit problems
  ```

## Tests

- 66 new Zig tests: 26 in the container reader, 4 in the publication metadata, 13 in
  the serve surface, 4 in the family, 8 in the producer, 5 in the two prompt sets
  and 6 in the contract; 8 gate unit tests and 6 parity unit tests in Python.
- `zig build test`: 552 steps succeeded, 161 of 162 tests passed and one skipped.
- CI: the format check, the code, the tests and the tools are green on Linux,
  Windows and macOS. The app's macOS packaging job is cancelled by the runner
  pool, four attempts across two runs, while the base tree's own runs are
  cancelled the same way in the same window; the job's four steps were run here
  under the job's own pins, and they pass.
- The lean checker finds 427 problems on this branch against 428 on the base it
  merged, `b30d5b9d3`: zero new.

## Not covered

- CUDA and HIP: this machine has no NVIDIA or AMD GPU, so neither backend's
  contract run nor its `SUMMARY` line exists for this branch.
- Apple chips other than this one: the limits for M1 to M5 are checked without a
  GPU by `zig build test-kernel-contract`, and the device gates ran on this
  M4 Pro.
- Decode and prompt speed before and after: the feature did not exist on this
  branch before the change, so there is no earlier build to measure against. The
  unsteered path is unchanged by construction, because the projection is skipped
  when no direction is in force.
- The Gemma 4 rounded-dense device gate cannot run on this chip, which has no
  tensor units. That gate belongs to the base tree and stops the full contract
  run on this machine before it reaches any kernel of this branch.

Only this prose is retained. The raw receipts, the published file and the
rotary accuracy receipt are machine- or host-specific and are not part of the
tree.
