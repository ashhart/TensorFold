# Reproduce the fidelity comparisons

`fidelity-prompts/manifest.json` contains the public source texts, actual token counts, dtype and hashes.
The `prose-513` fixture has 481 actual rows; its original name was a requested length, not a measured count.
Convert any saved ID array to the exact original NumPy fixture:

```sh
python3 -c "import json,numpy as np; np.save('TOKENS.npy', np.asarray(json.load(open('docs/receipts/lfm3/fidelity-prompts/code-64.ids.json')), dtype='<u4'))"
zig build tf-lfm3-forward -Dcpu=apple_m1
./zig-out/bin/tf-lfm3-forward MODEL_DIR TOKENS.npy native-prompt.bf16
./zig-out/bin/tf-lfm3-forward MODEL_DIR TOKENS.npy native-decode.bf16 decode
python3 tools/lfm3/fp64_reference.py --model-dir MODEL_DIR --tokens TOKENS.npy --output fp64.bf16
python3 tools/lfm3/compare_logits.py --native native-prompt.bf16 --reference fp64.bf16 --tokens TOKENS.npy --vocabulary 128000 --phase prompt --output prompt.json
```

The explicit decode mode prefills the first 63 rows of this fixture, then executes the last token with one row.
It writes exactly one vocabulary row and logs the current token, history length and SHA-256 of the raw u32LE history.
Slice the matching final FP64 row before comparing; do not label a row from a 32-row native block as decode:

```sh
python3 -c "import numpy as np; x=np.fromfile('fp64.bf16',dtype='<u2').reshape(-1,128000); x[-1].tofile('fp64-decode.bf16')"
python3 tools/lfm3/compare_logits.py --native native-decode.bf16 --reference fp64-decode.bf16 --tokens TOKENS.npy --vocabulary 128000 --phase decode --row-offset 63 --output decode.json
```

`six-width-fp64-prompt.json` reports all 64 prompt rows for both families at 4/6/8 bits.
`six-width-fp64-decode.json` reports the actual one-row decode for those six checkpoints.
`all-fp64-top-differences.json` reports every native/FP64 top difference in the 17 previously selected prompt cells,
with both margins, cross-contender gaps and the worst rows and columns. It does not select only the original 18 MLX disagreements.
The prompt receipts identify the earlier post-RoPE diagnostic binary; they are not final prompt-tile qualification.

To determine whether a discrepancy follows from production reduction rounding, replay the independent affine calculation:

```sh
python3 tools/lfm3/fp64_reference.py --model-dir MODEL_DIR --tokens TOKENS.npy --output rounded.bf16 --projection-rounding metal-row
python3 tools/lfm3/compare_logits.py --native native-prompt.bf16 --reference rounded.bf16 --tokens TOKENS.npy --vocabulary 128000 --phase prompt --output rounded.json
```

This option changes the affine projection reduction only. Remaining operations retain the FP64 reference arithmetic.
`row_projection_reference.py` independently unpacks the checkpoint, rounds each lane's operations to FP32,
then applies the reduction tree. It does not cast the final FP64 model output to manufacture agreement.
The fused affine step uses an FP64 multiply-add followed by FP32 rounding; the recorded checkpoint replay is not a general IEEE fmaf proof.
`projection-rounding-replay.json` records exact equality of all 1,579,008 captured pre-store FP32 outputs for the first VL4 projection.
No end-to-end tolerance or automatic fidelity pass is applied by these tools.

For image-conditioned text, pass independently computed projected rows as a numeric two-dimensional NumPy array:

```sh
python3 tools/lfm3/fp64_reference.py --model-dir MODEL_DIR --tokens TOKENS.npy --image-rows projected.npy --image-start START --output fp64-image-text.bf16
```

The reference validates the entire image-placeholder span, geometry and finite values, then inserts the rounded projected rows.
Supplying native projected rows isolates the decoder; supplying FP64 projected rows additionally measures vision propagation.
These two experiments must be named separately. The input option itself does not qualify a real-image vision fixture.

The optional CPU accelerator has no native inference or build dependency. Compile and select it explicitly:

```sh
cc -std=c11 -O3 -ffp-contract=off -shared -fPIC tools/lfm3/row_projection_replay.c -o row-replay.so
LFM3_ROW_REPLAY_LIBRARY="$PWD/row-replay.so" python3 tools/lfm3/fp64_reference.py --model-dir MODEL_DIR --tokens TOKENS.npy --output rounded.bf16 --projection-rounding metal-row
```

Only the explicit affine `fmaf` contracts operations. Input sums and placed inputs are prepared once without changing their order.
Both implementations reproduce all 1,579,008 recorded pre-store FP32 values exactly; integer fixtures exercise all three bit widths.
`rounding-reproduction.json` retains the original ideal-FP64 contenders and the rounded reference's maximum sets and margins.

Reproduce the real-image vision reference from the exact canonical patches emitted by the native diagnostic:

```sh
./zig-out/bin/tf-lfm3-vision MODEL_DIR --image IMAGE.png native-image.bf16 native-trace
python3 -c "import numpy as np; np.save('patches.npy', np.fromfile('native-image.bf16.patches.f32',dtype='<f4').reshape(280,1536))"
python3 tools/lfm3/vision_fp64_reference.py --model-dir MODEL_DIR --patches patches.npy --grid-h 14 --grid-w 20 --output fp64-image.bf16
```

`real-image-fp64.json` pins the official model card's cat URL, downloaded JPEG, canonical PNG and patch hashes.
The photographed image produces 70 projected rows. It reports all vision-row errors and separates identical-input decoder comparisons
from native/FP64 towers feeding their own rows. On identical FP64 projected rows, the 6/8-bit decoders match all 16 suffix top tokens;
the 4-bit decoder has one lowest-index difference with overlapping maximum sets. With each tower feeding its own rows, 4/6-bit match
all 16 suffix top tokens; 8-bit matches 14/16. Those different-input results are retained and are not an accepted end-to-end pass.
This is one real-image fixture, not a broad real-image benchmark.

The download URL can change its bytes; reproduce this receipt only with the recorded hashes.
The image itself is not redistributed here. The token fixture is the `tokens` array in `real-image-fp64.json`:

```sh
python3 -c "import json,numpy as np; np.save('image-tokens.npy',np.asarray(json.load(open('docs/receipts/lfm3/real-image-fp64.json'))['tokens'],dtype='<u4'))"
python3 -c "import numpy as np; x=np.fromfile('fp64-image.bf16',dtype='<u2'); np.save('projected.npy',(x.astype('u4')<<16).view('f4').reshape(-1,2048))"
./zig-out/bin/tf-lfm3-forward MODEL_DIR image-tokens.npy native-on-fp64-rows.bf16 images fp64-image.bf16 4
python3 tools/lfm3/fp64_reference.py --model-dir MODEL_DIR --tokens image-tokens.npy --image-rows projected.npy --image-start 4 --output fp64-image-text.bf16
```

The affine-only replay still had one disjoint VL6 reasoning position repeated at three fixture lengths.
`rounded-fp64-resolution.json` adds the explicitly named FP32 RMS-normalization and seven-tap convolution round sites for those cells:

```sh
LFM3_ROW_REPLAY_LIBRARY="$PWD/row-replay.so" python3 tools/lfm3/fp64_reference.py --model-dir MODEL_DIR --tokens TOKENS.npy --output rounded.bf16 --projection-rounding metal-row --norm-conv-rounding metal
```

All intermediate BF16 arrays in the first two convolutional layers then match native bytes exactly on the traced 64-row history.
The three repeated histories have zero disjoint maximum sets; 19 lowest-index differences remain across the combined 5,581-row view,
all with overlapping maximum sets. Each cell names its rounding sites. This does not apply a new tie rule or erase ideal-FP64 differences.
`earliest-stage-rounding.json` records both actual first-divergence fixtures, shared-helper raw captures and unchanged derived bounds.

### Real-image rounding causality

The ideal FP64 comparison remains in `real-image-fp64.json`, including the two disjoint 8-bit suffix rows and both margins. `real-image-rounding-reproduction.json` records a separate diagnostic: independently implemented CPU arithmetic with explicit production FP32 boundaries and Metal scalar elementary functions. It reproduces all 14 vision layers, projected image rows, 24 text layers, and all logits byte for byte. This is rounding causality evidence, not a new tolerance or an independent FP64 fidelity pass.

The optional Metal elementary backend reuses Apple `exp`, `exp2`, `cos`, `sin`, and `rsqrt` functions only. Matrix products, reductions, geometry, and online attention are replayed by the CPU reference. NumPy elementary functions remain separately selectable. Decoder attention replay currently rejects prompts above 1024 rows rather than approximating additional partitions.

```sh
# Build row-replay.so with the C accelerator command above, then use the frozen photo patches/tokens.
LFM3_ROW_REPLAY_LIBRARY="$PWD/row-replay.so" python3 tools/lfm3/vision_fp64_reference.py --model-dir MODEL_DIR --patches patches.npy --grid-h 14 --grid-w 20 --output rounded-image.bf16 --trace-dir rounded-vision --dense-rounding metal-row --operator-rounding metal-fp32 --elementary-backend metal
python3 -c "import numpy as np; x=np.fromfile('rounded-image.bf16',dtype='<u2'); np.save('rounded-image.npy',(x.astype(np.uint32)<<16).view(np.float32).reshape(70,2048))"
LFM3_ROW_REPLAY_LIBRARY="$PWD/row-replay.so" python3 tools/lfm3/fp64_reference.py --model-dir MODEL_DIR --tokens image-tokens.npy --image-rows rounded-image.npy --image-start 4 --output rounded-image-text.bf16 --trace-dir rounded-text --projection-rounding metal-row --norm-conv-rounding metal --attention-rounding metal-fp32 --elementary-backend metal
# Native vision tracing includes first_attention_q/k/v/context at the first attention layer.
./zig-out/bin/tf-lfm3-vision MODEL_DIR --image canonical.png native-image.bf16 native-vision
python3 tools/lfm3/vision_attention_rounding.py --native-trace native-vision --output-dir attention-rounding --rows 280 --capture
```

The last diagnostic evaluates the existing production attention helper at fixed native Q/K/V, then independently replays its FP32 score reduction and accumulator using captured elementary-function outputs. On this fixture all 1,254,400 scores and 291,200 raw accumulator values match exactly. Metal precise `exp` differs from FP64-rounded exp at 463,691 of 2,508,800 calls, each by one FP32 ULP. This primitive capture is explicitly scoped; it is not an independent reference for the full attention equation.
