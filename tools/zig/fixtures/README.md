# Reference outputs for the Nemotron Intel GPU checks

`ref_nemotron.json` and `ref_nemotron_wide.json` are measurements of the TensorFold 0.6.4 CUDA engine (an NVIDIA GB10 in a
container image with torch 2.13 and CUDA 13), run on the checkpoint `TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit`
(revision `d9d758fb83953437f7263256b0d96157e2a348b8`): serial greedy decoding (temperature 0), the MTP head not loaded, no chat
template, raw token ids without a BOS token. Per generated token the files hold the token id, the five (wide: more) largest
log-probabilities and the log-probability of the chosen token, as `log_softmax` of the float32 conversion of the bf16 logits.

| File | Prompts | Steps | Size |
| --- | --- | --- | --- |
| `ref_nemotron.json` | 4 | 48 each | 38 KB |
| `ref_nemotron_wide.json` | 15, from 4 to 1,850 prompt tokens | 96 each (1,440 positions) | 299 KB |

Prompt sources: the four short prompts of the first file and the first four of the second ("The capital of France is", a Python
function header, "Once upon a time", a 305-token passage) were written for this test. The other eleven prompts of the second file are
slices of the first column of the wikitext-2 raw training text at the character offsets in their `text` field ("wikitext@OFFSET len
N"), stored as token ids of the Nemotron tokenizer. WikiText-2 is by Stephen Merity et al. ("Pointer Sentinel Mixture Models"),
distributed under the Creative Commons Attribution-ShareAlike 3.0 license (https://creativecommons.org/licenses/by-sa/3.0/); the
token ids in `ref_nemotron_wide.json` derive from it and carry the same license. The generated token ids and log-probabilities are
measurements of the engine named above on those prompts.

Used by `tools/zig/xpu_compare_ref.py` (forced-token and greedy agreement) and `tools/zig/xpu_compare_wide.py` (1,440 forced positions).
Per-operation fixtures are not shipped because they contain slices of the checkpoint's weights; `tools/zig/xpu_fixtures_*.py`
regenerate them from a local checkpoint.
