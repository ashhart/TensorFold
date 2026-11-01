# Public tiled LFM3 status


The [maintainer response](https://github.com/ashhart/TensorFold/issues/1491#issuecomment-6537143418)
ratifies “no further from an fp32 run than the reference’s own bf16,” with every top-1 difference and margin
retained. Partial comparisons stay partial. It accepts approximately 0.92x standard decode for the opt-in text
family and records the cold-prefill gap as separate follow-up work rather than a blocker for the text slice.
The release context tier is 32k until long-context reference fidelity is measured; longer memory/time runs
must not be labelled fidelity qualification. This policy does not automatically pass the tiled candidates.

The [public-revision runtime receipt](m5-public-tiled-prompt-runtime.json) binds default and
enabled binaries to `8df18abc4`. M5 TEXT4 cold TTFT fell from 0.858745 to 0.555548 seconds at 2k and
4.619115 to 2.413850 seconds at 8k (35.31% and 47.74% lower). Exact rendered input IDs, zero prompt-cache
use, 16 output tokens, matching output text hashes and stable ABBA legs are retained. These are TTFT
measurements, not sustained decode or token-level ITL results. The source-bound test probe exercised 72
FFN and 22 attention dispatches and reproduced the private candidate logits exactly. The bounded
prompt/continuation logit RMS/max checks pass; the proposed probability diagnostics still breach.

The [post-change stage profile](m5-public-tiled-stage-profile.json) verifies every chunk's
logits against uninstrumented same-route execution in eight mirrored diagnostic runs. At 8k, the enabled
stage sum is 2.421 seconds: FFN down accounts for 25.38%, attention 14.98%, gate 14.47%, up 14.46%, and
convolution input 13.43%. This identifies the next measured optimization targets; it is not evidence of
hardware saturation or a separate served speed qualification.
