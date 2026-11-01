# GLM-5.3-Flash on CUDA

The native CUDA build serves GLM-5.3-Flash at tensor parallel over hosts: from NVIDIA's ModelOpt NVFP4 checkpoint
([1.0.12](../../CHANGELOG.md#1012)), on two ranks from Mia-AiLab's EXL3/TR3 checkpoint
([1.1.0](../../CHANGELOG.md#110), [the checkpoint](glm-5.3-flash.md#exl3)), and in [1.1.5](../../CHANGELOG.md#115) (#1050) from
NVIDIA's FP4-Local checkpoint as stored and from XSFP4, which keeps FP4-Local's tensors outside the routed experts.
Rank 0 serves HTTP and the other ranks follow its steps, with MTP drafts, concurrent streams and kept prompt states.
The [Python 0.6.6 recipe](glm-5.3-flash.md) and its `--tp`, `--rank` and `--master` flags belong to the frozen engine.

## Run

`TF_GLM_RANK` of `TF_GLM_WORLD` places each rank, and `NCCL_COMM_ID` names rank 0's address and port on the link
between the hosts ([placement](../multi-rank-receipts.md)). Give every rank the same arguments: before the weights load
the ranks compare their settings, model and build, and name the field and rank that differ. `--context` takes a
positive limit for GLM. On two GB10s, with 192.0.2.1 as rank 0's address on their direct link:

```bash
tensorfold pull nvidia/GLM-5.3-Flash-NVFP4@423acf37583782c51c142d145aef733d72943d93
# on rank 1
TF_GLM_RANK=1 TF_GLM_WORLD=2 NCCL_COMM_ID=192.0.2.1:29583 \
  tensorfold serve nvidia/GLM-5.3-Flash-NVFP4 --context 245760 --parallel 1
# on rank 0 (192.0.2.1)
TF_GLM_RANK=0 TF_GLM_WORLD=2 NCCL_COMM_ID=192.0.2.1:29583 \
  tensorfold serve nvidia/GLM-5.3-Flash-NVFP4 --context 245760 --parallel 1
```

Set `NCCL_SOCKET_IFNAME` and `NCCL_IB_HCA` for the link between the hosts as NCCL documents them. Each rank keeps a
TCP line to rank 0 on the port after `NCCL_COMM_ID`'s (29584 here), opened before NCCL: a rank that fails, refuses or
dies stops the others.

The contributed runs read that revision. The repository's later commit `da920bb0`, which declares the MTP layer as
BF16 in the quantization ignore lists, was not among them. On shipped 1.1.0 these settings loaded the whole checkpoint
in 302.1 s, and rank 0 reported `model 96.11 GiB; 1 stream at once, 3.16 GiB each at a 245760-token window, of 5.8 GiB
left after a 12.2 GiB reserve` and a 5.8 GiB prompt cache
([#1040](https://github.com/ashhart/TensorFold/issues/1040)).

## Requests

- `--vision` on every rank reads images and video: rank 0's tower encodes them and sends their rows and spans to the
  other ranks ([1.1.0](../../CHANGELOG.md#110)). Video needs `libtfvideo`
  ([video](../vision.md#video-on-the-native-server)). `TF_GLM_VISION_HOST=1` keeps the tower's bf16 weights in pinned
  host memory and copies each part of a pass into one device slot as it runs (240 MiB for GLM-5.3-Flash's tower instead
  of its 1.05 GiB), leaving rank 0 that much more room for KV pages. Checked on two RTX PRO 6000 Blackwell cards.
- Nonstreaming `/v1/completions` requests return `prompt_logprobs`, structured output follows its schema with drafts
  on, and required and named tool calls use the model's tool markup ([1.1.1](../../CHANGELOG.md#111): #1016 to #1018).
- `--kv-cache fp8` is an opt-in FP8 latent cache. BF16 remains the default, and its native quality and device
  qualification are pending ([1.1.2](../../CHANGELOG.md#112)).
- `TF_GLM_HOST_CACHE=1` on every rank keeps kept prompt states in pinned host memory instead of on the card, so a
  long conversation's next turn resumes when the card has no room left for its state (#1510). `--prompt-cache-gib`
  sizes the host cache on each rank (8 GiB unasked, at most half of the available memory), and the KV pool sets no
  device cache aside. Checked on two RTX PRO 6000 Blackwell cards at TP=2 with the EXL3 4bpw checkpoint.
- `TF_GLM_CHUNK_SOLO=4096` on every rank (opt-in) lets a prompt pass take 4,096 rows while no other stream is decoding.
  Unset or `2048`, every pass takes 2,048 rows and nothing extra is held. See [Prompt chunk size](#prompt-chunk-size).

## More ranks

`TF_GLM_WORLD=3` places a third rank the same way. From 1.1.2 the uneven KDA input projections of TP=3 (22, 21 and
21 heads) run instead of being refused; TP=3 device qualification is pending
([#1067](https://github.com/ashhart/TensorFold/issues/1067)). How a prompt's rows reach the other ranks
(`TF_GLM_PREFILL_REDUCE`) and the two-rank RoCE gathers (`TF_ROCE`) are in
[placement](../multi-rank-receipts.md#the-settings-the-ranks-agreed-on).

## Prompt chunk size

A prompt runs in passes of 2,048 rows. Kept prompt states sit on that grid, and a stream decoding beside a prompt waits
for each pass to end. A pass that takes more rows reads each routed expert's weights fewer times, so a lone prompt
comes sooner. `TF_GLM_CHUNK_SOLO=4096` lets a pass take two of those chunks when no other stream is decoding as it
starts. With one decoding, or with `prompt_logprobs` or image rows, the pass takes 2,048. A drafter's steps take one
chunk, so `--drafter` beside it is refused at load.

The grid does not move, so warm turns resume where they do unset, and streams decoding beside a prompt see the same
passes. Replies are the same bytes either way. Two costs remain:
- A request that arrives during a solo pass waits for that pass to end before its own prompt runs. After that, the
  passes are short again.
- The prompt buffers are sized for 4,096 rows for the whole run: 1.92 GiB more at TP=2 and 1.56 GiB at TP=3, which
  comes out of the room for KV pages and the prompt cache. In the runs below, TP=2's default cache goes from 4.7 to
  2.8 GiB.

| | TP=2 unset | TP=2 `=4096` | TP=3 unset | TP=3 `=4096` |
| --- | --- | --- | --- | --- |
| First token of a cold prompt, 8K / 32K / 128K (s) | 6.08 / 23.94 / 98.16 | 5.72 / 22.22 / 91.33 | 5.00 / 19.61 / 80.26 | 4.64 / 17.99 / 73.45 |
| First token of a warm turn, 10K history + 500 / 1,500 / 3,000 tokens (s) | 1.40 / 2.06 / 2.87 | 1.40 / 2.05 / 2.83 | 1.14 / 1.67 / 2.35 | 1.14 / 1.68 / 2.31 |
| First token of a warm turn, 40K history + 500 / 1,500 / 3,000 tokens (s) | 1.25 / 2.03 / 3.05 | 1.26 / 2.04 / 3.03 | 1.02 / 1.67 / 2.51 | 1.02 / 1.66 / 2.48 |
| Gap between a decoding stream's tokens beside a 32K prompt, median / longest (ms) | 1,474 / 1,749 | 1,478 / 1,751 | 1,219 / 1,444 | 1,219 / 1,459 |
| That stream's tokens during the prompt | 17.0 | 16.5 | 16.0 | 16.5 |
| First token of a short request sent 6 to 8 s into a 32K prompt, mean (lowest to highest) (s) | 1.27 (0.88 to 1.65) | 2.48 (1.35 to 3.47) | 1.24 (0.70 to 1.67) | 1.73 (1.07 to 2.67) |
| That prompt's first token (s) | 23.54 | 23.27 | 19.32 | 18.99 |

Each setting is two fresh servers (one at TP=2 for the cold row), `--parallel 4 --context 524288`, `TF_KV_PAGES=on`,
`Bizuayeu/GLM-5.3-Flash-XSFP4` with its MTP drafts. Cold prompts are Python source at `--prompt-cache-gib 0`, and a
repeated filler line gains 2 to 5% at the same sizes. The other rows ran with the prompt cache on (the default at TP=2,
2 GiB at TP=3). Three rows at TP=2 that waited on the heat guard are left out.

## Long runs on GB10

Long prompts and long runs of replies heat a GB10. The runbook's [heat bands](../../RUNBOOK.md#linux-and-cuda) (with a
1,035,295-token prompt at TP=3), [CPU pinning](../../RUNBOOK.md#gb10-cpu-pinning-for-long-cuda-runs) and
[GPU clock cap](../../RUNBOOK.md#gb10-gpu-clock-cap-for-long-cuda-runs) each carry a measured GLM-5.3-Flash run.
