# GLM-5.3-Flash on CUDA

The native CUDA build serves GLM-5.3-Flash at tensor parallel over hosts: from NVIDIA's ModelOpt NVFP4 checkpoint
([1.0.12](../../CHANGELOG.md#1012)), on two ranks from Mia-AiLab's EXL3/TR3 checkpoint
([1.1.0](../../CHANGELOG.md#110), [the checkpoint](glm-5.3-flash.md#exl3)), and in [1.1.5](../../CHANGELOG.md#115) (#1050) from
NVIDIA's FP4-Local checkpoint as stored and from XSFP4, which keeps FP4-Local's tensors outside the routed experts.
Jammed XSFP4, on `main` after 1.2.2 ([#2041](https://github.com/ashhart/TensorFold/pull/2041)), holds those experts at block 32 in the `csf0_v1` form and loads the same way.
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
- With KV pages (the default at two or more ranks), the pool sets the windows the room holds aside before the prompt
  cache, which gets what they leave of its 16 GiB. `--prompt-cache-gib N` sets N GiB aside first, after the engine's
  own window and a fixed part and page for each other stream, and the pool takes the rest.
- `TF_GLM_KV_HOST=1` on every rank (with KV pages, BF16) keeps the KV pages' latent rows in pinned host memory: the
  device keeps the pooled index keys, and each sparse call copies only the rows it picked into device slots. The KV
  pool then costs the card 768 bytes a token instead of 13,056, so a 1,048,576-token window fits beside the weights.
  `TF_GLM_KV_HOST_GIB` sizes the host pages (one full window unasked); `TF_GLM_KV_HOST_SHARED=1` gives the ranks of
  one host a single copy in `/dev/shm`. Checked on two RTX PRO 6000 Blackwell cards at TP=2.
- `TF_GLM_CHUNK_SOLO=4096` on every rank (opt-in) lets a prompt pass take 4,096 rows while no other stream is decoding.
  Unset or `2048`, every pass takes 2,048 rows and nothing extra is held. See [Prompt chunk size](#prompt-chunk-size).
- `TF_GLM_DENSE_BITS=4` (EXL3 at TP=2, off by default) repacks the BF16 dense layers to 4 bits at load. Decode is
  faster (24.8 to 35.0 tok/s in one user's run), but replies differ from the default, so it is lossy and stays opt-in
  (#1761).

## Verified copy drafts

`TF_GLM_COPIES=1` on rank 0 tries the lane host's cost-priced root/context copies, with a four-token history match.
Unset or 0 retains the existing policy. Copies use the normal target verify/partial-keep path and require no external
drafter; `--no-drafts` and `"draft": false` remain plain. The prices timed at load set the rows and observed copy
rejections can cut an offer to zero rows. The opt-in's head confidence is a maximum prefix; cost may choose plain.
See [copy pricing and the required TP=2 comparison](cuda-lane-copies.md#glm-on-cuda). Device exactness and speed are
pending, so keep the default policy until that checkpoint's complete cost and reply checks pass (#1910).

## Window prices

Decode windows reach 32 rows and MTP chains 15 drafts. At load, rank 0 runs real rounds over a fixed text on every rank:
windows of 1, 2, 3, 4, 6, 8, 12, 16, 24 and 32 rows, straight lines between them, and a draft step. It also races the
classic NVFP4 expert kernel against the factored one and keeps the faster, which gives the same bits. The startup log
names both: `glm: windows (timed): 9.91 ms at 1 row, 11.25 at 4, 26.03 at 32, a draft 0.20 ms; experts factored`.

The times stay in `~/.cache/tensorfold/window-costs`, one table per build, GPU, checkpoint, drafter and setting that
moves a window's time. A new build's first load takes about 2 s longer on the cards below and `(kept)` loads none.
With `--no-drafts` nothing is timed. Timings steer windows, never tokens.

## Four cards in one host

NCCL allows direct card-to-card transfers on AMD hosts only up to two GPUs, so set these on every rank at three or more:

```sh
export NCCL_P2P_LEVEL=SYS NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4 TF_GLM_PREFILL_REDUCE=gather
```

`NCCL_P2P_LEVEL=SYS` takes a 32 MiB all-gather from 8.3 to 36.9 GB/s. `gather` replaces the default `split`, whose direct
exchanges took 86% of a rank's prompt time here, and gives the same tokens. Four channels shorten the small decode
gathers by up to 20%. Without these, the same server managed 166 tok/s on code, 128 on chat and 1,330 on prompts.

Four RTX PRO 6000 Blackwell Max-Q cards at TP=4 (sm_120, 300 W, one Threadripper PRO host), `nvidia/GLM-5.3-Flash-NVFP4`
at revision `09b04e5e`, `--context 245760`, tok/s. Before is `main` at `9200a3064` with the settings above, and after is
the windows, prices and expert race on the same commit. Every reply equalled its `"draft": false` run and every
concurrent one its run alone.

| Drafter | Case | Before | After |
| --- | --- | ---: | ---: |
| MTP head | Code, one stream, greedy | 235.8 | 291.8 |
| MTP head | Chat, one stream, greedy | 146.4 | 190.3 |
| DFlash2 | Code, one stream, greedy | 254.3 | 304.8 |
| DFlash2 | Chat, one stream, greedy | 167.0 | 202.2 |
| DFlash2 | Code, one stream, temperature 1 | 187.7 | 215.2 |
| DFlash2 | Chat, one stream, temperature 1 | 158.3 | 189.1 |
| DFlash2 | Code, 8 streams, eight different prompts | 473.7 | 542.8 |
| DFlash2 | Code, 16 streams | 649.2 | 685.4 |
| DFlash2 | Prose, 8 streams, eight different prompts | 392.7 | 429.7 |
| DFlash2 | Prose, 16 streams | 526.0 | 583.6 |

One-stream runs are `tools/bench_openai.py` at 512 tokens and stream runs `tools/bench_concurrent.py` at 256. DFlash2's
weights are CC BY-NC-ND 4.0. After uses `TF_GLM_DRAFT_CONFIDENCE=0.4` and before keeps the default 0.7. Cold prompts
are unchanged at about 4,430 tok/s from 8K to 131K tokens, and a repeated code reply reaches about 850 tok/s with
windows of up to 32 rows. Other GPUs and rank counts are not covered here.

## More ranks

With `TF_GLM_PREFILL_REDUCE=ring`, the startup peer warm-up touches only each rank's next and previous neighbours,
using the existing rank order. The default split and gather modes still warm every peer. This removes the eager
opposite-peer warm-up; it does not add a switchless RoCE relay or qualify a four-node ring (#1971).

`TF_GLM_WORLD=3` places a third rank the same way. From 1.1.2 the uneven KDA input projections of TP=3 (22, 21 and
21 heads) run instead of being refused; TP=3 device qualification is pending
([#1067](https://github.com/ashhart/TensorFold/issues/1067)). How a prompt's rows reach the other ranks
(`TF_GLM_PREFILL_REDUCE`) and the two-rank RoCE gathers (`TF_ROCE`) are in
[placement](../multi-rank-receipts.md#the-settings-the-ranks-agreed-on).

## Prompt chunk size

A prompt runs in passes of 2,048 rows. Kept prompt states sit on that grid, and a stream decoding beside a prompt waits
for each pass to end. A pass that takes more rows reads each routed expert's weights fewer times, so a lone prompt
comes sooner. `TF_GLM_CHUNK_SOLO=4096` lets a pass take two of those chunks when no other stream is decoding as it
starts. With one decoding, or with `prompt_logprobs` or image rows, the pass takes 2,048. Beside `--drafter`, DFlash2
reads a solo pass's taps whole and keeps the prompt's last 2,048 rows as before; its tap buffer holds 4,096 rows, 80 MiB
more on each rank.

The grid does not move, so warm turns resume where they do unset, and streams decoding beside a prompt see the same
passes. Replies are the same bytes either way. Two costs remain:
- A request that arrives during a solo pass waits for that pass to end before its own prompt runs. After that, the
  passes are short again.
- The prompt buffers are sized for 4,096 rows for the whole run: 1.92 GiB more at TP=2 and 1.56 GiB at TP=3 (and
  DFlash2's 80 MiB), which comes out of the room for KV pages and the prompt cache. In the runs below, TP=2's default cache goes from 4.7 to
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

At TP=3 with `TF_GLM_ROCE_MULTI=1` (#1524) the gain is a little under half: on three GB10s at `--parallel 8` with
`TF_GLM_KV_SHARE=32` and `TENSORFOLD_PREFILL_SHARE=0.7`, not the table's settings, cold Python-source prompts of 8K to
128K tokens came 3.2 to 3.4% sooner with the wire and 7.3 to 8.4% sooner without it, two fresh servers each. The wire's
extra time is in the side stream's exchanges and grows when solo doubles a pass (#1994). `TF_GLM_OVERLAP_PIECES`
(1 to 8, the same on every rank; unset, 8 when `TF_GLM_CHUNK_SOLO=4096` and `TF_GLM_ROCE_MULTI=1` are both set at three
ranks or more, where the wire opens, and 3 otherwise) cuts a prompt chunk's exchanges into that many row pieces.
In a further run
at those settings, solo on the wire came 1.7 to 3.6% sooner at 3 pieces and 6.1 to 7.8% sooner at 8, against the wire
without solo at 3, with the same tokens; in another, 8 pieces without solo stayed within 1% of 3 (#1994).

## Long runs on GB10

Long prompts and long runs of replies heat a GB10. The runbook's [heat bands](../../RUNBOOK.md#linux-and-cuda) (with a
1,035,295-token prompt at TP=3), [CPU pinning](../../RUNBOOK.md#gb10-cpu-pinning-for-long-cuda-runs) and
[GPU clock cap](../../RUNBOOK.md#gb10-gpu-clock-cap-for-long-cuda-runs) each carry a measured GLM-5.3-Flash run.
