# Flash Next MTP drafter weights: main against the branch

`TF_FLASHNEXT_MTP_WEIGHTS=checkpoint|mxfp8`. Unset or blank is `checkpoint`, main's load. `mxfp8` re-codes
the drafter's BF16 linears to MXFP8 as they load, leaving its two fusion projections (`fc_embedding`,
`fc_hidden`) and the layer's shared expert (gate, up, down) as the checkpoint stores them. The server states
the split itself on every `mxfp8` boot:

    info: flashnext: TF_FLASHNEXT_MTP_WEIGHTS=mxfp8 re-coded 135 MiB of the drafter's BF16 linears and left 34 MiB as stored

## Source and environment

- Main `4fda6a020605fb7accb70f36354d3fd22adb9b15`, branch `b43b10f46df555b1eb9e2b7ba19832c0d4da78f0` (this pull request's head with that main merged). Every number below is a run of these two commits, one server a side, alternated.
- One NVIDIA GB10 (sm_121), one DGX Spark; checkpoint `/root/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300`, every arm; bf16 KV; a fresh server per arm.
- Decode and the exactness boots at a 32,768-token window and 8 slots (8 streams need it); the cold prompts at 131,072 and 4 slots with the prompt cache off.
- States, in the order run (mirrored, two runs each): M1 U1 X1 M2 X2 U2; the cold pass the same with the C suffix; a rerun pass on fresh boots only if a cell flagged slower.
- Main's own `tools/bench_openai.py` and `tools/bench_concurrent.py --alone --serial` for every arm.

## Decode speed and tokens a round

`cells.csv` has every cell of every arm; `rule5.md` has each cell's runs and every comparison's verdict. `tokens_per_round` is the tools' tokens a streamed chunk, `accepted_per_round` that less one, and `server_acceptance` the server's `tensorfold:mtp_accepted_total` over `tensorfold:mtp_drafted_total` across the invocation. `mxfp8` is the mean of its two runs against the mean of main's two:

| comparison | faster | level | slower | cells |
|---|---|---|---|---|
| U vs M | 0 | 32 | 0 | 32 |
| X vs M | 22 | 10 | 0 | 32 |

The noise rule: no cell was slower in the first pair, so no rerun was taken.

Unset against main: U vs M 0 faster, 32 level, 0 slower.

## Cold prompts

| case | M | U | X | X vs M |
|---|---|---|---|---|
| cold2k-off | 1.059, 1.084 | 1.074, 1.067 | 1.066, 1.073 | +0.2% level |
| cold8k-off | 4.021, 4.004 | 4.004, 3.999 | 3.996, 4.025 | +0.1% level |
| cold32k-off | 16.049, 16.032 | 16.067, 16.089 | 16.038, 15.997 | +0.1% level |
| cold64k-off | 32.947, 33.125 | 33.099, 33.167 | 33.092, 33.087 | -0.2% level |
| first_ttft_s | 1.217, 1.231 | 1.228, 1.224 | 1.227, 1.228 | -0.3% level |

## Off, on and a refused value

- M and U log no re-coding line, and X logs it on every boot.
- `R1`: `tensorfold: TF_FLASHNEXT_MTP_WEIGHTS="MXFP8": expected checkpoint or mxfp8 (unset or blank keeps checkpoint, the rows as stored; mxfp8 re-codes the drafter's BF16 linears on CUDA, less its two fusion projections and shared expert)`
- `R2`: boot rc 0, no re-coding line
- `R3`: boot rc 0, no re-coding line

## Exactness

`exactness.md`: drafted against `"draft": false`, resumed against fresh and concurrent against solo, on every state with the prompt cache on and off, and every reply's token ids against main's.

## Resident memory

| state | boots | model GiB after load |
|---|---|---|
| main | 6 | 72.71, 72.71, 72.73, 72.73, 72.71, 72.71 |
| unset | 6 | 72.71, 72.71, 72.73, 72.73, 72.71, 72.71 |
| mxfp8 | 6 | 72.64, 72.64, 72.67, 72.67, 72.64, 72.64 |

## Not established

Metal and HIP refusing `mxfp8`, and a CUDA drafter linear stored as anything but BF16 refusing it, are covered by host tests on this head, not run on that hardware or on such a checkpoint: the checkpoint this window ran keeps BF16 `mtp.*` projection rows.
