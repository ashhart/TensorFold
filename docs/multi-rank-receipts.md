# Receipts from a group of ranks

A server is one rank on one machine or one rank of a group, and only one rank of a group answers requests. A client
therefore measures a group through a single endpoint, and every rate it reports carries what the ranks exchanged to
produce it. Each path that serves more than one rank places its ranks its own way:

| Path | Places ranks by | Documented by |
| --- | --- | --- |
| Flash Next on two Macs | `--speed-up FILE`, one file per rank | [Speed-up mode](speed-up-mode.md) |
| GLM-5.3-Flash on CUDA | `TF_GLM_RANK` of `TF_GLM_WORLD`, with `NCCL_COMM_ID` on rank 0's address | [Its recipe](recipes/glm-5.3-flash.md) starts two; the engine refuses a world past 16 |
| Nemotron 3.5 Super on CUDA | `TF_NEMOTRON_RANK` of `TF_NEMOTRON_WORLD`, with `NCCL_COMM_ID` | [Its recipe](recipes/nemotron-3.5-super-vl.md) starts two; a world must divide the expert shares |
| Python 0.6.6 engine, and the ROCm build | `--tp`, `--rank`, `--master` | [The GLM recipe](recipes/glm-5.3-flash.md); [the two-rank RoCE gather](roce-gather.md) |
| The `fabric/tp` transport's harness | `TF_TP_WORLD`, `TF_TP_RANK`, `TF_TP_MASTER` | [The transport's README](../zig/src/fabric/tp/README.md) |

The native CUDA build serves neither `--tp` nor `--rank`: on it, a family's own variables place the ranks, and
`tf-tp-test` is the transport's harness rather than a server. `tensorfold benchmark` starts a single-rank server of
its own and refuses those flags, so a group is measured as an attached server: start the group, then point the client
at the rank that serves.

## What a receipt can carry

A reply's `tensorfold` object carries the engine's name, its decode seconds and rate, `prefill_seconds`,
`time_to_first_token`, `sampling`, `drafts` and `token_sha` / `token_sha256`. It carries no rank count, no transport
and no per-rank timing, so a reply cannot show that a group served it. A receipt's `settings` adds `rank_count`, and
nothing else about ranks.

The attested manifest is the only evidence a ranked comparison accepts, and it stops at two ranks. It names a rank
count only for a world of 1 or 2: given a larger one, `TF_NEMOTRON_WORLD` or `TF_BENCHMARK_RANK_COUNT` write no field
at all rather than a count the board does not rank. `settings.rank_count` is checked as 1, 2 or null, and a manifest
that declares anything else fails the client's checks as a whole, so the attached run keeps unknown hardware and an
unknown revision instead of an unranked count beside known facts. A run the command started itself records 1; an
attached run takes the manifest's count, or none when the manifest did not pass. A run at three or more ranks is
therefore unranked by construction, however well it measured.

## The settings the ranks agreed on

Each path exchanges the settings its ranks must share at boot and stops every rank when two of them differ, naming the
field that differs and the rank it differs on. Record the settings every rank agreed on, and not only the serving
rank's, together with the start order: a serving rank that waits for the others inside a timeout explains a failed
launch that a receipt would otherwise record as a slow one. A prompt split also fixes how rows reach the other ranks:
on GLM-5.3-Flash, `TF_GLM_PREFILL_REDUCE` defaults to `split`, which sends every rank's rows to every other, and a
world whose ranks reach only their neighbours takes `ring` instead.

Where a path offers a transport, the choice moves the exchange cost and not the tokens. On the CUDA families that
serve two ranks, `TF_ROCE` takes the small all-gathers over RDMA writes and `TF_ROCE=0` on either rank keeps NCCL;
`TF_ROCE_HCA` names the devices, and GLM-5.3-Flash takes the largest gather it will send that way from
`TF_ROCE_MAX_KB`. On the `fabric/tp` harness, `TF_COMM_BACKEND` picks `nccl`, `mailbox` or `roce`, and
`TF_TP_ROCE_FALLBACK` decides what a failed setup or probe does: `nccl` keeps the run with the same replies and a
different exchange cost, `error` refuses. A run that reports the RoCE path sets `error`, so a silent fallback cannot
pass as the transport it names. The boot line is the evidence for which transport ran, and the harness prints it as
`tp: rank N of M up` beside the transport kind and the library version it opened: keep it with the output.

## Prompt and decode numbers

Both are aggregates, because a prompt's chunks and every decode round are split between the ranks. A cold prompt's
`ttft_s` from `tools/bench_cache.py --require-cold` on a long prompt is the group's first token, not one device's
prefill, and the ranks' own prefill times do not add up to it. Take it with the prompt cache off
(`--prompt-cache-gib 0`, where the family supports it) and a fresh process for an independent cell, as
[cache measurements](cache-benchmarks.md) describe.

`tools/bench_openai.py` warms each cell once and then measures its repetitions. Its rate is the stream's delivery rate
at the client, one fewer than the completion count over the span between the first and last delivery, so it includes
every exchange a round needs; a delivery can carry several tokens, so it is not an engine clock either. Quote that
rate for a group, and do not present it as any single rank's rate.

`tools/bench_concurrent.py --alone --serial` still applies: `--alone` compares each concurrent reply's token hash with
its solo run, and `--serial` compares the solo runs with `"draft": false`. Each rank count has its own serial
reference, so a report says which world size each equality ran at; an equality at one rank count is not one at
another. A family that admits one request at a time turns a level above one into a measure of the queue, so quote the
level and what `--parallel` admitted beside the throughput.

## What stays out

No machine names, addresses or local model paths, as for any receipt. A local model directory cannot be published:
name the checkpoint's public ID and revision, or keep the run local. Add the hardware class of each rank and the class
of the link between them, because the exchange cost is a property of that fabric; two runs at the same rank count on
different links are not comparable at the same numbers. When a change reaches the transport, the before column is the
base commit at the same rank count.

## Reporting a run past two ranks

Prose, and the receipt stays local with its run: the rank count, the settings every rank agreed on, the start order,
each rank's hardware class and the link's, the client that measured it, which equalities ran, and what it could not
attest. It does not enter a comparison beside rank-1 and rank-2 receipts, whatever it measured.

Ours is four GB10 nodes at TP=4 over RoCE, serving a 2.9-bpw EXL3 pack of a DeepSeek-V4.1-Flash model. A
completion-route client we wrote, sampled at 0.6 with 512-token replies, read 37 to 41 tokens a second from the
stream. It is one rank count with no single-rank column beside it, so it is not a speed-up; that client divides every
completion token by the wall time from the first token to the last byte, where `tools/bench_openai.py` divides one
fewer by the delivery span; no `--alone` and no `"draft": false` equality was run; and the pack is a local model
directory, so it has no public checkpoint ID. It is the shape of a four-rank report, not a measurement of this tree.
