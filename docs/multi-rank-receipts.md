# Receipts from a group of ranks

A client measures a serving group through the single endpoint on its serving rank. The measured latency and rate
include the exchanges needed to produce its replies. Ranks can share a machine; rank count is not machine count.
Each path places its ranks its own way:

| Path | Places ranks by | Documented by |
| --- | --- | --- |
| Flash Next on two Macs | `--speed-up FILE`, one file per rank | [Speed-up mode](speed-up-mode.md) |
| Native GLM-5.3-Flash on CUDA | `TF_GLM_RANK` of `TF_GLM_WORLD`, with `NCCL_COMM_ID` on rank 0's address | [Native placement and limits](../zig/src/families/glm/cuda_native.zig); the engine refuses a world past 16 |
| Nemotron 3.5 Super on CUDA | `TF_NEMOTRON_RANK` of `TF_NEMOTRON_WORLD`, with `NCCL_COMM_ID` | [Its recipe](recipes/nemotron-3.5-super-vl.md) starts two; a world must divide the expert shares |
| Legacy Python 0.6.6 engine | `--tp`, `--rank`, `--master` | [The Python GLM recipe](recipes/glm-5.3-flash.md) |
| Native ROCm build | `--tp`, `--rank`, `--master` | [Native ROCm placement](../zig/src/native/hip.zig) |
| The `fabric/tp` transport's harness | `TF_TP_WORLD`, `TF_TP_RANK`, `TF_TP_MASTER` | [The transport's README](../zig/src/fabric/tp/README.md) |

The native CUDA build serves neither `--tp` nor `--rank`: a family's own variables place its ranks. `tf-tp-test` is
the transport's harness rather than a server. `tensorfold benchmark` accepts no group placement flags and records 1
for a managed run. Its child inherits the parent's rank environment, so start managed runs from a single-rank
configuration. Measure a group as an attached server: start the group, then point the client at the rank that serves.

## What a receipt can carry

A reply's `tensorfold` object carries the engine's name, its decode seconds and rate, `prefill_seconds`,
`time_to_first_token`, `sampling`, `drafts` and `token_sha` / `token_sha256`. It carries no rank count, no transport
and no per-rank timing, so a reply cannot show that a group served it. A receipt's `settings` adds `rank_count`, and
nothing else about ranks.

The native manifest reads `TF_NEMOTRON_WORLD` first, then `TF_BENCHMARK_RANK_COUNT` only when the former is unset.
It emits `rank_count` only when that value parses as 1 or 2. It does not derive the count from `TF_GLM_WORLD`, a
speed-up settings file or ROCm's `--tp`, and does not cross-check that hint against the running engine's world.
Never label a larger group as rank 1 or 2 to make its receipt fit the schema. Its hardware probe describes the
serving process's visible hardware, not an inventory gathered from every rank.

`settings.rank_count` accepts 1, 2 or null. A manifest declaring another count fails the client's checks as a whole,
leaving the attached run's hardware and revision unknown. A valid manifest with no count or a null count keeps its
other accepted facts while the rank count stays unknown. Both cases are unranked; the client does not infer a count
from reply speed or the number of GPUs. Runs past two ranks belong in the local prose report below. Community-v2
publication is currently disabled for every rank count, as [community benchmarks](community-benchmarks.md) explains.

## The settings the ranks agreed on

Boot agreement is path-specific. Native GLM compares a settings/model/build fingerprint and reports the differing
field and rank. Nemotron compares its model, cut, precision and build, reporting a category and rank. ROCm checks a
join identity and coordinates startup values; the `fabric/tp` harness compares a settings digest. These checks do
not establish that every path compares every serving setting. Record each rank's effective settings and the boot
checks that actually ran, together with the start order and any timeout or refusal. A failed launch is not a slow
successful request. A prompt split also fixes how rows reach the other ranks:
on GLM-5.3-Flash, `TF_GLM_PREFILL_REDUCE` defaults to `split`, which sends every rank's rows to every other, and a
world whose ranks reach only their neighbours needs `ring` or `gather` instead, as the collective path supports.

Transport choice changes exchange cost; verify token equality at the same world size rather than assuming it. On
native GLM on CUDA with two ranks, `TF_ROCE` takes the small all-gathers over RDMA writes and `TF_ROCE=0` on either rank keeps NCCL;
`TF_ROCE_HCA` names the devices and
`TF_ROCE_MAX_KB` bounds the largest gather sent that way. Nemotron's serving path opens the NCCL group without
enabling its optional RoCE gather. On the `fabric/tp` harness, `TF_COMM_BACKEND` picks `nccl`, `mailbox` or `roce`, and
`TF_TP_ROCE_FALLBACK` decides what a failed setup or probe does: `nccl` permits fallback to NCCL, while `error`
refuses. A harness run that reports the RoCE path sets `error`, so a silent fallback cannot
pass as the transport it names. The boot line is the evidence for which transport ran, and the harness prints it as
`tp: rank N of M up` beside the transport kind and the library version it opened: keep it with the output.

## Prompt and decode numbers

Both describe the group as served; the work distributed to other ranks differs by path. A cold prompt's
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

Prose, and the receipt stays local with its run: the rank count, each rank's effective settings and completed boot checks, the start order,
each rank's hardware class and the link's, the client that measured it, which equalities ran, and what it could not
attest. It does not enter a comparison beside rank-1 and rank-2 receipts, whatever it measured.

A contributor reports four GB10 nodes at TP=4 over RoCE, serving a 2.9-bpw EXL3 pack of a DeepSeek-V4.1-Flash
model. Their completion-route client, sampled at 0.6 with 512-token replies, reportedly read 37 to 41 tokens a
second from the stream. It is one rank count with no single-rank column beside it, so it is not a speed-up; that
client divides every completion token by the wall time from the first token to the last byte, where `tools/bench_openai.py` divides one
fewer by the delivery span; no `--alone` and no `"draft": false` equality was run; and no public checkpoint ID or
matched revision was supplied for that local pack. This unverified example illustrates a four-rank report; it is
not a measurement of this tree.
