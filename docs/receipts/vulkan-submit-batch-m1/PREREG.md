# One submission per batch of dispatches: pre-registration (written 2026-11-03 before any GPU run of this change)

## Question

Every dispatch of the Vulkan engine is its own submission with a fence wait. The profile of the tile build (unicode-24) shows 31,104 dispatches
in a 64-step run, about 19,000 of them small kernels at about 100 us each, and the tiny ones (copy, gather, dequant) at 89 to 111 us with a
minimum of 66 to 79 us: mostly submit and wait. Does recording many dispatches into one command buffer and submitting once per batch shorten a
run, with byte-identical logits? No kernel and no sum changes.

## Arms (two ELFs of `tf-qwen35-model`, ReleaseSafe, aarch64)

- old: the tile build, commit `6a158edc`, ELF `d4949bef...`.
- new: commit `0d73b0fb`, ELF `d8ee787b...`: dispatches are recorded into the open command buffer; it is submitted with 128 waiting dispatches, or when
  an upload, download or fill needs the queue. Each pipeline has a ring of 256 descriptor sets.

## Gates and metric

As before: exit 0, pinned ids, one logits digest per case (unicode-24 `e5447d9a...`, document-256 `147a2d2e...`, document-1024 `7f9d8e65...`), run 2 duration
from the probe's milestones, three pairs in the order old,new,new,old,old,new. Also: the stage 1 to 5 probes pass on a software device (done), and a hang
or a validation error in any process fails the unit.

## Decision rule

Median gain of at least 3 percent and all pairs positive and all gates hold. Otherwise not proposed.

## Forecast (a prediction)

unicode-24 gains 15 to 35 percent: the floor is about 40 percent of the run, and I expect to keep half to four fifths of it (each dispatch still costs a
record, a barrier and the GPU's launch gap). document-256 and document-1024 gain less (a smaller share of their time is dispatch cost): 5 to 15 percent.
I give 15 percent to a gain under 3 percent on unicode-24, and 10 percent to a failure (a hang, a wrong digest or a crash) that needs a fix before any number counts.

## Not claimed

Other chips and models. The software-device pass is correctness evidence only.
