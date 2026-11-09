# Persisted serving state identities

A resumed state belongs to the loaded model and arithmetic that produced it. The native disk stores bind the
running serving binary with a streamed SHA256 fingerprint. This covers compiled host code, embedded shaders,
compiler options and snapshot layouts. Rebuilding or changing the binary can cold-start the optional disk cache.
Old files remain in their original namespace and follow the existing cache budget and eviction rules.

| Persistence owner | Source and arithmetic | Checkpoint | KV and draft state | Weight learning |
| --- | --- | --- | --- | --- |
| Nemotron, Metal | Binary, kernel texts, precision mode, prompt step, loaded coop K-slice split and its scratch rows, probe, OS build and GPU | Config, index, shard stamps and MTP header | Canonical snapshot, head flag and probe | Shared host clears RAM and detaches disk stores when weights move |
| Dense Qwen hybrid, Metal | Binary, shared/family shader texts, tensor mode, loaded ANE columns and prompt policy, probe, OS build and GPU | Config, index and shard stamps for target and external drafter | Canonical recurrence/KV/tap image and draft end | Unsupported learners refuse; the store remains scoped to its loaded model |
| GLM, Metal | Binary, compiled source hash, resolved rank/route/prompt policy, probe, OS build and GPU | Config/index identity plus current shard stamps | Canonical KDA/MLA image and checked snapshot header | Unsupported learner on this adapter |
| Flash Next, Metal | Binary, actual loaded shader texts/bindings, rotary/precision and resolved execution policy, probe, OS build and GPU | Pack metadata identity, current source and loaded-pack stamps, loaded reference bytes | Checked snapshot layout; canonical session image | Unsupported learner on this adapter |
| Nemotron, CUDA | Binary, embedded images, captured manifest and cubin bytes, resolved checkpoint math, expert ranks, NVFP4 decode form, lane half-tile threshold, expert grids and captured attention variant, probe, GPU and driver | Config, index and shard stamps | Head flag and canonical state image | Shared host clears RAM and detaches disk stores when weights move |
| Qwen dense/MoE hybrids, CUDA | Same shared CUDA identity | Same shared checkpoint identity | Explicit BF16/FP8 format, tap flag, prompt-int8 mode and actual rotary table; image states stay off disk | Same shared learning barrier |
| Flash Next, CUDA | Same shared CUDA identity, plus resolved prompt chunk heights, prompt kernel forms and tile thresholds, expert grids and captured-set use | Same shared checkpoint identity | Head flag and canonical recurrence/KV/PLE image; tensor-rank disk persistence is refused | Same shared learning barrier |
| DeepSeek sessions, CUDA reference path | Config/options/mode, checkpoint digest, rank, selected layers and piece layout | Checkpoint digest required | Versioned lossless session pieces and exact token prefix | No learner; the disk tier starts fresh for each process |

Other native family adapters have no attached cross-process serving-state tier. Their snapshots remain owned by one
open model. This includes the other Metal families (Qwen4, Fish, Qwen MoE, LFM2, EXLR8 GLM, Gemma, Step, Spark,
MiniMax and Llama), other CUDA families (Omni, GLM, MiniMax, Kolibri, Gemma, Spark, Dory, LFM2, Fish and Qwen4),
and the HIP and XPU adapters. A family name or a RAM snapshot alone does not imply persistent disk reuse.

Tokenizer and chat-template bytes are covered through the canonical numeric token prefix that the store actually
matches. Changed rendering or tokenization that changes an ID misses. An identical ID prefix has the same target
state under identical weights and arithmetic, so it can remain reusable. Planned stores also compare their chunk
boundaries, while the shared cache compares decoded spans and image identities. These stores do not persist
rendered answer text. No separate hot-swappable LoRA state is admitted by these persistence adapters; checkpoint
changes and supported external drafters use the identities above.

Checkpoint keys include every validated indexed shard path, including subfolders and any suffix accepted by the loader, plus
auxiliary top-level safetensors files. Their names, sizes, inodes and modification/change stamps are bound. A weight
write with unchanged config, index, tensor header and size moves their namespace even when a fixed probe does not
visit the changed tensor. Flash Next also hashes the loaded variant text and input/output/meta bindings in sorted
function-name order. Captured CUDA sets hash each cubin named by the parsed manifest, through the same loader
paths and size limit. A missing or changed file fails identity construction instead of opening an old namespace;
a shard the index names but the folder lacks is refused as `IndexedShardMissing`, as the loaders refuse it.

Resolved policies are bound even when a fixed startup probe does not exercise their difference. The Metal Qwen
key includes loaded ANE columns and the prompt chunk height, not the frame's buffer rows. GLM includes effective
rank slices, routing, its prompt chunk height and sparse kernel, and whether the MTP head's prompt eh_proj runs on
the tile kernel as the load built it (`GLM_EHPROJ_NAX=0` or a failed tile build: scalar). A different chunk height
starts these two stores cold: GLM's prompt windows and chunks sum in different orders (#743), and Qwen's may too.
Flash Next includes its resolved execution branches, including the DeltaNet window kernels, but not the serial
encoder choice, which orders the same launches. It also binds stamps of the files actually indexed (including an
external weight pack) and the bytes of loaded metadata such as a supplied reference. Descriptor and mapping order
do not change that key. CUDA Qwen keys include the rotary table the target consumes, for dense and routed
checkpoints. These additions change the optional disk namespace; snapshot payloads and layouts are unchanged.

Flash Next on CUDA keys what its prompt passes resolved at load: the solo and busy chunk heights (a different
`TF_FLASHNEXT_CHUNK` or `_CHUNK_SOLO`, or `--parallel 1` taking the larger pass, starts cold), the DeltaNet, read-out,
hyper-connection and in-pass mark forms, the NVFP4 prompt-tile threshold, K32 tile, BF16 prompt tiles and the SM count
they are weighed against, the lane's half-tile threshold (`TF_QMMF_HALF_BELOW`), the experts' decode warps and resident
grids, and whether a captured set runs with which entries on our glue. `TF_FLASHNEXT_KV_PAGES` and `_KV_PAGES_ALL` are
keyed through the profile rows. Its draft, window, round, graph, lane-parts (`TF_FLASHNEXT_LANE_PARTS`), load-thread, I/O and dense-split settings and
`TF_FLASHNEXT_KV_PAGE_TEST` are not keyed: kept states come only from prompt passes, which those settings either never
reach, reach with the same launches or the same kernels addressing pages, or refuse for disk at `--tp 2`.
Nemotron on CUDA keys `--precision checkpoint`'s math, the expert ranks of `TF_NEMOTRON_WORLD`, a ModelOpt
checkpoint's NVFP4 decode warps and lane half-tile threshold, the experts' resident grids and, with a captured set, the
attention variant `--context` picks by its chunk count. Its pages, poison fill, carveout, segments, graph log and trace,
and the context on our own glue, are not keyed: they place memory, fill pages before use, log, run only passes that
keep no state (segments), or set a stride our merge ignores. Nemotron on Metal also keys `TF_COOP_SPLIT` as loaded and,
with the split on, the scratch rows a prompt chunk's partials must fit; its page plan and stream count otherwise only
size buffers.

The executable and kernel fingerprints use a 32 KiB read buffer and check file stamps across the read. Checkpoints,
executables and kernel assets must remain stable while a model opens and serves. The existing learner owns runtime
weight changes and clears the store before another reply; it does not attach an old disk tier after a lesson.

Host checks:

```sh
zig build test-persistence-identity-host test-cuda-persistence-source-host compile-persistence-host -j1
zig build test-checkpoint-identity test-cuda-disk-open-host test-sessions test-frame-cache-identity-host -j1
zig build test-disk-identity-host -j1
```

The first command's Metal consumer compile and checkpoint fixtures require a macOS host but execute no GPU call.
These checks cover stale namespaces, real same-header shard rewrites, binary and captured-cubin mutations, loaded
variant bindings, allocation failures, canonical token matching and whole-file fingerprint coverage. Actual
checkpoint/device qualification still requires exact drafted/plain, resumed/fresh and concurrent/solo token arrays.
