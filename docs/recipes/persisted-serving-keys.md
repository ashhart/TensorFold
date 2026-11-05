# Persisted serving state identities

A resumed state belongs to the loaded model and arithmetic that produced it. The native disk stores bind the
running serving binary with a streamed SHA256 fingerprint. This covers compiled host code, embedded shaders,
compiler options and snapshot layouts. Rebuilding or changing the binary can cold-start the optional disk cache.
Old files remain in their original namespace and follow the existing cache budget and eviction rules.

| Persistence owner | Source and arithmetic | Checkpoint | KV and draft state | Weight learning |
| --- | --- | --- | --- | --- |
| Nemotron, Metal | Binary, kernel texts, precision mode, probe, OS build and GPU | Config, index, shard stamps and MTP header | Canonical snapshot, head flag and probe | Shared host clears RAM and detaches disk stores when weights move |
| Dense Qwen hybrid, Metal | Binary, shared/family shader texts, tensor mode, loaded ANE columns and prompt policy, probe, OS build and GPU | Config, index and shard stamps for target and external drafter | Canonical recurrence/KV/tap image and draft end | Unsupported learners refuse; the store remains scoped to its loaded model |
| GLM, Metal | Binary, compiled source hash, resolved rank/route/prompt policy, probe, OS build and GPU | Config/index identity plus current shard stamps | Canonical KDA/MLA image and checked snapshot header | Unsupported learner on this adapter |
| Flash Next, Metal | Binary, actual loaded shader texts/bindings, rotary/precision and resolved execution policy, probe, OS build and GPU | Pack metadata identity, current source and loaded-pack stamps, loaded reference bytes | Checked snapshot layout; canonical session image | Unsupported learner on this adapter |
| Nemotron, CUDA | Binary, embedded images, captured manifest and cubin bytes, probe, GPU and driver | Config, index and shard stamps | Head flag and canonical state image | Shared host clears RAM and detaches disk stores when weights move |
| Qwen dense/MoE hybrids, CUDA | Same shared CUDA identity | Same shared checkpoint identity | Explicit BF16/FP8 format, tap flag, prompt-int8 mode and actual rotary table; image states stay off disk | Same shared learning barrier |
| Flash Next, CUDA | Same shared CUDA identity | Same shared checkpoint identity | Head flag and canonical recurrence/KV/PLE image; tensor-rank disk persistence is refused | Same shared learning barrier |
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

Checkpoint keys include every validated indexed shard path, including subfolders and nonstandard suffixes, plus
auxiliary top-level safetensors files. Their names, sizes, inodes and modification/change stamps are bound. A weight
write with unchanged config, index, tensor header and size moves their namespace even when a fixed probe does not
visit the changed tensor. Flash Next also hashes the loaded variant text and input/output/meta bindings in sorted
function-name order. Captured CUDA sets hash each cubin named by the parsed manifest, through the same loader
paths and size limit. A missing or changed file fails identity construction instead of opening an old namespace.

Resolved policies are bound even when a fixed startup probe does not exercise their difference. The Metal Qwen
key includes loaded ANE columns and prompt sizing; GLM includes effective rank slices, routing and prompt choices.
Flash Next includes its resolved execution branches, stamps of the files actually indexed (including an external
weight pack), and the bytes of loaded metadata such as a supplied reference. Descriptor and mapping order do not
change that key. CUDA Qwen keys include the rotary table the target consumes, for dense and routed checkpoints.
These additions change the optional disk namespace; snapshot payloads and layouts are unchanged.

The executable and kernel fingerprints use a 32 KiB read buffer and check file stamps across the read. Checkpoints,
executables and kernel assets must remain stable while a model opens and serves. The existing learner owns runtime
weight changes and clears the store before another reply; it does not attach an old disk tier after a lesson.

Host checks:

```sh
zig build test-persistence-identity-host test-cuda-persistence-source-host compile-persistence-host -j1
zig build test-checkpoint-identity test-cuda-disk-open-host test-sessions test-frame-cache-identity-host -j1
```

The first command's Metal consumer compile and checkpoint fixtures require a macOS host but execute no GPU call.
These checks cover stale namespaces, real same-header shard rewrites, binary and captured-cubin mutations, loaded
variant bindings, allocation failures, canonical token matching and whole-file fingerprint coverage. Actual
checkpoint/device qualification still requires exact drafted/plain, resumed/fresh and concurrent/solo token arrays.
