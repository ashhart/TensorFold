# CUDA load-time prefetch

`TF_LOAD_PREFETCH` selects how many reader threads read a layer's checkpoint intervals into the file cache while its loader packs the weights. The value is a count from 0 to 16. Set 0 to disable read-ahead. Invalid values refuse startup by name. @jschmied introduced this option for Gemma 4 and Qwen 3.5 in #1928, where the default remains 16. The new Nemotron/Bolt, MiniMax M3, Kolibri-1, Spark 2.5 and DeepSeek-V4.1 ports are opt-in: unset means 0 until their checkpoint and topology pass the device load and memory cells. Set TF_LOAD_PREFETCH=16 explicitly to qualify them.

| CUDA loader | Layer interval |
| --- | --- |
| Nemotron and Bolt | The configured backbone scope, including the vision model's language_model prefix |
| GLM | The rank's ordered model.language_model layer list |
| MiniMax M3 | language_model.model layers |
| Kolibri-1 and Spark 2.5 | model layers |
| DeepSeek-V4.1 | Planned backbone and MTP layers in the EXL3 pack |

GLM retains `TF_GLM_LOAD_PREFETCH` when the universal option is unset. An explicit `TF_LOAD_PREFETCH` takes precedence, including 0. The default with both options unset is 16. For a no-reader comparison against an older GLM build, set its existing `TF_GLM_LOAD_PREFETCH=0`; that build's unset option starts readers.

The new adapters read file intervals only. Upload order, stored formats, conversions and existing page-release positions stay with each family. Each reader owns its temporary buffer and descriptor, and the loader joins readers before a source closes or its existing release runs. Disabled adapters allocate no reader state and read no intervals. The checked adapters refuse a failed open or short read with the source path. A refused load joins its readers and frees their state.

DeepSeek's production loader keeps its prepared-image cache and per-layer upload completion. A cache hit skips original pack read-ahead; a miss reads the planned layer prefix during image construction. Its low-level whole-model loader also keeps the existing builder/upload order. No CUDA kernel or launch changes.

`zig build test-load-prefetch-host` checks actual family reader adapters over small owned files, including scoped names, layer boundaries, chunk edges, disabled paths, failed reads and allocation failures. The GLM loop also checks a noncontiguous rank layer list and fail-stop ordering. These host checks do not qualify checkpoint tokens, memory use or load speed on a GPU.

Before release, qualify every checkpoint and TP topology with exact replies and matched load-time measurements at 0 and 16. Keep all raw repeats and the same file-cache starting conditions. Record load seconds, peak RSS and peak page cache per rank. MiniMax M3 and DeepSeek TP readers cover whole layer tensors rather than rank slices; DeepSeek cache misses can also warm a range after the builder drops it. Keep these ports opt-in until those peaks are qualified against main. TP qualification also requires deferring shared-file cache drops until every rank has finished its pending reads. Keep drafted/plain, resumed/fresh and shared/solo controls. The device campaign and actual before/after receipts determine qualification; no new-family speed is claimed here.
