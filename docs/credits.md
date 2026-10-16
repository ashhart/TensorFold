# Credits

TensorFold's native engine (1.0 and later) grew out of the Python engine (0.2.0 to 0.6.6), and much of the Python
engine was written by people who sent pull requests. This page names whose work each native part carries or follows,
with links to their pull requests and to their files at the `v0.6.6` tag. The Python engine is frozen at that tag and
still installs from it.

Upstream projects whose code TensorFold adapts are named in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
Everyone who sent a pull request, a measurement or a bug report is listed under Contributors in the
[README](../README.md#contributors).

## The server and its API

The native server is one server for Metal and CUDA. Its request handling follows the Python server's, so these
contributions are in it.

- @kky42 wrote the Anthropic Messages API in [#223](https://github.com/ashhart/TensorFold/pull/223)
  ([anthropic_translate.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/server/anthropic_translate.py),
  [anthropic.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/server/anthropic.py)). The native
  `zig/src/server/anthropic.zig` and `anthropic_translate.zig` follow that translation. The #223 commits carry the
  account's former login, evilpsycho42.
- @philip-pentatonic wrote structured output in [#80](https://github.com/ashhart/TensorFold/pull/80)
  ([grammar.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/engine/grammar.py)), now
  `zig/src/server/grammar.zig` and `zig/src/grammar/tf_grammar.cc`. The same author's
  [#83](https://github.com/ashhart/TensorFold/pull/83) found message starts in chat templates that drop a reply's
  think block ([prefill_plan.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/engine/prefill_plan.py));
  `zig/src/server/chunk_plan.zig` finds them the same way.
- @chris247474 typed tool-call parameters by the tool's schema and honoured a request's reasoning effort in
  [#28](https://github.com/ashhart/TensorFold/pull/28)
  ([tool_parameters.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/tool_parameters.py),
  [tool_draft.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/engine/tool_draft.py)).
  @outcastofmusic added Python-spelled values in [#135](https://github.com/ashhart/TensorFold/pull/135). Both are in
  `zig/src/server/tool_params.zig` and `tool_stream.zig`.
- @jeidbugs404 parsed GLM's `<arg_key>` and `<arg_value>` tool calls in [#35](https://github.com/ashhart/TensorFold/pull/35)
  (`zig/src/server/tool_parse.zig`), and @salmanarshad321 made malformed tool-call history render safely in
  [#233](https://github.com/ashhart/TensorFold/pull/233)
  ([messages.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/server/messages.py),
  `zig/src/server/messages.zig`).
- Reasoning controls in `zig/src/server/fields.zig` come from @EugeneClaw's nearest named effort
  ([#161](https://github.com/ashhart/TensorFold/pull/161)), @MiaAI-Lab's `max` and `xhigh` efforts
  ([#183](https://github.com/ashhart/TensorFold/pull/183)) and @MiaAI-Lab's `chat_template_kwargs.thinking` switch
  ([#236](https://github.com/ashhart/TensorFold/pull/236))
  ([request_options.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/server/request_options.py)).
- @gprot42 split a thought channel out of a reply wherever it opens, with thinking on or off, in
  [#171](https://github.com/ashhart/TensorFold/pull/171)
  ([text.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/server/text.py)), now
  `zig/src/server/reply_text.zig`. @gprot42's [#177](https://github.com/ashhart/TensorFold/pull/177) placed the history
  boundary one token short when a template adds no generation prompt
  ([prompt_blocks.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/server/prompt_blocks.py)), as
  `zig/src/server/prompt.zig` does.
- @kingjamez's patch in the [#9](https://github.com/ashhart/TensorFold/pull/9) thread closes a dangling `<think>` when
  thinking is off, in `zig/src/server/prompt.zig` as in the Python `text.py`.
- @JordiPosthumus read chunked request bodies in [#244](https://github.com/ashhart/TensorFold/pull/244)
  ([request_body.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/server/request_body.py)), now
  `zig/src/server/http_body.zig`. @SxMShaDoW made a refused request's body unable to reach the next request on its
  connection in [#182](https://github.com/ashhart/TensorFold/pull/182)
  ([http.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/server/http.py)), now
  `zig/src/server/routes.zig`. @jayleaton made the client-gone check work past descriptor 1023 in
  [#218](https://github.com/ashhart/TensorFold/pull/218)
  ([cancellation.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/server/cancellation.py)), now
  `zig/src/server/http_conn.zig`.
- @MiaAI-Lab added `/tokenize` and `/detokenize` in [#237](https://github.com/ashhart/TensorFold/pull/237)
  ([token_routes.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/server/token_routes.py),
  `zig/src/server/tokens.zig`) and image parts in Responses tool outputs in
  [#235](https://github.com/ashhart/TensorFold/pull/235)
  ([responses_translate.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/server/responses_translate.py)).
- @mikolaj92 wrote `/v1/decisions` in [#127](https://github.com/ashhart/TensorFold/pull/127)
  ([decisions.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/server/decisions.py)).
  `zig/src/server/decisions.zig` checks a request the way decisions.py does.
- Metrics and health: @MiaAI-Lab's live token totals on `/health` ([#79](https://github.com/ashhart/TensorFold/pull/79),
  [health.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/cuda/health.py)), the vLLM metric names
  and the disconnect and preemption counters from @jeidbugs404 and @di37
  ([#156](https://github.com/ashhart/TensorFold/pull/156), [#158](https://github.com/ashhart/TensorFold/pull/158),
  [metrics.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/server/metrics.py)),
  @sxuff's `request_decode_seconds` ([#269](https://github.com/ashhart/TensorFold/pull/269)) and @akol1's
  `process_footprint_bytes` ([#277](https://github.com/ashhart/TensorFold/pull/277)). They are in
  `zig/src/server/metrics.zig` and `status_routes.zig`.
- @di37's request log records image parts as `<redacted>` ([#64](https://github.com/ashhart/TensorFold/pull/64)),
  now `zig/src/server/request_log.zig`. @jschmied's `TENSORFOLD_SEED_SALT`
  ([#175](https://github.com/ashhart/TensorFold/pull/175),
  [exact_sampling.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/engine/exact_sampling.py)) moves
  every prompt-derived seed in the native server too.

## Prompt reuse

- @nood-co1's [#190](https://github.com/ashhart/TensorFold/pull/190) evicts first a kept state its conversation has
  moved past, and keeps a shared system block and a state a conversation forks back to
  ([streams.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/cuda/streams.py)). The native prompt
  cache (`zig/src/core/prompt_cache.zig`, `prompt_radix.zig`) evicts by the same rule. @nood-co1's
  [#65](https://github.com/ashhart/TensorFold/pull/65) kept a prompt's end state one token before its end, as the
  native cache does for several families.
- The message starts and history boundaries above (#83, #177) decide where the native cache keeps states.

## Apple Silicon

- @chadhurley25075-png brought GLM-5.3-Flash to Apple Silicon in [#9](https://github.com/ashhart/TensorFold/pull/9)
  and wrote its absorbed sparse-attention prefill in [#39](https://github.com/ashhart/TensorFold/pull/39). The native
  GLM-5.3-Flash engine's decode kernels are generated from the Metal kernels of #9
  ([kernels/glm/flash/v1](https://github.com/ashhart/TensorFold/tree/v0.6.6/src/tensorfold/kernels/glm/flash/v1)), it
  reads the mlx-lm checkpoint layout as the #9 loader did
  ([layouts.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/families/glm5_next/layouts.py)), and
  its prompt chunks attend the way the #39 prefill does, each query over its own keys in the latent cache
  ([mla.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/families/glm5_next/mla.py)). The native
  code is under `zig/src/families/glm` and `zig/kernels/metal/glm`.
- @quigles1977 gave Flash Next a prompt path on Macs in [#29](https://github.com/ashhart/TensorFold/pull/29)
  ([prefill_hc.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/kernels/qwen/flash_next/v1/prefill_hc.py),
  [prefill.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/kernels/qwen/flash_next/v1/prefill.py)).
  The native hyper-connection prompt kernels in `zig/kernels/metal/prefill/fn_prompt.metal` are the #29 kernels, and
  `fn_attn.metal` reads only each row's selected blocks and tail, as the #29 attention did.
- @gilby applied an n-gram table's `weight_scale` at lookup in [#148](https://github.com/ashhart/TensorFold/pull/148)
  ([embed.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/kernels/qwen/flash_next/v1/embed.py)),
  so oQ checkpoints of Flash Next load. The native `zig/src/families/flashnext/ple_scale.zig` applies the same rule.
- @jasontitus gave the lane matmul 3- and 2-bit weights in [#34](https://github.com/ashhart/TensorFold/pull/34)
  ([lane_qmm.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/kernels/qwen/dense/v1/lane_qmm.py)),
  widening each group to nibbles for the 4-bit tensor op. The native `zig/kernels/metal/core/lane_projection.metal`
  stages 2- and 3-bit codes the same way.
- @di37 made the engine read the GPU generation once from `applegpu_gN`, and a virtual machine's GPU as none, in
  [#209](https://github.com/ashhart/TensorFold/pull/209)
  ([device.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/kernels/device.py)), as
  `zig/src/metal/device.zig` does.

## CUDA

- @pmeenan made CUDA loads read checkpoints with O_DIRECT and pack MLX experts on the GPU in
  [#82](https://github.com/ashhart/TensorFold/pull/82)
  ([direct_read.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/cuda/direct_read.py),
  [experts_pack.cu](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/cuda/experts_pack.cu)). @pmeenan
  also wrote the native O_DIRECT reader, `zig/src/core/direct_io.zig`, and `zig/kernels/cuda/experts_pack.cu` is the
  same packing kernel.
- @jschmied brought block-scaled FP8 weights ([#126](https://github.com/ashhart/TensorFold/pull/126)) and the NVFP4
  prompt rows that sum a tile's K slices in one block ([#242](https://github.com/ashhart/TensorFold/pull/242),
  [qmmf.cu](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/cuda/nvfp4/qmmf.cu)) to the Python
  engine, and ported both to the native CUDA engine as well (`zig/src/cuda/fp8.zig`, `qmmf.zig`).
- The native CUDA memory budget takes `TENSORFOLD_MEMORY_RESERVE_GIB` from @MiaAI-Lab's
  [#133](https://github.com/ashhart/TensorFold/pull/133) and `TENSORFOLD_CUDA_MEMORY_LIMIT_GB` from @eleqtrizit's
  [#165](https://github.com/ashhart/TensorFold/pull/165)
  ([capacity.py](https://github.com/ashhart/TensorFold/blob/v0.6.6/src/tensorfold/cuda/capacity.py),
  `zig/src/native/cuda_memory.zig`).

## EXL3

@vcruz305 wrote the shared EXL3 module in [#42](https://github.com/ashhart/TensorFold/pull/42): the format reader, the
row-invariant linear, grouped experts and the 27B and Flash Next packs
([cuda/exl3](https://github.com/ashhart/TensorFold/tree/v0.6.6/src/tensorfold/cuda/exl3),
[exl3.md](https://github.com/ashhart/TensorFold/blob/v0.6.6/docs/recipes/exl3.md)). The native EXL3 decoder
(`zig/src/core/exl3_format.zig`, `exl3_projection.zig`) follows the format notes from #42, and the EXLR8 codebook
that serves GLM-5.3 builds on that decoder. The native engine does not serve EXL3 packs yet.

## More Python engine work

These contributions shipped in the Python engine and install from the `v0.6.6` tag. Some reach the native engine by
its own route, such as the one server for both backends, and many wait for its families and backends. All of them are
credited here.

- EXL3, caches and memory: @vcruz305 ([#42](https://github.com/ashhart/TensorFold/pull/42) packs and
  `return_token_ids`, [#47](https://github.com/ashhart/TensorFold/pull/47) int8 and int4 KV caches and
  `--mtp-confidence`, [#69](https://github.com/ashhart/TensorFold/pull/69)), @gilby
  ([#68](https://github.com/ashhart/TensorFold/pull/68) disk spill and `--checkpoint-slots`), @akol1
  ([#150](https://github.com/ashhart/TensorFold/pull/150), [#151](https://github.com/ashhart/TensorFold/pull/151)),
  @barelyworkingcode ([#186](https://github.com/ashhart/TensorFold/pull/186)), @shantanugoel
  ([#145](https://github.com/ashhart/TensorFold/pull/145)), @grearjake-star
  ([#254](https://github.com/ashhart/TensorFold/pull/254)), @Chedrian07
  ([#49](https://github.com/ashhart/TensorFold/pull/49)), @sanjaibalajee
  ([#74](https://github.com/ashhart/TensorFold/pull/74)), @di37 ([#206](https://github.com/ashhart/TensorFold/pull/206)).
- Images and video: @di37 ([#64](https://github.com/ashhart/TensorFold/pull/64)), @edurdias
  ([#170](https://github.com/ashhart/TensorFold/pull/170)), @barelyworkingcode
  ([#187](https://github.com/ashhart/TensorFold/pull/187)), @MiaAI-Lab
  ([#239](https://github.com/ashhart/TensorFold/pull/239), [#240](https://github.com/ashhart/TensorFold/pull/240), and the
  Flash Next vision patch adapted in #146), @mgoldwasser ([#101](https://github.com/ashhart/TensorFold/pull/101)),
  @shantanugoel ([#146](https://github.com/ashhart/TensorFold/pull/146)), @BobClawblaw
  ([#473](https://github.com/ashhart/TensorFold/pull/473)).
- Model families: @cshintov (Gemma 4 in [#10](https://github.com/ashhart/TensorFold/pull/10), Qwen3.6 MoE on Macs in
  [#138](https://github.com/ashhart/TensorFold/pull/138)), @di37 ([#224](https://github.com/ashhart/TensorFold/pull/224)),
  @philip-pentatonic (Qwen3.6 on CUDA in [#78](https://github.com/ashhart/TensorFold/pull/78) and
  [#84](https://github.com/ashhart/TensorFold/pull/84)), @BHCC2025 ([#270](https://github.com/ashhart/TensorFold/pull/270)).
- Flash Next on CUDA: @tournierjc (NVFP4 checkpoints in [#67](https://github.com/ashhart/TensorFold/pull/67), then
  [#102](https://github.com/ashhart/TensorFold/pull/102) and [#105](https://github.com/ashhart/TensorFold/pull/105)),
  @jschmied ([#174](https://github.com/ashhart/TensorFold/pull/174), [#176](https://github.com/ashhart/TensorFold/pull/176),
  [#178](https://github.com/ashhart/TensorFold/pull/178), [#180](https://github.com/ashhart/TensorFold/pull/180) with
  @plotarmordev's test), @plotarmordev ([#222](https://github.com/ashhart/TensorFold/pull/222)), @BHCC2025
  ([#141](https://github.com/ashhart/TensorFold/pull/141)), @MovieMaker93
  ([#40](https://github.com/ashhart/TensorFold/pull/40), [#73](https://github.com/ashhart/TensorFold/pull/73),
  [#93](https://github.com/ashhart/TensorFold/pull/93)), @benthecarman ([#124](https://github.com/ashhart/TensorFold/pull/124),
  with @Arminova's resend test), @lcgutierrez ([#163](https://github.com/ashhart/TensorFold/pull/163)),
  @mcclanahanaman ([#201](https://github.com/ashhart/TensorFold/pull/201), [#202](https://github.com/ashhart/TensorFold/pull/202)),
  @simon-lin88 ([#248](https://github.com/ashhart/TensorFold/pull/248)), @MiaAI-Lab
  ([#238](https://github.com/ashhart/TensorFold/pull/238)), @BobClawblaw ([#468](https://github.com/ashhart/TensorFold/pull/468)),
  @Mirrdhyn (decisions on Flash Next in [#232](https://github.com/ashhart/TensorFold/pull/232), and the shared-prefix
  design in 0.6.1).
- GLM-5.3-Flash on CUDA: @taussoe ([#43](https://github.com/ashhart/TensorFold/pull/43), and the latent MLA cache to
  256k tokens in [#54](https://github.com/ashhart/TensorFold/pull/54)), @MiaAI-Lab
  ([#128](https://github.com/ashhart/TensorFold/pull/128), [#129](https://github.com/ashhart/TensorFold/pull/129),
  [#131](https://github.com/ashhart/TensorFold/pull/131), [#132](https://github.com/ashhart/TensorFold/pull/132),
  [#134](https://github.com/ashhart/TensorFold/pull/134)), @mikolaj92 ([#140](https://github.com/ashhart/TensorFold/pull/140)),
  @akol1 ([#152](https://github.com/ashhart/TensorFold/pull/152)), @plotarmordev
  ([#225](https://github.com/ashhart/TensorFold/pull/225)), @liumorrisclaw (four ranks in
  [#465](https://github.com/ashhart/TensorFold/pull/465)).
- Qwen3.8-27B on CUDA: @nood-co1 ([#58](https://github.com/ashhart/TensorFold/pull/58),
  [#59](https://github.com/ashhart/TensorFold/pull/59), [#92](https://github.com/ashhart/TensorFold/pull/92),
  [#125](https://github.com/ashhart/TensorFold/pull/125)), @jkuepker ([#100](https://github.com/ashhart/TensorFold/pull/100),
  landed through [#112](https://github.com/ashhart/TensorFold/pull/112)), @Arminova
  ([#268](https://github.com/ashhart/TensorFold/pull/268)).
- Shared CUDA: @jayleaton (the communicator interface in [#219](https://github.com/ashhart/TensorFold/pull/219)),
  @liumorrisclaw (Ampere cards with compute capability 8.0 in [#464](https://github.com/ashhart/TensorFold/pull/464)).
- The Python servers: @nood-co1 ([#57](https://github.com/ashhart/TensorFold/pull/57),
  [#61](https://github.com/ashhart/TensorFold/pull/61), [#62](https://github.com/ashhart/TensorFold/pull/62),
  [#63](https://github.com/ashhart/TensorFold/pull/63)), @cshintov ([#199](https://github.com/ashhart/TensorFold/pull/199)),
  @olexale ([#114](https://github.com/ashhart/TensorFold/pull/114)), @MiaAI-Lab
  ([#75](https://github.com/ashhart/TensorFold/pull/75)), @philip-pentatonic
  ([#111](https://github.com/ashhart/TensorFold/pull/111), and `--name-priority` in
  [#445](https://github.com/ashhart/TensorFold/pull/445)), @akol1 (`tensorfold plan` in
  [#281](https://github.com/ashhart/TensorFold/pull/281)).
- Apple Silicon: @gilby ([#147](https://github.com/ashhart/TensorFold/pull/147),
  [#149](https://github.com/ashhart/TensorFold/pull/149), [#164](https://github.com/ashhart/TensorFold/pull/164)), @feni6
  ([#118](https://github.com/ashhart/TensorFold/pull/118), [#119](https://github.com/ashhart/TensorFold/pull/119)).
- Tooling, tests and docs: @kingjamez ([#44](https://github.com/ashhart/TensorFold/pull/44)), @di37
  ([#189](https://github.com/ashhart/TensorFold/pull/189)), @akol1 ([#276](https://github.com/ashhart/TensorFold/pull/276)),
  @plotarmordev ([#221](https://github.com/ashhart/TensorFold/pull/221), [#288](https://github.com/ashhart/TensorFold/pull/288),
  [#289](https://github.com/ashhart/TensorFold/pull/289)), @taussoe ([#66](https://github.com/ashhart/TensorFold/pull/66)),
  @jasontitus ([#33](https://github.com/ashhart/TensorFold/pull/33)), @feni6
  ([#120](https://github.com/ashhart/TensorFold/pull/120)).
