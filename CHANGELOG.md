# What's new in TensorFold

`tensorfold update` prints the sections below that are newer than the version you had. Each release's page on
GitHub has the full notes and the measurements behind them.

## 0.3.7 (29 Sep 2026)

- **Nemotron on Macs serves concurrent requests again.** 0.3.6.3 kept one draft slot for all of a server's streams,
  so several Nemotron requests at once could fail with HTTP 500 (5 of 6 in our test on an M3 Ultra). Each stream now
  keeps its own: six requests at once all finish, and each reply equals the same request sent alone.
- **Qwen3.8-27B on M1-M4 serves 9, 17 or 25 streams at once.** A round whose last group of recurrent streams held
  one stream built a kernel Metal refused, and every stream in that round failed. That group now builds like the
  others; rounds of 2-8 streams run the same kernels as before.
- Versions have three parts from here on.

## 0.3.6.3 (29 Sep 2026)

- **Images in, on Macs and Sparks.** `--vision` lets Qwen3.8-27B read up to four images a request, as data URLs or,
  with `--vision-urls`, HTTPS links. Image replies keep drafting and equal `"draft": false` ones (77.6 against 37.3
  tok/s serial on an M3 Ultra), and the first token comes as fast as mlx-vlm's. Thanks to @di37 for HTTPS-only
  fetching, bounded image preparation and redacted request logs (#64).
- **DeepSeek-V4-Flash on a 256 GB Mac.** The new `deepseek_v4` family serves `mlx-community/DeepSeek-V4-Flash-4bit`
  on the lane engine with DSpark or MTP drafts converted from DeepSeek's releases, and replies equal `"draft": false`
  ones. On an M3 Ultra it decodes 1.7-3.1x and reads prompts 1.8-2.0x as fast as mlx-lm PR #1797's server. Thanks to
  @jeffpeng3 (#14).
- **NVFP4 checkpoints on CUDA, as published.** Flash Next's NVFP4 exports (local-inference-lab's and RadixArk's)
  and NVIDIA's Qwen3.8-27B NVFP4 load through `tensorfold serve` and stay exact: drafted replies equal `"draft": false`
  ones, resumed prompts equal fresh ones, and 84 of 84 concurrent streams equal their solo runs. The experts read
  their routing on the GPU, so CUDA graphs replay any routing. On one Spark, Flash Next NVFP4 decodes 1.13-1.52x vLLM
  on the same checkpoint. Two ranks, `--ple-on-ssd` and images on NVFP4 checkpoints stop at startup with a message
  until they're qualified. Thanks to @tournierjc for the reader (#67).
- **Ternary Bonsai 2 27B** (prism-ml's 2-bit pack) serves on the lane engine with Qwen3.8-27B's DFlash2 drafts,
  exact. Against mlx_lm running the pack's own runtime on an M5 Max: decode 3.7-5.8x on code and 1.7-2.2x on chat,
  prompts 1.4-1.7x. Thanks to @gprot42 (#18).
- **CUDA server work from @nood-co1:** stop strings on every CUDA engine and `ignore_eos` on the 27B (#63);
  malformed requests get a 400, and failed ones a 500 or a stream error event (#61); a silent start explains itself
  with extension-build progress and stale-lock notes, and `kill -USR1 <pid>` prints every thread's stack, at start or
  while serving (#62); the 27B keeps its prompt cache entry one token before the prompt's end, so the next chat turn
  resumes from it (#65).
- **Replies keep decoding while long prompts prefill on Macs.** A prompt now fills one planned chunk at a time and
  running replies take rounds between its chunks, for `--decode-share` of each chunk's time (default 0.25, about a
  fifth of the time; 0 prefills whole prompts first, as before); every reply still equals its solo run. On an M3
  Ultra, a reply's longest pause while three 17K-token prompts arrive fell from 164 s to 7 s, and a request alone is
  unchanged. Thanks to @benwilson (#72).
- **Long conversations stay warm on Macs.** A checkpoint that can't fit no longer evicts the others first, a resumed
  turn no longer holds the stored prefix it copied, and the 27B's DFlash2 prompt taps take 64 KB a token instead of
  114. On a 64 GB budget (emulated on an M3 Ultra), one conversation grew to 143K tokens and each turn resumed in
  38-46 s, where 0.3.6.2 prefilled every turn past about 100K from the start. Startup names the longest request
  whose prompt is kept. Thanks to @sanjaibalajee (#74), and to @benwilson for the report and his
  64 GB measurements, now in the README labelled 0.3.5.1 (#71, #70).
- **Qwen3.6 on CUDA keeps serving past 8,192 rows** with expandable allocator segments: its graphs now capture into a
  new pool after the buffers grow, where every later request failed. Thanks to @philip-pentatonic (#78).
- **Typed tool arguments on CUDA:** a Qwen tool call's array, object, number and boolean parameters arrive as JSON
  values, as on the Mac. Thanks to @MiaAI-Lab (#75).
- **Flash Next reads a prompt chunk's n-gram rows on 16 threads,** same bytes, so prompt times vary less when the
  tables are not all in the page cache. Thanks to @MovieMaker93 (#73).
- **On M1-M4, the MoE prompt matmuls give each GPU tile one expert's rows,** the scheduling idea of MLX's gather_mm
  change (ml-explore/mlx#4567), written for 4-bit gathers. Same bits. On an M3 Ultra, Flash Next's prompts run 2-4%
  faster (1.02-1.20x oMLX's) and GLM-5.3's 3.4-3.8% faster (1.08-1.11x mlx-vlm's at 8k-32k).
- **Flash Next prompts on M1-M4:** the block scores run as one batched matmul, the GQA kernel scores two query heads
  a simdgroup and the top-512 selection finds its cuts with a simdgroup scan. Same bits, about 1% faster at 16k-64k;
  on an M3 Ultra it is level with oMLX at 32k and 64k.
- **MLX stays at 0.32.2 for now.** MLX 0.32.3 changed a kernel our prompt kernels build on, so fresh installs fell
  back to slower prompts with one log line; the next release follows the new kernel. Thanks to @ecohash-co (#88).

## 0.3.6.2 (28 Sep 2026)

- **EXL3 replies stop at the end of the turn.** Flash Next and 27B EXL3 packs list `<|im_end|>` only in
  `generation_config.json`, so replies ran past their turn and leaked tool calls and think tags. The CUDA engine now
  reads that file too. Thanks to @vcruz305 (#69).
- **pip installs serve 27B EXL3 packs.** The package was missing the 27B's CUDA sources; a test now checks that every
  CUDA source ships. Thanks to @taussoe (#66).
- **A quantized KV cache for Flash Next on CUDA.** `--kv-dtype int8` or `int4` holds about 1.7x or 2.6x the default
  window in the same memory, and drafted replies still equal serial ones. `--mtp-confidence` sets where MTP chains
  stop. Thanks to @vcruz305 (#47).
- **Flash Next on CUDA reaches the first token sooner.** The head runs on a prompt's final chunk only, and the prompt
  kernels load at startup, so the first 2k prompt takes 1.26 s instead of 1.79 s. Thanks to @MovieMaker93 (#40).
- **GLM on two Sparks holds 256k tokens** with a latent attention cache; its next-token loss is within 0.001 nats of
  the per-head cache. Thanks to @taussoe (#54).
- **Mixed-bit Qwen checkpoints** (4-bit with some 5- and 6-bit layers, such as oQ4) load on every lane backend,
  exact, with new row kernels for 2- to 8-bit weights. On an M3 Ultra, oQ4 27B decodes 114-120 tok/s on code and
  59-61 on chat, against mlx_lm's 34-36.
- **Concurrent 27B on CUDA:** `--parallel 16` serves 161.7 tok/s on one Spark in 25.4 GiB, each reply equal to its
  solo run (#38). **Qwen3.6-35B-A3B on CUDA,** exact: decode 1.36-1.49x vLLM with MTP, prompts 1.21-1.37x (#45).
- **Gemma 4 drafts (opt-in):** `--drafter z-lab/gemma-4-26B-A4B-it-DFlash` decodes 1.3-2.1x mlx_lm on an M3 Ultra,
  exact.
- **Tool calls:** `tool_choice: "required"` and a named tool are enforced on both servers (#52), and a complete tool
  call inside an unclosed think block comes back as a tool call (#60).
- **Conversations come back warm on Macs.** `--spill-gib N` writes a conversation pushed out of the prompt cache
  to disk, up to N GiB, and reads it back when the conversation returns. On a 48 GB budget a 35k-token
  conversation came back in 0.27 s on an M5 Max instead of 75 s, with the same reply. Off by default;
  `--checkpoint-slots` sets how many conversations stay in memory. Thanks to @gilby (#68, #55).
- **Memory:** `TENSORFOLD_MEMORY_LIMIT_GB` raises the budget above the default share, and Flash Next's memory check
  counts its host-mapped n-gram tables, so it starts on a 128 GB Mac. Thanks to @Chedrian07 (#49, #50).
- **CUDA server fixes from @nood-co1:** an abandoned request stops within a round (#57), keys and values stay within
  the admitted window (#58), and a failed admission no longer stops the scheduler (#59).
- **M1-M4:** a prompt split into parts attends exactly as it does in one piece at every length; two 8,192-key cases
  rounded differently in 0.3.6.

## 0.3.6.1 (28 Sep 2026)

- **CUDA builds inside NVIDIA's containers again.** Their `TORCH_CUDA_ARCH_LIST` names every architecture back to
  sm_80, so the kernels' thread-block clusters and FP8 MMA failed to compile for GPUs that lack them. Every extension
  now builds for the GPU that is present, and a GPU older than compute capability 9.0 gets a clear message. Thanks to
  @ss-cong for the report and the exact errors (#56).

## 0.3.6 (28 Sep 2026)

- **GLM-5.3-Flash on Macs** with 256 GB, drafted replies equal to serial ones. Prompts process at or above mlx-vlm
  from 2k to 32k tokens on an M3 Ultra, and tool calls parse in both servers. Thanks to @chadhurley25075-png (#9,
  #39) and @jeidbugs404 (#35).
- **Gemma 4 26B-A4B on the lanes,** exact at every width, with prompts at or above mlx_lm from 2k to 64k tokens on
  an M3 Ultra. Thanks to @cshintov (#10).
- **Bigger models on smaller Macs.** `--ple-on-ssd` reads Flash Next's n-gram tables from disk, so a 128 GB Mac holds
  it (#16). `--ssd-experts GIB` streams routed experts from the checkpoint into a GPU pool of that size, so Flash Next
  fits a 64 GB Mac and GLM a 128 GB one. Replies are the resident model's tokens; decode runs at 0.31-0.39x resident
  speed for Flash Next and 0.13-0.17x for GLM (measured on an M3 Ultra; `pip install "tensorfold[ssd]"` first) (#17).
- **EXL3 checkpoints on CUDA (experimental):** Qwen3.8-27B and Flash Next packs from turboderp, exact on the lanes.
  Decode runs 1.6-3.6x vLLM with MTP; prompt processing is about half the MLX checkpoints' speed for now, and the
  fix is next. Thanks to @vcruz305 (#42).
- **Faster prompts.** Flash Next sizes its prompt chunks to the memory it has, Nemotron takes up to 8,192 tokens a
  chunk on M5 GPUs, and the weights stay wired while a server runs.
- **Fixes:** GLM on two Sparks answered "!" past about 2,000 prompt tokens, and EXL3 GLM prompts past 128 tokens
  failed (#53). A reply that isn't a tool call comes back as content, not an HTTP 500 (#51). The server hands MLX's
  freed buffers back when it goes idle (@kingjamez, #44).
- **MLX 0.32.2 or newer** is required on Macs.
- **What's new after an update:** `tensorfold update` prints these notes when it finishes.
- **Known:** replies to prompts longer than one prompt chunk can differ between machines with different memory,
  because the chunk size follows the memory budget. Within one server, drafted replies always equal serial ones and
  resumed prompts equal fresh ones.

## 0.3.5.1 (28 Sep 2026)

- Qwen3.8-27B loads on M1 and M2 Macs again. Kernels there fit Metal's per-kernel thread limit, with the same sums in
  the same order, so drafted output still equals serial output.
- M3, M4 and M5 run 0.3.5's machine code unchanged.
- Thanks to @hichaiuse, @simonmd, @gcarusso, @tonydehnke, @Cyb3r-Monk and @tinyapps for the reports and the repro.

## 0.3.5 (27 Sep 2026)

- **Concurrent requests share each verification round.** `--parallel auto` is on by default, and every stream's reply
  equals the same request served alone, on Metal and on CUDA.
- **Follow-up turns resume at the start of their newest messages,** with output identical to a fresh prompt. A 12-turn
  agent session with the 27B spent 14.9 s on first tokens instead of 31.9 s.
- **Memory that fits.** The whole process stays inside 70% of RAM, an omitted `--context` defaults to the window the
  machine can hold, and a prompt past it gets a clear 400.
- **Flash Next prefill** with sparse prompt attention, 1.1-1.4x faster than 0.3.4.1 on an M3 Ultra (@quigles1977, #29).
- **2- to 8-bit weights** on the lanes, so mixed-precision 27B checkpoints decode fully (@jasontitus, #34).
- **CUDA:** FP8 prefill and shared expert kernels, concurrent streams for the 27B and Flash Next, and admission from
  available memory before loading.
- **API:** raw `/v1/completions` prompts, `ignore_eos` and `stop`, request `reasoning_effort` and typed tool arguments
  (@chris247474, #28), `developer` messages, `parallel_tool_calls: false`, and cancellation when a client disconnects.

## 0.3.4.1 (27 Sep 2026)

- Prompt processing is back to MLX's speed on every model. Prompts prefill through MLX's own forward on a fixed
  2,048-token grid, so a resumed conversation still equals a fresh one byte for byte.
- Flash Next's peak memory stays within 20 GB of its weights up to a 196k-token prompt.

## 0.3.4 (26 Sep 2026, pre-release)

- Every model runs on the lane engine, and the serial engine is gone. Nemotron drafts on M1 to M4 with row-exact
  kernels.
- Qwen3.8-27B on M1 to M4 decodes at 1.9 to 4x serial, through a new 4-bit matmul on the simdgroup matrix units.
- CUDA: every default path verifies at least two rows a round.

## 0.3.3 (26 Sep 2026)

- Qwen3.8-27B verifies drafted tokens together on every M1 to M5 GPU, with output byte-identical to serial decoding.
- Streamed `/v1/completions` send plain text.

## 0.3.2 (26 Sep 2026)

- `tensorfold update` installs the newest release.
- GLM-5.3-Flash reads Mia-AiLab's EXL3 weights on two DGX Sparks (experimental).

## 0.3.1 (26 Sep 2026)

- `tensorfold info` shows how a checkpoint stores its weights and which backends read them. `serve` and `pull` refuse
  checkpoints no engine reads yet, before anything downloads.

## 0.3.0 (26 Sep 2026)

- NVIDIA GPUs: `tensorfold serve` picks CUDA on Linux and runs on one GPU or two (one rank per DGX Spark).
- A new family, GLM-5.3-Flash, on two Sparks.

## 0.2.0 (25 Sep 2026)

- A rewrite: `tensorfold serve`, `pull`, `models` and `info` for Nemotron 3.5 Lightning, Qwen3.8-27B and Qwen3.8
  Flash Next on Apple Silicon, with drafts that never change the output.
