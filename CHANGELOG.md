# What's new in TensorFold

Each release's page on GitHub has its notes and measurements. See [the 1.0.0 release notes](RELEASE-NOTES-1.0.0.md) for the native binary's supported models and migration details.

## 1.0.2

- GLM-5.3-Flash serves several requests at once on its two Macs. Eight concurrent requests decode at 199 tokens a second in total, against 100 in 1.0.1, and each stream equals its solo run.
- GLM's two Macs split the attention heads, the dense layers and the vocab head as well as the experts. One stream decodes plain at 91 tokens a second, from 83. With drafts it decodes 114 to 131 tokens a second on prose and 138 to 148 on code, each reply equal to its plain one. A fresh 7,300-token prompt reaches its first token in 4.4 seconds instead of 6.5.
- GLM keeps prompt states between requests. A resumed prompt equals a fresh one token for token, and a 7,300-token conversation resumes in 0.18 seconds.
- A new conversation that starts with the same system prompt and tools resumes all of that shared prefix: its first token came in 0.06 seconds after a 4,700-token system prompt.
- `--learn` keeps those shared prefixes on disk. After a restart, the first new conversation on a known system prompt reached its first token in 0.64 seconds instead of 4.43. An upgrade keeps them when a startup check finds its prompt passes give the same bits. `--learn-gib` caps their disk, 32 GiB by default. GLM only for now.

## 1.0.1

- The shipped binary has `tensorfold models`, `tensorfold info MODEL` and `tensorfold pull REPO`. In 1.0.0 they were built into the development CLI only, so a Homebrew install didn't have them.

## 1.0.0

- TensorFold is a native Zig server. The shipped engine loads checkpoints, tokenizes requests, renders chat templates and serves HTTP without a Python engine. Python 0.6.6 remains on `python-0.6` and the `v0.6.6` tag.
- Four model families are qualified for this release: Nemotron 3.5 Lightning on Metal from M1 through M5 and on NVIDIA GB10 CUDA, greedy and sampled; Qwen3.8 Flash Next on M5 Ultra Metal; GLM-5.3-Flash across two M5 Ultra Macs; and Qwen3.5-2B on M5 Max Metal.
- Flash Next loads its checkpoint directly. Serving it no longer requires a Python kernel recording. The native loader builds the layouts it needs from the checkpoint and uses the embedded Metal sources.
- Native server interfaces: OpenAI chat/completion routes, Anthropic Messages and token counting, tools, streaming, tokenization, health, metrics and the optional dashboard. `capabilities --json` and `serve MODEL --help` describe the binary's supported options.
- Opt-in context compaction (`--compact-at`, `--compact-keep`, `--compact-memory`) compacts a conversation that would overflow instead of refusing it. Nemotron runs concurrent requests in shared rounds on Metal and CUDA.
- Metal runtime fixes: `--keep-warm` controls idle GPU keepalive, and a load-time probe selects a prebuilt packed-kernel library when the runtime compiler refuses that kernel on macOS 26.3.
- Native release archives target macOS arm64 and Linux aarch64 for the GB10, with the executable and bundled license notices. CUDA deployment still needs its qualified kernel assets and NVIDIA driver; paired Metal serving needs the documented transport setup.
- CUDA GB10 qualification is greedy. Sampled requests do not yet honor the seed, `top_k: 0` fails on that path, and health/metrics do not yet report CUDA device and pinned-memory usage. Those fixes are planned for 1.0.1.
- Qwen3.8-27B is excluded: drafted/plain equality passes, but its paired served decode and cold-prefill measurements remain slower than Python 0.6.6. Bonsai, Gemma 4, Qwen3.6 and DeepSeek-V4 remain ports for 1.0.x, subject to their model gates.
- Contributor credits: all 145 public contributors are listed in [RELEASE-NOTES-1.0.0.md](RELEASE-NOTES-1.0.0.md).

Earlier entries document the Python release line.

## 0.6.5 (3 Oct 2026)

- **Qwen3.6-35B-A3B drafts with its own MTP layer on Macs,** as on CUDA. Chains of up to four drafts are verified in
  the lane rounds, alone or with other streams, and each round's depth comes from the Mac's measured costs, with plain
  rounds where drafts do not pay. Replies equal `"draft": false`. On an M3 Ultra it decodes 1.5-2.1x `--no-drafts` on
  short prompts, thinking on or off, and 1.3x at a 28,400-token prompt with the same time to first token: 1.5-1.7x
  mlx-vlm 0.7.4 with its MTP drafter. DFlash v1 stays available with `--drafter z-lab/Qwen3.6-35B-A3B-DFlash`.
- **Nemotron on CUDA drafts as deep as its rows pay for.** The MTP head drafts up to 15 levels, one CUDA graph each,
  and a chain stops where the next verify row would cost more time than its draft is expected to save. Window and
  level costs are measured at startup on each GPU. Against 0.6.4's default this decodes about 6% faster on one DGX
  Spark and on two, and 7% faster on an RTX PRO 6000, with replies equal to `"draft": false`. `--mtp-confidence` sets
  a fixed floor instead.
- **Nemotron verify windows on M5 Macs read each routed expert once from two rows** (eight before). One stream decodes
  1-3% faster at three to five lanes, with the same tokens.
- **API keys on both servers:** `--api-key` (repeatable), `--api-key-file` (one key, or `label: key`, a line; reread on
  `SIGHUP`) or `TENSORFOLD_API_KEY`. Requests send `Authorization: Bearer` or `x-api-key`. `/health` stays open;
  `/metrics` needs a key unless `--metrics-open`.
- **Thinking stays the chat template's default, and the server says so:** one startup line names `--no-thinking`, and
  a reply that reaches `max_tokens` before it leaves its think block (empty `content`, all `reasoning_content`) gets a
  warning line.
- The test suite collects on Macs that have PyTorch but not Triton.

## 0.6.4 (3 Oct 2026)

- **Flash Next on two DGX Sparks serves concurrent requests.** `--parallel N` with two CUDA ranks runs every stream
  in shared lane rounds, and drafted replies still equal one-token decoding (#141, #180, #219, closes #123). The two
  ranks now share one communicator interface, so a faster transport can plug in without touching the engines.
  Thanks to @BHCC2025, @jschmied, @plotarmordev and @jayleaton.
- **Startup memory on Macs:** prompt chunks are probed smallest first, and a larger probe runs only when its worst
  case fits the memory budget (#271). Thanks to @boxabirds for the report.
- **`tensorfold plan`** prints the model and context budget before any weights load (#281), and both `/metrics`
  routes report the process memory footprint (#277). Thanks to @akol1.
- Flash Next on CUDA builds its extensions before loading weights (#202) and gathers prompt n-gram rows before
  waiting on earlier copies (#201). Thanks to @mcclanahanaman.
- Flash Next on CUDA rereads EXL3 n-gram tables after warm-up and pins their page runs (#254). Thanks to
  @grearjake-star.
- `--decode-share` takes effect on Flash Next CUDA: shared prompt rows are sized from completed stand-alone passes
  (#248, closes #230). Thanks to @simon-lin88.
- CUDA tree attention folds its partial sums in fp32 groups. Precision is unchanged, but long replies can differ
  from 0.6.3 in the last bits (#268). Thanks to @Arminova.
- Flash Next on CUDA refuses cards below sm_120 at startup, before loading weights, and its CUDA tests skip there
  (#261). Thanks to @mcclanahanaman for the report.
- `docs/recipes/README.md` lists the minimum card and Mac memory for each family (#139). Thanks to @tomByrer for
  asking.
- Benchmark helpers fail early on Python below 3.11 (#276). Thanks to @akol1.
- Tests collect without the optional MLX or terminal-interface dependencies (#288), and the docs say `--alias`
  works on the CUDA server too (#289). Thanks to @plotarmordev.

## 0.6.3 (2 Oct 2026)

- **Nemotron on M5 Macs: copied text verifies up to 64 tokens a round.** A lone stream's copy window grows from 16 to
  64 rows while each lands whole, a window attends in one call, and wide windows keep their Mamba states every 8th
  row. On an M5 Max, one stream editing code decodes about 1.4x faster (550 to 760 tok/s, and up to 1,000 on a cool
  machine), prose and code writing gain 1.6-3.3%, and drafted replies still equal one-token decoding.
- **Anthropic Messages API.** `/v1/messages` and `count_tokens` on the Mac and CUDA servers, with streaming, tools,
  thinking and cache usage (#223, closes #168). Thanks to @kky42.
- **Control Room.** `tensorfold service` runs a model as a launchd service, and `tensorfold tui` shows its decode and
  prefill speed and its connections live.
- **Flash Next on Macs before M5:** `TF_FLASH_DENSE=matrix` runs every dense projection on the matrix units, and the
  fused DeltaNet stack uses the matrix kernel by default (#149). Thanks to @gilby.
- **Vision:** `--vision-offload` keeps the CUDA image tower in host RAM between images (#187, closes #185); image
  parts are accepted inside tool results (#235); `--vision-image-tokens` lets many images share a larger budget
  (#239); Flash Next on CUDA takes video (#240). Thanks to @barelyworkingcode and @MiaAI-Lab.
- **CUDA:** `/v1/decisions` on Flash Next scores labels from the prompt's last logits, the questions filling in one
  pass (#232); `TENSORFOLD_PREFILL_ROWS` sets the prompt piece rows (#238); NVFP4 and FP8 prompt rows add a tile's K
  slices in one block, with the same bits (#242); `TENSORFOLD_MEMORY_RESERVE_GIB` moves the memory floor the startup
  keeps free, by default a tenth of the pool and at least 4 GiB as before (#165); Flash Next's chain kernel takes
  5-17% less time at 2 to 8 rows, with the same bits. Thanks to @Mirrdhyn, @MiaAI-Lab, @jschmied and @eleqtrizit.
- **Server:** chunked request bodies are decoded before JSON parsing (#244); `/tokenize` and `/detokenize` carry
  vLLM's fields (#237); `chat_template_kwargs.thinking` is heard as `enable_thinking`, and GLM-5.3 keeps earlier
  turns' reasoning (#236); streaming usage gets its own chunk when `stream_options.include_usage` asks for it (#216);
  malformed tool-call history renders safely (#233); `/metrics` times each request's decode (#269); `/health` carries
  the live decode and prefill speed. Thanks to @JordiPosthumus, @MiaAI-Lab, @salmanarshad321 and @sxuff.
- **Fixes:** an EXL3 layer with no bias no longer faults in the split-K reduction (#186); an 8-bit `lm_head` keeps its
  format in Qwen3.6's MTP draft head on CUDA (#270); a lone Flash Next stream's cache growth on CUDA stays inside the
  explicit memory budget; two ranks refuse to start with different prompt rows. Thanks to @barelyworkingcode and
  @BHCC2025.
- **Models moved to the TensorFold Hugging Face org.** `Vontra/<name>` ids redirect, and the tree now names
  `TensorFold/<name>`.
- **Contributing.** `CONTRIBUTING.md` says what a pull request needs to land and how it lands, and new pull requests
  open with a receipt template.

## 0.6.2 (2 Oct 2026)

- **Flash Next on Macs at 64k-128k.** On an M3 Ultra, one stream runs 1.2-3.4% faster at 64k and 3.9-5.5% at 128k,
  with the same tokens: a window's n-gram ids are hashed on the GPU, and the chain's first step is built while the
  GPU verifies.
- **27B with several streams on CUDA.** The GDN tree kernel takes 8-35% less time with the same bits. On an RTX PRO
  6000 at its 250 W limit, 4 and 8 streams of the NVFP4 27B decode 1.1-4.2% faster.
- **Fixes:** the config check accepts the FP8 n-gram table in NVIDIA's MIXED_PRECISION Flash Next export; GLM-5.3
  on CUDA counts its drafts in `/health`, `/metrics` and replies, and names a mixed-bit EXL3 checkpoint when it
  refuses one; a client that leaves is noticed past file descriptor 1023; the CUDA server prints a line a request, as
  the Mac server does; a failed snapshot write no longer leaves its partial file; Gemma 4's QKV kernel reserves its
  1024 threads for M1, M2 and macOS VMs, and a VM's GPU is no longer taken for an M5.

## 0.6.1 (1 Oct 2026)

- **NVFP4 checkpoints in their own math.** `nvidia/Qwen3.8-27B-NVFP4` runs the 4-bit activations its checkpoint
  names, as vLLM does; `--precision full` runs 16-bit activations against the same weights. On an RTX PRO 6000 at its
  250 W limit, one stream decodes 1.4-2.0x vLLM and prompts fill at 0.95-0.97x its speed.
- **Waiting prompts fill together on CUDA.** With `--parallel`, prompts that arrive together now share one prefill
  forward instead of filling one a round: on an RTX PRO 6000 at its 250 W limit, 8 streams of the 27B run 1.14-1.36x
  faster and the slowest first token comes in 0.05-0.10 s instead of 0.4-0.8 s (1.6 s to 0.15 s on a DGX Spark), with
  the same replies.
- **More 27B tokens with several streams on CUDA.** Streams plan on their measured round cost, and wider lane blocks
  on RTX PRO and RTX 50 cards add 4-8% at 8 streams, with the same tokens.
- **Flash Next on Macs at long context.** One stream runs up to 9.6% faster on code at 64k and 10.2% on chat at 128k
  on an M3 Ultra, with the same tokens.
- **`/v1/decisions`** scores a choice, a score or a yes/no from the next-token logits, on the shared prompt lanes.
- **Flash Next on CUDA:** image input, forks that resume from their shared prefix, shared system prompts copied
  instead of filled again, and short prompts admitted while a long one fills.
- **RTX cards without Docker:** pip alone installs and builds the CUDA kernels. Native Windows is in as an
  experimental host layer, not yet run on Windows hardware.
- **Fixes:** a refused request no longer breaks the next one on its connection; Gemma 4 thought blocks stay out of
  replies with thinking off; an unnamed reasoning effort goes to the nearest named level; mlx-lm 0.32 support.

## 0.6.0 (30 Sep 2026)

- **RTX 40 cards.** CUDA now runs on compute capability 8.9 (Ada). On one RTX 4090 the 27B serves with DFlash2 in a
  40,182-token window, exact: drafted replies equal serial ones, and a resume or resend gives a fresh run's reply.
  Prompts fill at 2.4-2.6k tok/s from 2k to 32k tokens, and replies decode at 64-86 tok/s. On one GPU the 27B's kept
  prompt states give way, oldest first, when a live reply needs the room, so admission counts one live window.
- **Prompts fill inside the decode rounds.** With `--parallel` on CUDA, Flash Next prefills a queued prompt in the
  same forward as the live replies instead of stopping them: on one DGX Spark, first tokens came 2.8-3.1x sooner than
  on 0.5.0. On Macs several prompts fill side by side, the fewest tokens left first: on an M3 Ultra, short requests
  queued behind a long prompt got their first token in a median 9.9 s instead of 113 s (the slowest 11.7 s, not 130).
- **Conversations resume on more engines.** An identical resend or the next thinking turn now resumes from the kept
  prompt state on Flash Next, Qwen3.6, Nemotron and GLM-5.3 on CUDA, as the 27B did, with a fresh run's reply: an
  18.7k-token Flash Next resend went from 8.3 s to 0.08 s. On Macs the prompt cache keeps each conversation's newest
  checkpoint and grows into memory the model leaves idle.
- **CUDA prompts run at bf16 by default.** Against an fp32 reference the 27B's prompt rows are 20x closer than with
  FP8 (KL 0.0031 against 0.0624). `--prefill-fp8` keeps 0.5.0's faster FP8 prompts for those who want them.
- **Tool calls for agents.** The CUDA server streams tool-call arguments as the model writes them (the longest
  silence in a long call fell from about 30 s to half a second), Python-spelled values like `False` and `None`
  decode to their schema types, Gemma 4's bare tool calls parse (#121), and a prompt past the context window gets
  OpenAI's `context_length_exceeded`, so clients compact instead of retrying.
- **More checkpoints.** GLM-5.3 on Macs reads 8-bit and Q8_0 GGUF checkpoints and takes images, and has an opt-in
  float32 activation mode (bf16 stays the default). On Macs, Flash Next loads oMLX's oQ checkpoints with scaled
  n-gram tables; on CUDA it reads consolidated EXL3 tables and block-scaled FP8 linears.
- **Faster.** Flash Next NVFP4 decodes 4.9-7.1% faster and fills prompts 15-16% faster at 2k-16k, bit-identical. On an
  M5 Ultra the 27B with DFlash2 on an oQ4e checkpoint went from 33 to 131-160 tok/s. Before M5, a lone stream's copy
  windows widen to 128 rows (an M3 Ultra edit ran 190 -> 233 tok/s), and Flash Next's concurrent rounds use the matrix
  units. GLM-5.3's DFlash2 reads only its sliding window: 8% faster at 52k, and 3.8 GiB lighter a rank on two Sparks.
- **Operations.** Prometheus `/metrics` on both servers, `TENSORFOLD_MEMORY_RESERVE_GIB` for the CUDA startup
  reserve, `--checkpoint-slots` for the 27B's concurrent decoder on CUDA, and an idle GLM rank no longer spins.
- **Apache-2.0.** TensorFold is licensed under the Apache License 2.0 from this release, which adds an explicit patent
  grant from contributors. Releases up to 0.5.0 stay MIT, and code written before 0.6.0 keeps its MIT notice in
  `LICENSES/MIT.txt`.
- **Community pull requests.**
  - Resumed resends and thinking turns on Flash Next (#124). Thanks to @benthecarman, and to @Arminova for the
    GB10 measurements and the third-resend test.
  - Streamed tool-call arguments on CUDA (#114), `--alias` on CUDA (#111) and Python-spelled tool parameters
    (#135). Thanks to @olexale, @philip-pentatonic and @outcastofmusic.
  - GLM-5.3 on Macs: image input (#101), 8-bit and Q8_0 GGUF checkpoints (#119), float32 activations (#118) and
    the backbone wiring notes (#120). Thanks to @mgoldwasser and @feni6.
  - GLM-5.3 on two Sparks: lighter prompt buffers, the EXL3 estimate, the MTP setting, the idle rank and DFlash2's
    ring (#128, #129, #131, #132, #134), visible-pool selection (#140), and the CUDA startup reserve setting (#133).
    Thanks to @MiaAI-Lab and @mikolaj92.
  - Flash Next: scaled n-gram tables (#148, reported in #142) and the M5 draft head fix (#147), consolidated EXL3
    tables (#145), block-scaled FP8 (#126) and the NVFP4 expert speedups (#102, #105). Thanks to @gilby,
    @cwschroeder, @shantanugoel, @jschmied and @tournierjc.
  - On M5 Macs the 27B fuses the projections of 4-bit group-32 checkpoints such as oQ4e too (#164): four streams went
    from 315-320 to 332-334 tok/s on an M5 Ultra, with the same bits. Thanks to @gilby.
  - `--checkpoint-slots` (#125), and the DFlash2 drafter's admitted size (#112, from @jkuepker's ROCm work). Thanks
    to @nood-co1 and @jkuepker.

## 0.5.0 (29 Sep 2026)

- **OpenAI's Responses API on both servers.** `/v1/responses` runs as a chat completion: every response has its chat
  completion's token SHA and prompt tokens, and a tool round trip gives the same reply through `previous_response_id`,
  the turn resent whole, or chat. A second turn resumes from the cache.
- **Long context on Sparks, the same bits as 0.3.6.3.** The 27B's attention reads each key chunk once for a round's
  rows, with 16-byte loads, and prompt attention runs in its own CUDA kernel. On one Spark (MLX 4-bit + DFlash2):
  decode at 128k 18.4 -> 38.9 tok/s, at 96k 23.6 -> 45.9; cold prefill at 128k 860 -> 1,173 tok/s and at 255k
  545 -> 809. On the same NVFP4 weights, prefill runs 1.16-1.27x vLLM at every depth from 32k to 255k.
- **The CUDA server reads requests as the Mac server does.** `reasoning_effort` and `thinking_budget`, usage with the
  cached prompt tokens in every reply (streamed ones too), typed tool arguments, `min_p` on both backends,
  `ignore_eos` on Flash Next and Nemotron, and `top_k` 0 drawing the whole nucleus on two GPUs as on one. Thinking
  uses the template's own default effort on both servers.
- **`priority: background` on CUDA.** A background request yields to foreground ones and replays from its prompt,
  so its reply still equals its solo run. Under `--parallel`, the 27B and Qwen3.6 prefill a background prompt 1,024
  rows a step (measured waits in docs/api.md).
- **Community pull requests.**
  - Faster CUDA startup and model loading (#82). Thanks to @pmeenan.
  - Qwen3.6-35B-A3B serves `--parallel N` on CUDA, each reply equal to its solo run (#84), and `response_format`
    JSON schemas on the 27B, exact under drafts (#80). Thanks to @philip-pentatonic.
  - `/health` publishes live token totals (#79). Thanks to @MiaAI-Lab.
  - The 27B's concurrent drafter keeps no stale context after a prefill step (#92). Thanks to @nood-co1.
  - Flash Next lists its sparse-attention blocks past 131,072 keys in tiles, the same lists (#93). Thanks to
    @MovieMaker93.

## 0.4.0 (29 Sep 2026)

- **oQ formats and MLX 8-bit at 4-bit speed on M1-M4.** 5-, 6- and 8-bit linears now run on the matrix units with the
  same exactness. Qwen3.8-27B at oQ4e has half the KL against bf16 of MLX's 4-bit for 6% more bytes. On an M3 Ultra
  it now verifies 8-row windows in 31.9 ms instead of 40.8, within 1-5% of the 4-bit at every width. Served with
  DFlash2 drafts, sampled code runs 14% faster, chat 5-7%, and greedy code is level. The 5- and 6-bit layers use the
  matrix chain's arithmetic now, so replies can differ from 0.3.x's by a token here and there, and still equal their
  own `"draft": false` replies.
- **More streams on Macs.** A stream holds memory for its next 2,048 tokens, not its whole reply. A shared round's
  working memory is charged to the streams that share it. On a 64 GB Mac the 27B serves 16 streams at 32k contexts,
  where 0.3.x served 4-9. When memory runs short, kept prompts go first, then the newest streams wait, then the
  newest ends with an error that names `--parallel`.
- **Prompts inside the promised window are admitted and kept (#95).**
  - A finished request's rollback rows are released before the next prompt is sized.
  - The startup line counts the probe round once: "0 streams of 8,192 tokens fit" became 13 with the reporter's
    flags.
  - The same flags give the same window on every start.

  Thanks to @benwilson.
- **Bonsai on Macs short of memory for the full widening** now widens as many layers as fit (60 of 64 at 25.2 GiB).
  Code runs 55 -> 78 tok/s on an M3 Ultra at the #94 reporter's budget. Thanks to @MESevenJourney.
- **Flash Next on M1-M4** runs 2-7% faster, drafted and serial, on an M3 Ultra.
  - The attention gate runs in the merge kernel, the PLE layer and router are fused, and chained drafts queue as
    they're built.
  - From two rows, the 4-bit dots skip the int-to-float convert (on M3 and M4 through half-precision nibbles), so
    windows of 3-16 rows cost 3-4% less with the same bits.
- **Flash Next's 2-8-bit and mixed checkpoints on Macs.** oQ4, oQ4e and oQ5e, and MLX 6- and 8-bit checkpoints,
  load and serve, each module in its checkpoint's own format. Before, a mixed checkpoint with a 4-bit base passed
  `tensorfold info` and then failed to load. 4-bit weights in groups of 32 keep their kernels. Other widths run new
  kernels whose rows keep their bits at any row count, so drafted replies still equal `"draft": false`.
- **GLM-5.3 oQ4 loads as downloaded.** Its config lists 46 `mlp_layer_types` for 45 layers (the MTP layer too), and
  transformers refused it, so the tokenizer didn't load. TensorFold now reads it as is.
- **A live line under `tensorfold serve` on Macs.** In a terminal, one line under the log shows open connections and
  decode and prefill tok/s, redrawn in place. Log files see nothing new, and `TENSORFOLD_NO_LIVE=1` turns it off.
- **MLX 0.32.3 (#88).**
  - TensorFold runs on MLX 0.32.2 and 0.32.3, and a fresh install gets 0.32.3. Output is the same bit for bit on
    both.
  - SSD expert streaming builds its extension with the nanobind each MLX was built with (the `ssd` extra brings it).
  - A prompt kernel that doesn't build prints a warning naming the MLX version and the fix, and
    `TF_REQUIRE_KERNELS=1` stops at startup instead.
- **DeepSeek-V4-Flash draft heads on Hugging Face.** `Vontra/DeepSeek-V4-Flash-DSpark-MLX` (the default once pulled)
  and `Vontra/DeepSeek-V4-Flash-MTP-MLX`, converted from DeepSeek's MIT releases. A head is a folder holding
  `model.safetensors` and a `config.json` that names it.
- **Fixes.**
  - Object and array tool arguments that the model leaves one closing bracket short are closed, and kept if they
    then match the schema (#87).
  - Qwen3.5 and Qwen3.6 chat templates get the same resume points as Qwen3.8, so a second turn reuses the first
    (#83). Thanks to @philip-pentatonic.
  - A prompt or image that can't fit even alone no longer evicts kept prompts first.
- **Correction to 0.3.6.3's notes.** Nemotron's chat decode on the M3 Ultra wasn't 2% slower. It spreads about
  ±2.5% between server starts, even at a fixed draft depth.

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
- GLM-5.3-Flash reads Brandon M. Music's EXL3/TR3 weights (re-hosted by Mia-AiLab) on two DGX Sparks (experimental).

## 0.3.1 (26 Sep 2026)

- `tensorfold info` shows how a checkpoint stores its weights and which backends read them. `serve` and `pull` refuse
  checkpoints no engine reads yet, before anything downloads.

## 0.3.0 (26 Sep 2026)

- NVIDIA GPUs: `tensorfold serve` picks CUDA on Linux and runs on one GPU or two (one rank per DGX Spark).
- A new family, GLM-5.3-Flash, on two Sparks.

## 0.2.0 (25 Sep 2026)

- A rewrite: `tensorfold serve`, `pull`, `models` and `info` for Nemotron 3.5 Lightning, Qwen3.8-27B and Qwen3.8
  Flash Next on Apple Silicon, with drafts that never change the output.
