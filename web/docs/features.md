# TensorFold Web: what it does today

A short map of TensorFold Web as of 14 October 2026, by screen, with the module behind each part and what is still open.
The engine questions and bug reports are in `questions-for-ash.md`.

## Two layers, and Home

Every screen has a plain layer and a detail layer (`docs/design/2026-10-11-studio-for-everyone.md`, 11 Oct). The
plain layer names things as a person knows them: the AI by its catalog name (`/api/engine` carries `model_name`),
"tokens a second" with the words a second beside it, conversations in pages, the speed-up rather than drafts. The
engine's own figures (KV cache, prompt cache, drafts kept, the command line) sit under a disclosure ("The engine's
own figures", "Details for the curious") on Monitor, Fleet, Models and Teach; Settings keeps the basics first and
the rest under Advanced; the Bench opens on one check with "More checks" for the eight kinds of run; Chat keeps its
thinking, presets, tools and parallel panel behind one Options button.

**Home** (`Home.tsx`) is the default screen (six tiles since 13 Oct: Chat, Teach, Connect, Bench, Monitor, Fleet; the Teach tile and the "What it has
learned" card stay when the AI running cannot be taught, greyed, saying why: another engine, a machine whose engine does
not learn, a family TensorFold does not learn into; `teachable.ts`; the machine line gives the AIs its memory holds and,
apart, those its engine serves): the state in one sentence ("Nemotron 3.5 Lightning is ready and idle", a Check other models button beside it (Chat has its tile; 13 Oct), with a second line for the hardware under it:
the maker's logo (Apple, NVIDIA, AMD, Intel, Qualcomm), then the machine, its chip without the board in brackets and
its memory; the status dot stays level with the model's line; a remote engine's machine as its Fleet
entry describes it, else by its host; one logo per maker when an engine spans several machines; 13 Oct),
"Nothing is running, Nemotron is on this machine, ready to start", "No AI is installed yet, Nemotron fits this
machine: 17 GB to download") with the one button for that state, then Chat, Teach it something, Use it in an app,
the last conversations, the teachable copies and what they remember, and the first-run list until the machine is
ready. The catalog marks one model `recommended` per machine (runs here and fits, the fastest first, one on disk
before one to download). The sections sit in the order a person uses them: Home, Models, Chat, Teach, Connect,
Bench, Monitor, Fleet, Settings (keys 1 to 9); the old addresses keep working. The sidebar folds to its icons (its
button at the foot of the menu, or the `[` key, 13 Oct), each name kept as a tooltip and for screen readers, the AI's
state as its dot alone; the browser remembers it.

**Logos** (12 Oct; 13 Oct: Meta, Mistral AI, Moonshot AI, MiniMax, Hugging Face, Baidu, ByteDance, Xiaomi added, and a
maker known without a logo file shows its initial): a model shows its family's or its maker's logo wherever it is named (Home, Models, Chat, Fleet,
the sidebar): NVIDIA for Nemotron and its teachable copies, Qwen, Google for Gemma, Z.ai for GLM, DeepSeek, Prism ML
for Bonsai. `ui/src/brand.ts` reads the maker from any name a model goes by (catalog, repository, copy, another
machine's); Connect shows each tool's logo. Both sit on a light tile so dark marks read in the dark theme; each
logo's source is in `ui/src/assets/models/README.md` and `ui/src/assets/logos/README.md`.

## Models
- Catalog of models with a memory verdict before download: fits, fits tight, does not fit (`fit.py`, `catalog.py`).
  A downloaded native model is judged on the engine's own `tensorfold info` floor. Each row says three things apart
  (12 Oct): the memory verdict ("Fits", "Fits, tight", "Needs 192 GB"), "Not on this engine" with the reason when
  the engine here cannot run it (an MLX model on a Spark), and "Loaded", "On disk" or "Not downloaded"; on a phone
  this column is the one shown.
- Download with progress (`pull.py`: its own process, a Stop that keeps what came through for the next Download to
  resume, refused before it starts when the disk cannot hold it; 13 Oct); start and stop; which engine serves each model, native 1.0 or the frozen
  Python line (`engines.py`, `native.py`).
- Starting a model while another runs asks in place: "X is running. Stop it, then start Y?".
- The 27B's mixed-width checkpoints (oMLX's oQ4 and oQ2, 14 Oct): native on Macs from TensorFold 1.0.10, which reads
  each module at its own width. `capabilities --json` lists each family's uniform formats only, so their entries say
  `native_from` instead of a format; a chip that refuses the 27B (the M6) still says so after one try.
- Which binary serves (14 Oct): `TENSORFOLD_BIN` when set, else the newest release among PATH, the official
  installer's `~/.tensorfold/bin` (`curl -fsSL https://tensorfold.dev/install.sh | sh`), Homebrew's and the
  archives' places, so an installer release beside an older Homebrew one is the one used.
- A native model on a Mac is judged by the native engine's own sizing (macOS's GPU limit, measured overhead): Nemotron
  "fits, 262,144 tokens" on 32 GB, as served.

## Monitor
- Live writing and prompt-reading speed, connections, KV cache fill, requests set aside, drafts kept, admitted
  context (`engine.py`); the prompt cache and the drafts kept beside memory at the top (13 Oct), the rest under "The
  engine's own figures". A figure an engine does not report says so; a request still running behind a key shows as
  running, not as 0 tok/s.
- The prompt cache (TensorFold 1.0.5): its budget and the share of prompt tokens restored from kept states; the
  prompt-reading speed counts only the tokens actually read.
- Usage kept for months (`usage.py`): tokens written per day and per hour by model, the daily peak writing and
  prompt-reading speeds on their own scales, totals by model (written, read, restored, requests), "idle, last
  served 12 min ago"; counted from the engine's own totals, across engine and TensorFold Web restarts.
- History kept by the server for eight hours, drawn at once, 30 min / 1 h / 8 h windows, gaps as gaps (`history.py`).
- Memory on this machine by holder (engine, everything else, free), the headroom in GB turning amber then red.
- Health findings, each with its evidence and what to try: memory headroom, macOS memory pressure and thermal state,
  GPU slowdowns on NVIDIA, requests set aside, full KV cache, queued requests, a prompt cache without memory, failed engine, refused key
  (`health.py`).
- The engine log, progress bars on one line. The engine outlives TensorFold Web: a restarted TensorFold Web finds it again
  (`engine.json`, log file), so a TensorFold Web update never stops the model.
- Open: jetsam events and the NVIDIA and AMD out-of-memory journals (need a Spark or an AMD machine).

## Chat

The list of conversations (`ChatList.tsx`, 12 Oct): a Pinned group on top, then the rest by date (Today,
Yesterday, This week, This month, Earlier), by folder or by model (the choice kept in the browser). Each
conversation has a "⋯" menu: pin to the top, rename, move to a folder (or to a new one), delete, each asked in the
page. Folders are made, renamed and deleted from the list grouped by folder; deleting one keeps its conversations.
A pin, a folder or a new title keeps the conversation's time and revision. The open conversation's own actions sit
above it: Verify speed-up & exactness (13 Oct: its name says it proves the replies unchanged), Replay, Compare with a
replay, Export (Markdown to read, JSON for all the data). While a reply is written, the first-token time and a live
speed (from the text received, at the last reply's characters per token: a stream event carries one verification
round, several tokens with drafts) show in the pills, the engine's exact figures replacing them
at the end, with a third pill for the drafts kept and the tokens per round; the reply bar adds the prompt's reading
speed (its tokens not restored from the cache over the engine's `prefill_seconds`). With Cortheon on (Settings), the
Options say "Web research" (off sends `"cortheon": false`), and a reply shows its searches, pages read and sources.
"Teach this" shows on a model TensorFold learns into as well as on a copy: on the base, Teach keeps the message and
says to make or serve a copy first.
Compare with a replay lists only the conversations that ask the same questions (its original and every replay of
it), each named by what wrote it ("Qwen3.8-27B-MLX-4bit, without the speed-up", "the original, …"), and the
comparison names its two sides the same way.
- Conversations kept by TensorFold Web (`conversations.py`): listed, searched, renamed, exported to Markdown or JSON;
  each reply with its model, engine, thinking setting, preset and figures; an interrupted reply said so.
- The last reply's time to the first token and its speed in two coloured pills at the top of the chat (green and
  accent), updated after each reply; the full figures stay under the reply.
- A thinking block that opens while the model thinks and folds when it answers; smooth streaming text; presets in
  three panels (`ChatPresets.tsx`): Instructions (the system prompt), Length of the reply (reply up to N tokens, up to
  four stops) and Randomness (temperature, then top_p, top_k, min_p and seed folded as finer control), each setting
  explained in a line; with no seed in the preset the page draws one per reply and keeps it with the reply (TensorFold
  derives an unseeded request's seed from its prompt, so asking again would give the same answer); the preset's actions in a bar below. Start the last model when nothing runs. A person's presets are kept on TensorFold Web
  (`/api/chat-presets`), so every browser sees them; those a browser held before move there on the next visit.
  Saving under a name already kept asks before replacing it; one's own preset can be saved again, renamed or
  deleted. Each reply records the sampling it was written with; a replay, greedy, keeps only the stops.
- Replay a conversation on the served engine with drafts on or off, or on another engine, and compare two
  conversations question by question (`replay.py`).
- A workshop for tool calls and JSON (`ChatWorkshop.tsx`), in two panels like the presets (Tools the model may call,
  Reply format), each setting with its sentence and the definitions folded: tools in OpenAI's format (calculator, time and random
  number run in the page and chain by themselves; others are answered by the user), each call checked against its
  tool's parameters; replies held to JSON or a JSON schema, checked, asked in the prompt when the engine refuses
  `response_format`.
- A parallel mode (`ChatParallel.tsx`, in Bench under More checks since 12 Oct; Chat keeps to conversations): 1 to 16 streamed requests at once, a card per reply, and a table per count
  tried (total speed, each reply's speed, first token from sending, gain against one request). Open: images.

## Fleet (`fleet.py`, `Fleet.tsx`)
- This machine and the others in one view, each read every 5 s from its own TensorFold Web with its token; the tokens stay
  in this TensorFold Web (`fleet.json`, readable by this user only), never in the page. Per machine: what it serves, speed
  now, connections, KV cache, context, prompt cache, memory by holder, warnings, versions, how long the read took.
- Add a machine without a restart: a Test lists seven checks (answers, is a TensorFold Web, token, another machine, same
  TensorFold Web version, same TensorFold version, its engine) as passed, warned, failed or skipped; the machine is added
  only when none fails. Change its token, remove it, open its TensorFold Web. Each card says when the machine last served.
- Another TensorFold release than this machine's (14 Oct) is a warning on the Test, on the machine's card and in the
  Bench's KL when that machine is the reference: since 1.0.10 Nemotron on CUDA can part from 1.0.9 at a near tie, so
  a comparison across two releases measures the release as well as the machine.
- Open: speed-up groups as one unit, power (wake-on-LAN, shutdown).

## Bench (`bench.py`, `report.py`, `compare.py`, `series.py`, `suites.py`, `quality.py`, `power.py`)
- **Kinds of run**, one tile each in a soft violet tint, the chosen one stronger, with a sign and a line on what it measures (13 Oct): quick score (3 prompts), usage profile (7 suites, 14 prompts), quality (228 items), tool calls
  (tool-eval-bench), prefill sweep (1k to the served window), load sweep (1 to 32 requests at once), KL divergence, compare
  configurations (2 to 4).
- **Protocol**, printed in every receipt: temperature 0, one request at a time, warm-up, thinking turned off and
  checked (six explicit ways, the one kept named), output format asked in the prompt.
- **Exactness**: each prompt sent again with `"draft": false` and compared by token digest.
- **Hardware**: power and temperatures without root (Mac SMC; `nvidia-smi`, `amd-smi`), energy per token above
  idle, coverage of the readings, cost at a price set in Settings, whether the machine slowed itself down.
- **Suites**: built-in ones versioned by a digest, fixed so that runs compare; your own suites, written, imported or
  copied from a built-in one to change its prompts (13 Oct), a prompt with earlier turns of a conversation if wanted,
  exported as JSON; chosen in a usage profile, a load sweep or a KL divergence (13 Oct: the last two also).
- **See it live** (with Load sweep only, 13 Oct): several requests at once, each reply streaming in its own card; nothing
  kept.
- **Live card**: phase, elapsed time, time left, a stop that asks once.
- **Report**: plain findings, a self-contained HTML report with the receipt inside, a share card (text, or an image
  copied or downloaded), the JSON receipt.
- **Compare two runs**: prompts paired one by one, a sign test on speed and an exact McNemar test on exactness,
  "within noise" or not, refused across different suites; side by side with a swap; an HTML report.
- **Another engine**: any OpenAI-compatible address (Ollama, mlx-lm, vLLM, llama.cpp…), same protocol, prefill
  estimated where the engine gives none.
- **Compare configurations**: model, context, memory budget, drafts; the engine restarted between them and put back
  at the end; a matrix against the first; speed or quality.
- **Quality**: 100 GSM8K problems and 100 MMLU questions (fixed seed, MIT, licences shipped), 20 instruction checks
  in code, 8 long texts whose fact is corrected later; scored strictly (the last value counts, two answers fail, a
  reply cut at its limit said apart); compared item by item with McNemar's test.
- **Load sweep**: waves of requests at once; total and per-request speed, first token p50 and p95 with the wait.
- **KL divergence**: top-20 log-probabilities against a reference (drafts off, or another engine at an address, or a
  Fleet machine), position by position where both wrote the same text, optionally under load (`kl.py`); mean, p99 and
  largest KL, same top token, identical replies, the largest divergences. Or a fixed text (13 Oct): both engines read
  it once through `prompt_logprobs` on `/v1/completions` (TensorFold 1.0.8 on CUDA, vLLM; refused on Metal and the
  Python line, said plainly) and every position is compared, however the replies would have parted.
- **Fleet machines as targets** (13 Oct): Bench's "another engine" and KL's reference offer the Fleet's machines
  beside an address typed by hand; TensorFold Web reaches the machine's engine through its TensorFold Web's relay, with that
  TensorFold Web's token or the engine's key, neither of which reaches the page or the run.
- **Tool calls**: tool-eval-bench (MIT), run by TensorFold Web on the target engine, 15 or 69 scenarios scored pass,
  partial or fail, compared scenario by scenario like the quality suite (`tooleval.py`); where it is missing, an Install
  button runs `uv tool install tool-eval-bench==2.8.0` for the user (13 Oct; installed on both Sparks that day).

## Teach: Living Weights (`living.py`, `Teach.tsx`)

TensorFold 1.0.7 teaches a served Nemotron facts into its own weights (`serve <folder> --slide`). TensorFold Web:

- **Living copies.** A teachable copy of a downloaded model, as real files under `~/tensorfold-living` (an APFS clone
  on a Mac, a checked 18 GB copy on Linux, refused if a link remains), served from Teach with `--slide` and its own
  fact graph file; where learning cannot run (no 1.0.7, a GPU other than a GB10) it is served without, and says so.
- **Preview before learning.** The copy lists the facts a text holds (JSON, constrained), each checked against the text:
  a claim the text does not make is flagged and left out unless ticked. Each fact shows its subject, the question that
  should get it back (editable) and what the model answers now; a fact about the user keeps the sentence they wrote,
  in the first person, as TensorFold learns it.
- **Lessons.** Each approved fact is its own learn request, its question asked before and after; kept or not, with
  the engine's reason (read from its log when the event gives none). The engine is not restarted mid-lesson (the
  gate), chat goes on meanwhile, and a lesson can stop after the fact being learned.
- **Versions** (`living_versions.py`): kept before each lesson and on demand; each file by content (a file equal to
  the downloaded checkpoint's is the base's, others stored once and shared by versions). Restore (the only way to
  take a learned fact out), branch into a new copy, forget a version. A restore adds the facts it took out to the
  checks, so the next run shows they are gone.
- **Checks** (`living_checks.py`): the questions of the facts kept, near misses and the user's own, eight unrelated
  ones, asked with thinking off and on, before and after each lesson, after a restart or on demand; two runs compared
  question by question, recall, empty replies and the leak matrix.
- **What learning costs** (phase C): Bench runs on a copy carry the facts it held; a quick score, the quality suite
  and KL against the untaught model served by another machine of the fleet (through its TensorFold Web's relay), listed per
  copy. "Teach this" in Chat hands a message to Teach. Measured on node 2: 7 facts cost 7% decode speed and moved
  the distributions slightly on unrelated text (mean KL 0.0032, same top token 99.3%), where two untaught engines
  give 0.
- **Forget, move, describe** (phase D): forget one fact TensorFold Web taught (back to the version before its lesson,
  the facts kept since taught again); fetch a copy taught on another machine of the fleet (only files that differ
  from the base travel, each checked by its hash); a model card (origin, facts and sources, lessons with answers
  before and after, versions, measurements, and the warning that taught facts live in the weights).
- **3D view** (`/living-view/`): TensorFold's own `/slide` page (the model's layers, each fact's neurons, the lesson
  live), served through TensorFold Web with its paths turned to TensorFold Web's and the engine's key added by TensorFold Web;
  opened from Teach with the token, which becomes a cookie for that path only and leaves the address.
- **Record and graph.** What each copy learned (the engine's graph: state, neurons, questions) and what TensorFold Web
  sent, kept beside the copy.
- **Who can teach and read.** Only through TensorFold Web's token: the relay refuses every route of the learner (the
  graph included), judged on the decoded path; a copy that learns binds to loopback; a copy's TensorFold Web data is private
  (`.studio/<name>/`, 0700); the 3D view's cookie comes from an authenticated POST. Codex reviewed Teach on 11 Oct;
  every finding was fixed.

## Connect and Settings
- Connect in three steps: where the app runs (this machine, or another computer through TensorFold Web's address
  with its key), which app, then its settings written for that place, every line included; the key hidden on screen,
  copied in full. An app not in the list gets the address and key for the same place; answering on the local
  network without TensorFold Web sits under Advanced. A coding agent whose prompts need a longer window than the one
  served says so, with a button to serve 65,536 tokens when the model can hold it, or the models that can.
- Settings: memory budget (a slider only where it has a range, the Python line on a Mac; elsewhere the figure TensorFold
  sizes itself, 13 Oct), context, thinking, appearance (only here since 13 Oct, no longer in the sidebar), electricity price, extra `serve` flags checked against the
  engine's own list (unknown, abbreviated, set by TensorFold Web, or secret: refused). Web research (Cortheon, TensorFold
  1.0.9, 13 Oct): off by default, shown when the engine offers `--cortheon`, with what leaves the machine (the search
  queries to DuckDuckGo, page reads from this machine's address, the risk of a page steering the model) and a warning
  when the engine is shared on the network, or when it predates 1.0.11, Cortheon's security release (14 Oct).
- Settings, Makers and their logos (13 Oct, `brands.py`, `SettingsBrands.tsx`): the model and hardware makers the
  TensorFold Web recognises, each with its logo and the words that recognise it; a person adds a maker (name, words, logo:
  PNG, JPEG, WebP or SVG up to 150 KB, an SVG with a script refused), gives a known one a logo, or undoes that. Kept on
  this TensorFold Web in `brands.json`, checked before the page's own list, so a new family needs no new TensorFold Web.
- Recipes for fifteen tools by kind (`recipes.py`): chat front ends (Open WebUI, LibreChat, AnythingLLM, Jan),
  coding agents (Claude Code, OpenCode, Codex CLI, Pi, OMP, OpenFox, Cline, and Cursor, which says why it cannot
  reach a private AI), personal agents (OpenClaw, Hermes Agent) and Python. Each has a Test connection that makes
  the tool's own calls (models, streaming, tool calls, Anthropic Messages, Responses, the window it needs) and the
  record of where it was checked with the real tool; an Open WebUI compose file. Continue left the list when Cursor
  bought it (June 2026) and stopped maintaining it.

## Setup and diagnostic (`diagnostic.py`, `SetupChecks.tsx`)
- Ten checks, each with how to fix it: graphics backend (Metal, CUDA, an AMD GPU waiting on ROCm), TensorFold
  installed (TensorFold's own installer, `curl -fsSL https://tensorfold.dev/install.sh | sh`, on a Mac and on NVIDIA
  Linux; 14 Oct), engine port, model folder and its free space, a model downloaded, engine running, memory headroom,
  who can reach TensorFold Web, whether it keeps running, and web research on an engine before 1.0.11 (14 Oct).
- A getting-started list on Home until every step is ready, or put away; Settings keeps all the checks, a JSON
  report to send (keys and tokens taken out; machine names and paths stay) and a summary to copy.

## Storage (`storage.py`, `SettingsStorage.tsx`)
- What TensorFold Web keeps, with count, size and age: Bench runs, conversations, usage history, the Monitor's readings;
  each cleaned older than 30 or 90 days or whole, counted before it asks; a running Bench run is kept, and so is a
  pinned conversation (13 Oct).
- The catalog's models on disk with their size, removed through huggingface_hub's own deletion; the served model,
  its drafter and a download in progress are kept. Other models in the same cache are not listed.

## Links, palette, app
- Every view a link: `#models/<repo>`, `#bench/<run>`, `#chat/<conversation>`; the address follows the selection
  without filling the browser's history.
- The palette (⌘K or Ctrl+K) forgives: pieces, letters in order ("bnch"), one typo ("nemtoron"), accents ignored;
  once something is typed it offers every model, the last 40 Bench runs and 60 conversations.
- Installable as an app (manifest, icons, shortcuts) where served over HTTPS: `https://<machine>.<tailnet>.ts.net:8443`.

## Accessibility
- WCAG 2.1 AA checked with axe-core on every screen in both themes; a skip link, focus moved to the content when
  the screen changes, a tab title per screen, keyboard use throughout.

## Running as a service (`service.py`)
- `tensorfold-web service install`: a LaunchAgent on macOS, a systemd user unit on Linux; `--check` says what
  it would write, `status` and `uninstall` too. The engine outlives TensorFold Web's restarts either way.

## Safety
- Loopback by default, a token required on any other address; without a token only this machine's names are
  answered (DNS rebinding refused); changes asked by another site's page refused; who can reach TensorFold Web and the
  engine shown in the sidebar; the page greys out when TensorFold Web stops answering.
- Getting in (`access.py`): `tensorfold-web token` shows the token on TensorFold Web's machine (`--copy`, `--new`); a
  page asks for it once, in a dialog that says where it is; Settings makes a one-time link to copy, and its QR code for
  a phone (good once, five minutes, traded for the token by the page that opens it, the code out of the address at once); Fleet
  opens another TensorFold Web signed in through such a link, asked with the token this TensorFold Web keeps for it.

## Measured so far (Mac mini M6, 32 GB)
- Nemotron 3.5 Lightning, native TensorFold 1.0.2: 102.7 tok/s with drafts, 72.2 without, 14 of 14 identical,
  0.33 J per token; prefill about 1,550 tok/s up to 16k, 945 at 128k.
- Same model on other engines: mlx-lm 65.9 tok/s (same weights), Ollama 59.7 (GGUF) and 52.1 (MLX build).
  Quality on the same 228 items: TensorFold 192, mlx-lm 196, Ollama 198, within noise of one another.
- The 27B on the Python line: 4-bit 44.3, oQ4e 46.9, Bonsai 42.6 tok/s with drafts (within noise of each other);
  without drafts Bonsai +36%. Quality: 4-bit 202, oQ4e 206, Bonsai 193 of 228 (Bonsai loses on
  MMLU only).
- Several at once: Nemotron native 89 tok/s alone, 136 at 8 at once, 140 at 16; Qwen3.8-27B 31 to 89.
- Tool calls (tool-eval-bench, 69 scenarios, thinking off): Qwen3.8-27B 90/100; Nemotron 57 (TensorFold) and 60
  (mlx-lm, same weights): the model, not the engine.

- On the DGX Spark (GX10, TensorFold 1.0.4): Nemotron 160.3 tok/s with drafts, prefill 7,000 tok/s (4,061 at
  128k), 135 to 218 tok/s at 1 to 32 at once, quality 193 of 228, as the Mac's within noise.

## Not yet done
- Step 1: the NVIDIA kernel journal and amdgpu errors (need root or an AMD machine).
- Step 3: KL at every position on Metal (needs `prompt_logprobs` there); KL against an unquantized reference run
  for real.
- Step 4: Codex itself not run yet; its request is answered from TensorFold 1.0.9 (13 Oct), refused before.
- Step 5: images, once a vision model is served natively.
- Step 6: power (needs root on the Sparks) and speed-up groups (needs TensorFold's speed-up mode on our machines);
  the fleet view, adding a machine, the usage history and the service are done.
- Step 7: publishing (PyPI, Homebrew, `uvx`) waits on Stéphane and Ash; everything else in it is done. Optional: a
  community score by machine.
