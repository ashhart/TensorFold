# Questions, requests and bug reports for TensorFold

What TensorFold Web would like from the engine, checked against TensorFold 1.0.7 (11 to 13 October 2026) and 1.0.9 (13
October). Items are numbered for reference (A1, B2…); the evidence for each (requests, logs, measurements) is kept and can
be sent on request, or filed as GitHub issues if that suits you better.

## A. Required to finish TensorFold Web (publishing waits on these)

| # | Question |
|---|---|
| A1 | **Name and brand.** May we publish it as "TensorFold Web" and use TensorFold's logo and colors? If not, which name? |
| A2 | **Where it lives.** Answered: the web view on top of the installable CLI. To agree: the whole TensorFold Web (its Python server and page) or only the page served by the native binary, and where it goes in the repository. |

## B. Important, not blocking (they shape what TensorFold Web can do)

| # | Question |
|---|---|
| B2 | **A separate right to teach.** Anyone holding the engine's API key can call `/v1/slide/learn` or clear the graph. A second key (or `--slide-key`) would let a server give chat to tools without giving them the weights. TensorFold Web guards its own side meanwhile |
| B4 | **Claude Code on CUDA.** The GB10 admits a 16,384-token window by default; Claude Code's first request needs more and gets a 400. Could the CUDA default follow the memory left, as Metal's does, or the startup line say how to raise it? |
| B5 | **Bonsai 2 and Gemma 4.** In what order and roughly when? TensorFold Web drops its Python line once they are native |
| B6 | **Log-probabilities on Metal**, and `prompt_logprobs` or `continue_final_message`, to measure a quantization's drift on a Mac and at every position (1.0.8 brought them on CUDA: streamed `logprobs` and `prompt_logprobs` on `/v1/completions`; Metal still to come) |
| B7 | **Two DGX Spark paired**, as two Macs are in speed-up mode, or is it Mac-only by design? |

## C. Improvements (useful, not needed to finish)

| # | Question |
|---|---|
| C1 | `--json` for `models`, `info` and `pull` (TensorFold Web reads `info`'s text today) |
| C2 | Family, backend, drafts state, and a speed-up pair's rank and peer in `/health` (the window is in `/v1/models` since 1.0.3) |
| C3 | Admission before loading: what would this machine admit with this checkpoint and `--context N`, without starting it |
| C4 | The `tensorfold` block on `/v1/messages`, streamed or not (chat has it) |
| C5 | Living Weights: a dry run of the fact finder (`"dry_run": true` returning the facts it would learn) |
| C6 | Living Weights: the subject, kind, questions kept, each round's outcome and the reason in the `learned` event |
| C7 | Living Weights: a save that writes only the shards it touched, or says which (each lesson rewrites 18.5 GB on a GB10) |
| C8 | Living Weights: a route to cancel a running lesson |
| C9 | Living Weights: `--slide` on the M6, and which models learn after Nemotron. On a GB10, Nemotron with `--slide` holds the model (17.99 GiB) and 25.54 GiB for the trainer (raw weights 18.3 GB, buffers 4.1 GB): about 43.5 GiB before any context, beyond the 32 GB M6's 28 GiB GPU limit. Does the Metal trainer need the same, or could it train on the served weights? (12 Oct: not tried on the Mac for that reason) |
| C10 | Living Weights: is a mean KL of 0.0032 on unrelated text with seven facts the expected cost, and how many facts does a pocket hold? |
| C11 | **The served model's family and maker, by name**, in `/v1/models` and `/health` (for instance `"family": "Nemotron", "maker": "NVIDIA"`). TensorFold Web shows the maker's logo beside the model, and today it guesses the maker from the model's name with its own list: a family that a new TensorFold release brings shows no maker until a new TensorFold Web, or until the person adds it in Settings. The engine knows the family as soon as it can load it. With it, for a model served across several machines (C2's speed-up pair), each machine's hostname and chip, so that Home can name them all with their makers' logos. Names only: TensorFold Web keeps the logos, Ash need not ship any |

## D. Bugs reported

All checked on 1.0.7.

| # | Bug |
|---|---|
| D1 | **A short `thinking_budget` leaks thinking into the answer.** At 200, `reasoning_tokens` stops at 201, then the model goes on reasoning in `content` (5,861 characters for a haiku) and writes a literal `</think>`. Clean at 1,000. The field is also missing from the `tensorfold` block (`thinking_budget: null`) |
| D2 | **Docs and server disagree on the draft counters.** `docs/api.md` puts `rounds`, `drafted`, `accepted`, `acceptance_rate`, `tokens_per_round` in the `tensorfold` block; the server sends them in a `speculative` object beside it |
| D3 | **No memory in `/health` on Metal.** `"memory": {}` on a Mac (only CUDA fills it); `/memory` and `/metrics` have the footprint |
| D4 | **`tensorfold models`** labels the 27B checkpoints `Qwen3.5-2B or Qwen3.8-27B (qwen3_5)`: it could tell them apart by geometry, as the loader does |
| D6 | **A copy broke after a lesson kept in 6 rounds** (11 Oct, GB10, 1.0.7). After eight lessons in a row (seven taken out, each still logging "the learned neurons are in the model's shards (250 MB written)"), "My favorite food is steak tartar" was kept at round 6. From then on the copy degenerated within a few paragraphs, greedy too ("bestbonk", then ". . . **. He . . ." and "... ... ..."), and an earlier fact came back wrong ("Your favorite sport is flying the flag" for rowing). The untaught model on the other node wrote the same story cleanly. The copy held 9 learned facts, two of them contradictory (founded in 2019 / in Tallinn). The copy was deleted to start again; its engine log on node 2 has the lessons |
| D5 | Living Weights, seen in trials (already in Ash's notes): (1.0.8 reports these learned more reliably; not re-checked here) with thinking on two kept facts came back as empty replies; a fact keyed on its subject does not answer the other direction; "I was born in Lisbon." is read as a fact about Lisbon and refused. Added 11 Oct evening, GB10: three plain personal facts in a row were taken out after 5 or 6 rounds each ("The user's dog is named Boogie", "I have a dog named Boogie", "My company's name is Eden": 1 of 4 held-out answers came back, 3 to 4 minutes each), while "My favourite sport is rowing" and "My cat is called Pixel" were kept the same day; the engine log of node 2 has the rounds |
| D7 | **Sampling: the penalties do nothing** (still on 1.0.9 on 13 Oct; 12 Oct, GB10, 1.0.7, Nemotron 3.5 Lightning). `repetition_penalty`, `frequency_penalty` and `presence_penalty` at 2.0 are accepted and change nothing (twenty "banana" in a row, greedy, same output with each), and `docs/api.md` does not list them: are they meant to work, or could the engine refuse them with a 400 so a client knows? `top_k`, `min_p`, `seed` and `stop` work per request. The same on the Mac mini M6 with Qwen3.8-27B (Python line): `repetition_penalty` 2.0 without effect. (An unseeded request repeating its draw, first reported here too, is by design: `docs/api.md` derives its seed from the prompt and `TENSORFOLD_SEED_SALT`. TensorFold Web's Chat now draws a seed per reply instead.) |

## E. For your information (no answer needed)

- **Services:** TensorFold Web now installs itself with launchd or systemd and keeps the engine across restarts, so the
  native binary needs no `service` command for us (this answers the earlier question).
- **Receipts:** the M6 receipt for Nemotron (1.0.2, exact, 85 to 108 tok/s with drafts); the Living Weights trials on
  two GB10; KL 0 between drafts on and off on a GB10, worth
  saying in the README. Bench writes JSON receipts; we follow CONTRIBUTING's format unless Ash prefers another.
- **What works:** `/v1/models`, streamed chat, OpenAI tool calls, Anthropic Messages with `tool_use`, Responses
  without that include; checked with Open WebUI, AnythingLLM, Claude Code and OpenCode.

## F. Answered since (no need to ask again)

From 1.0.9 (13 Oct): the 27B serves natively on the M6 (B1, #643; on our Mac mini M6 32 GB with its DFlash2 drafter: exact, 27.4 tok/s with drafts and 9.8 without, against 30 to 32 with drafts on the Python line, 19.5 GiB in use, no swap, and no memory left for the prompt cache); `/v1/responses` accepts OpenAI's `include` values, so Codex's request is answered (B3, #759; TensorFold Web's Connect test passes on node 2). `tensorfold pull` (fixed in 1.0.3; was the 8 October note's first point), the window in `/v1/models` (1.0.3),
`/health` open with a key, the runtime block on streamed chat, the 27B native (1.0.3), Ampere 8.6 and Blackwell 12.0
cards (1.0.3, 1.0.6), `response_format` and `tool_choice: "required"` (1.0.6).
