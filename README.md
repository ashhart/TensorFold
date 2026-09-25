# TensorFold

TensorFold serves a local LLM on Apple Silicon at an OpenAI-compatible endpoint, fast and exact. Name a model
on Hugging Face, choose the context window and sampling, and TensorFold downloads it, loads it with Metal
kernels written for that model family, and serves `/v1/chat/completions`.

Setting this up with an AI agent? Give it the [AI agent runbook](RUNBOOK.md) for the install, model download,
server startup and a request that checks the result.

```bash
pip install git+https://github.com/ashhart/TensorFold.git
tensorfold serve Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit --context 65536
```

Any OpenAI client can then use `http://127.0.0.1:8080/v1`, including coding agents, SDKs and `curl`:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit", "messages": [{"role": "user", "content": "Hi"}]}'
```

TensorFold needs a Mac with Apple Silicon and Python 3.11. The install pins MLX to its tested 0.31 release
so the exact kernels and Nemotron MTP drafts stay active.

## Models

Each model family has its own package of kernels, picked from the checkpoint's `config.json`. These are the
checkpoints TensorFold is built and tested with, all on Hugging Face:

| Model | Pull | Size | Mac |
| --- | --- | --- | --- |
| Nemotron 3.5 Lightning 30B-A3B | `Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` | 18.6 GB | 32 GB or more |
| Qwen3.8-27B | `Vontra/Qwen3.8-27B-MLX-4bit` and its draft model `z-lab/Qwen3.8-27B-DFlash2` | 16.1 GB + 3.8 GB | 32 GB or more; an M5-generation GPU for the fast kernels |
| Qwen3.8 Flash Next | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` | 113 GB | 192 GB or more |

The three main checkpoints come from the `Vontra` Hugging Face namespace; Qwen3.8-27B's optional DFlash2
drafter comes from `z-lab`.

```bash
tensorfold pull Vontra/Qwen3.8-27B-MLX-4bit z-lab/Qwen3.8-27B-DFlash2
tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit
```

`serve` downloads a model it doesn't have yet; `pull` downloads ahead of time. Models go into the Hugging Face
cache (`~/.cache/huggingface`), and a local model directory works too. `tensorfold models` lists the families
and their checkpoints.

What each checkpoint needs:

- Qwen3.8 Flash Next drafts with the MTP head stored in its checkpoint, and its kernels read 4-bit weights in
  groups of 32. Use the `-MLX-4bit-MTP` conversion. TensorFold refuses other bit widths before downloading
  anything, and a conversion without the MTP head runs without drafts.
- Qwen3.8-27B drafts with the DFlash2 draft model once it has been pulled; `serve` picks it up automatically.
  Its lane kernels need 4-bit weights in groups of 64 and Metal 4 tensor units (M5-generation GPUs). Elsewhere
  it runs on MLX's own kernels: every token is still the model's own sample, but drafted rows are checked at
  width rather than bit-identical to one-row decoding.
- Nemotron 3.5 Lightning drafts with its MTP head, which the checkpoint above ships as `mtp-4bit.safetensors`
  (converted from NVIDIA's BF16 release; the standard MLX conversion drops it), and from the context.
  `pull` checks for the head, and `serve` completes an older cache that lacks it before loading.

Want another model? [The recipe book](docs/recipes/README.md) describes what we did for each family and how
to add yours.

## Speed

Decode speeds we measured through the server. They depend on content: copies of earlier text (file edits)
and predictable output (code, tool calls) draft well, fresh prose less so. The Nemotron rows predate its MTP
drafts, which are now on by default; in-engine they reached 217 tok/s on prose and 228 on code.

| Model | Machine | Workload | tok/s |
| --- | --- | --- | --- |
| Nemotron 3.5 Lightning, 4-bit | M5 Max, 128 GB | short answer with thinking | 188-206 (mlx_lm: 138) |
| | | about 20k-token context | 175 |
| | | about 60k-token context | 162 |
| Qwen3.8-27B, 4-bit, DFlash2 drafter | M5 Max, 128 GB | short answer with thinking | 120-124 (27 without drafts) |
| | | code | 189 (26 without drafts) |
| Qwen3.8 Flash Next, 4-bit | M3 Ultra, 256 GB | short answer with thinking | 88-92 (79 without drafts) |
| | | code | 110 (80 without drafts) |
| | | file edit | 170-176 |
| | | 18k-token context | 77-84 |
| | | 23k-token agent prompt, 512 thinking tokens, then a long tool call | 91-99 |

## Exact means byte-identical

Speculative decoding usually trades determinism for speed: drafted tokens are accepted by a random test, so the
same prompt and seed give different text depending on how drafting went. TensorFold keeps two guarantees
instead.

1. Every token is the model's own sample. The token at position `p` is the argmax over the top-k/top-p
   candidates of `logit / T + g(seed, p, token)`, with `g` Gumbel noise from a hash of the seed, the position
   and the token id. That is an exact draw from the top-k/top-p distribution, and it depends only on that row's
   logits and its position.
2. A row of a multi-row verify pass gets the same bits as a one-row step. The kernels are written so that a
   row's arithmetic does not depend on how many rows share the pass.

So a draft is accepted exactly when it equals the token one-token-at-a-time decoding would sample there, and
drafted decoding writes the same bytes as serial decoding. Drafts change speed only. You can check it yourself:
send the same request with `"draft": false`, which decodes one token a round, and compare.

The default seed is a hash of the prompt, so the same conversation gets the same reply. Pass `"seed"` to vary
it. One limit: for Flash Next and Nemotron the prompt's cache can differ in its last bits depending on which
prefix was already cached, so a reply can too ([details](docs/recipes/README.md#a-known-limit)). Drafted and
serial decoding from the same cache always agree.

## Serve

```bash
tensorfold serve MODEL [options]    # MODEL: a Hugging Face repo id or a model directory
tensorfold pull REPO [REPO ...]      # download models or draft models
tensorfold models                   # families and the checkpoints they are tested with
tensorfold info MODEL               # which family serves a model (reads its config.json only)
```

| Option | Default | What it does |
| --- | --- | --- |
| `--host`, `--port` | `127.0.0.1`, `8080` | where to listen (`--host 0.0.0.0` for other machines) |
| `--name`, `--alias` | the model's name | the model id clients send |
| `--context N` | the model's `max_position_embeddings` | prompt plus reply tokens a request may use; longer prompts get HTTP 400; `0` removes TensorFold's cap |
| `--max-tokens N` | 4096 | reply length when a request does not set `max_tokens` |
| `--temperature`, `--top-p`, `--top-k` | the model's `generation_config.json` | sampling defaults; `--temperature 0` is greedy |
| `--thinking` / `--no-thinking` | on | open a think block when the chat template supports one |
| `--reasoning-effort` | `medium` | for chat templates that take one (Qwen3.8) |
| `--thinking-budget N` | no limit | most thinking tokens before the server closes the think block |
| `--no-drafts` | off | one token a round: the serial reference |
| `--drafter` | `auto` | the family's draft model once pulled; a repo id or directory; or `none` |
| `--mtp-drafts N` | 3 | most MTP drafts a round (Qwen3.8 Flash Next); 0 turns MTP drafts off (both MTP families) |
| `--prompt-cache-gib` | an eighth of RAM, at most 16 | memory for cached conversation prefixes |
| `--snapshot-dir` | `~/.cache/tensorfold/prefix-snapshots` | system blocks and conversations kept across restarts |

Requests can override the sampling fields, the thinking switch and the budget. See [the API notes](docs/api.md)
for the fields TensorFold reads and what it returns.

By default, `serve` reads temperature, top-p and top-k from the checkpoint's `generation_config.json`; a
checkpoint with `do_sample: false` decodes greedily. CLI sampling flags override those values, and each
request can override them again. The context default comes from the checkpoint's `config.json`; large
windows still need enough memory for the actual prompt and reply.

## Prompt caching

Agent clients resend the whole conversation every turn. TensorFold keeps the caches of recent conversation
prefixes, so a follow-up only prefills its new suffix. It saves the system block, which a client sends with
every session, to disk once, so a new session starts without prefilling it. The newest conversations are saved
at shutdown and read back on demand.

## Layout

```
src/tensorfold/
  cli.py                 the tensorfold command
  hub.py                 models by Hugging Face repo id
  server/                OpenAI HTTP layer (http.py) and the request queue, caches and streaming (app.py)
  engine/                the serial engine, the lane engine, exact sampling, prompt caches
  kernels/qwen/dense/v1/        Qwen3.8 dense lane kernels
  kernels/qwen/flash_next/v1/   Qwen3.8 Flash Next fused kernels
  kernels/nemotron/lightning/v1/  Nemotron 3.5 Lightning fused kernels
  drafters/              the DFlash2 drafter
  families/<name>/       one package per model family: forward pass and draft heads
docs/recipes/            what we did per family, and how to add one
```

## Development

```bash
git clone https://github.com/ashhart/TensorFold.git
cd TensorFold
pip install -e ".[test]"
pytest
```

The kernel tests of the lane engine need an M5-generation GPU and are skipped elsewhere.

## License

MIT. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md). Each model keeps its own license;
see its Hugging Face page.
