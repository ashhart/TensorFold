# TensorFold

TensorFold serves a local LLM on Apple Silicon at an OpenAI-compatible endpoint, fast and exact. Point it at a
model directory, choose the context window and sampling, and it loads the model with Metal kernels written for
that model family:

```bash
tensorfold serve ~/models/Qwen3.8-Flash-Next-MLX-4bit-MTP --context 65536 --port 8080
```

Any OpenAI client can then use `http://127.0.0.1:8080/v1`, including coding agents, SDKs and `curl`.

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

## Models

TensorFold picks a family package from the checkpoint's `config.json` `model_type`. Each family keeps its own
forward pass, kernels and drafting.

| Family | `model_type` | Drafting | Tested on |
| --- | --- | --- | --- |
| Qwen3.8 Flash Next | `qwen4_exp` | the checkpoint's MTP head (up to 3 drafts a round) and copies from the context | M3 Ultra, MLX 0.32.0 |
| Nemotron 3.5 Lightning (30B-A3B) | `nemotron_h` | copies from the context, GPU-side sampling one step ahead | M5 Max, MLX 0.31.2 |
| Qwen3.8 dense (27B) | `qwen3_5` | DFlash2 draft trees, copies, tool-call structure | M5 Max, MLX 0.31.2 |

Decode speeds we measured through the server. They depend on content: copies of earlier text (file edits)
and predictable output (code, tool calls) draft well, fresh prose less so.

| Model | Machine | Workload | tok/s |
| --- | --- | --- | --- |
| Qwen3.8 Flash Next, 4-bit | M3 Ultra, 256 GB | short answer with thinking | 88-92 (79 without drafts) |
| | | code | 110 (80 without drafts) |
| | | file edit | 170-176 |
| | | 18k-token context | 77-84 |
| | | 23k-token agent prompt, 512 thinking tokens, then a long tool call | 91-99 |
| Nemotron 3.5 Lightning, 4-bit | M5 Max, 128 GB | short answer with thinking | 188-206 (mlx_lm: 138) |
| | | about 20k-token context | 175 |
| | | about 60k-token context | 162 |
| Qwen3.8-27B, 4-bit, DFlash2 drafter | M5 Max, 128 GB | short answer with thinking | 120-124 (27 without drafts) |
| | | code | 189 (26 without drafts) |

The Qwen3.8 dense lane kernels use the Metal 4 tensor units of M5-generation GPUs. On older GPUs TensorFold runs
that family on MLX's own kernels: every token is still the model's sample, but drafted rows are checked at width
rather than bit-identical to one-row decoding.

Want another model? [The recipe book](docs/recipes/README.md) describes what we did for each family and how
to add yours.

## Install

TensorFold needs macOS on Apple Silicon and Python 3.11 or newer.

```bash
git clone https://github.com/ashhart/TensorFold.git
cd TensorFold
pip install -e .
```

A model is an MLX checkpoint directory: `config.json`, safetensors weights and the tokenizer files, as mlx-lm
writes them. TensorFold does not download weights. For Qwen3.8-27B with drafts, fetch the draft model into the
Hugging Face cache once: `huggingface-cli download z-lab/Qwen3.8-27B-DFlash2`. Qwen3.8 Flash Next drafts with the
MTP head stored in its checkpoint, so use a conversion that keeps the `mtp.*` weights.

## Serve

```bash
tensorfold serve MODEL_DIR [options]
tensorfold models            # the families this build supports
tensorfold info MODEL_DIR    # which family serves a directory (reads config.json only)
```

| Option | Default | What it does |
| --- | --- | --- |
| `--host`, `--port` | `127.0.0.1`, `8080` | where to listen (`--host 0.0.0.0` for other machines) |
| `--name`, `--alias` | directory name | the model id clients send |
| `--context N` | no limit | prompt plus reply tokens a request may use; longer prompts get HTTP 400 |
| `--max-tokens N` | 4096 | reply length when a request does not set `max_tokens` |
| `--temperature`, `--top-p`, `--top-k` | the model's `generation_config.json` | sampling defaults; `--temperature 0` is greedy |
| `--thinking` / `--no-thinking` | on | open a think block when the chat template supports one |
| `--reasoning-effort` | `medium` | for chat templates that take one (Qwen3.8) |
| `--thinking-budget N` | no limit | most thinking tokens before the server closes the think block |
| `--no-drafts` | off | one token a round: the serial reference |
| `--drafter ID_OR_DIR` | none | DFlash2 draft model (Qwen3.8 dense) |
| `--mtp-drafts N` | 3 | most MTP drafts a round (Qwen3.8 Flash Next) |
| `--prompt-cache-gib` | 16 | memory for cached conversation prefixes |
| `--snapshot-dir` | `~/.cache/tensorfold/prefix-snapshots` | system blocks and conversations kept across restarts |

Requests can override the sampling fields, the thinking switch and the budget. See [the API notes](docs/api.md)
for the fields TensorFold reads and what it returns.

## Prompt caching

Agent clients resend the whole conversation every turn. TensorFold keeps the caches of recent conversation
prefixes, so a follow-up only prefills its new suffix. It saves the system block, which a client sends with
every session, to disk once, so a new session starts without prefilling it. The newest conversations are saved
at shutdown and read back on demand.

## Layout

```
src/tensorfold/
  cli.py                 the tensorfold command
  server/                OpenAI HTTP layer (http.py) and the request queue, caches and streaming (app.py)
  engine/                the serial engine, the lane engine, exact sampling, prompt caches
  kernels/               lane matmul and attention (tensor units) and the Qwen3.8 dense GDN helpers
  drafters/              the DFlash2 drafter
  families/<name>/       one package per model family: forward pass, kernels, draft heads
docs/recipes/            what we did per family, and how to add one
```

## Tests

```bash
pip install -e ".[test]"
pytest
```

The kernel tests of the lane engine need an M5-generation GPU and are skipped elsewhere.

## License

MIT. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
