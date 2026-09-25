# AI agent runbook

Use this when someone asks you to set up TensorFold and serve a model on their Mac. Run these commands in a
terminal on the Mac that will host the model. TensorFold uses MLX and Metal on Apple Silicon; the NVIDIA in
Nemotron's name refers to the model, not to a CUDA device.

## 1. Check the Mac and choose one model

```bash
uname -m
python3.11 --version
```

Continue when the architecture is `arm64` and Python 3.11 is available. If Python 3.11 is missing and you
use Homebrew, run `brew install python@3.11`. Check available RAM and disk space before downloading a
checkpoint. Ask which model to use if the person has not picked one; download only that model and its
optional drafter.

| Model | Hugging Face repo | Download | Mac memory |
| --- | --- | --- | --- |
| Nemotron 3.5 Lightning | `Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit` | 18.6 GB | 32 GB or more |
| Qwen3.8-27B | `Vontra/Qwen3.8-27B-MLX-4bit` | 16.1 GB; optional drafter 3.8 GB | 32 GB or more |
| Qwen3.8 Flash Next | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` | 113 GB | 192 GB or more |

These are the checkpoints this CLI is built and tested with. Qwen3.8-27B's fast lane kernels require an
M5-generation GPU; on older Apple Silicon Macs it uses MLX kernels instead. Allow extra disk space for the
Hugging Face cache and extra memory for context and runtime buffers. See the [model notes](README.md#models)
for checkpoint requirements.

## 2. Install the CLI

If the repository is not already on the Mac, clone it first. Then install into a virtual environment:

```bash
git clone https://github.com/ashhart/TensorFold.git
cd TensorFold
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
tensorfold --version
tensorfold models
```

If you already have a checkout, start at `cd` in that checkout and skip `git clone`. Keep the environment
active for the remaining commands. `tensorfold models` lists the supported repos and their kernel packages.

## 3. Pull the chosen model

Run the matching command below. A pull downloads the checkpoint into the Hugging Face cache, normally under
`~/.cache/huggingface/hub`. `serve` can also download a missing model, but a separate pull makes download
failures easier to spot.

Nemotron 3.5 Lightning:

```bash
tensorfold pull Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit
```

This repo includes `mtp-4bit.safetensors`, the converted Nemotron MTP draft head. `pull` checks that it
arrived and prints `required model files ready: mtp-4bit.safetensors`. A later `serve` also finishes an
older cache that has the model weights but lacks the head.

Qwen3.8-27B, with its optional DFlash2 drafter for faster decoding:

```bash
tensorfold pull Vontra/Qwen3.8-27B-MLX-4bit z-lab/Qwen3.8-27B-DFlash2
```

Qwen3.8 Flash Next:

```bash
tensorfold pull Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP
```

For Qwen3.8-27B without the drafter, pull only its main repo. You can run `tensorfold info REPO_ID` to check
the detected family and kernel package; `info` fetches only `config.json` and does not pull model weights.

## 4. Start the server

Substitute the repo you pulled for `REPO_ID`:

```bash
tensorfold serve REPO_ID
```

Wait for TensorFold to print that it is serving at `http://127.0.0.1:8080/v1`. Leave that terminal running.
The default address accepts connections from this Mac. TensorFold takes the context limit from the model's
`config.json` and temperature, top-p and top-k from `generation_config.json`. The tested checkpoints each
advertise a 262,144-token context, but a long request needs enough memory to hold its cache. Use a smaller
cap such as `--context 8192` when appropriate; `--temperature`, `--top-p`, `--top-k` and `--max-tokens` can
set server defaults, and a request can override the sampling values and reply length. For Nemotron, check
that startup prints `Nemotron MTP head: active`; `inactive` means the head is present but drafting did not
activate because of flags or this MLX/GPU. Stop the server with Ctrl-C.

## 5. Check the endpoint

Use another terminal on the same Mac:

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/v1/models
```

Read the model ID from `/v1/models`. By default it is the final part of the repo ID, though `--name` can
change it. For the Nemotron checkpoint above, a chat request is:

```bash
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":128}'
```

For a different checkpoint, replace the `model` value with the ID returned by `/v1/models`. A successful
response has a `choices` array; reasoning, when enabled, may appear separately from the final answer.
Point an OpenAI-compatible client at `http://127.0.0.1:8080/v1` and use that same model ID. See the
[API notes](docs/api.md) for request fields and response details.

## If something fails

- `tensorfold: command not found`: activate `.venv` again in the current terminal.
- A download fails: check the exact repo ID, network access and free disk space, then rerun `tensorfold pull`.
- `info` works but `serve` still downloads: this is expected because `info` only needs the model config.
- The checkpoint is rejected: compare its quantization and draft head with the [model notes](README.md#models).
- A client cannot connect: keep `serve` running, check `/health`, and confirm its base URL and model ID.
