# Model Provider Discovery

TensorFold treats model backends as OpenAI-compatible providers wherever possible. This keeps
local and cloud models usable in the same composer, automations, tickets, and rooms.

## Local Providers

- **oMLX** reads `~/.omlx/settings.json` for the API server port/token and
  `~/.omlx/model_settings.json` plus `~/.omlx/stats.json` for configured model names.
- **Ollama** probes `http://127.0.0.1:11434/api/tags`, then uses
  `http://127.0.0.1:11434/v1` for OpenAI-compatible chat.
- **LM Studio** probes `http://127.0.0.1:1234/v1/models`, with
  `http://127.0.0.1:1234/api/v0/models` as a richer fallback. If LM Studio
  requires authentication, TensorFold keeps the provider card and asks the user to
  paste the local token.

Local discovery only returns redacted audit metadata: provider kind, base URL, model count,
and whether an API key exists. It must not log token values.

## OpenRouter

OpenRouter is just another OpenAI-compatible provider at `https://openrouter.ai/api/v1`.
TensorFold's refresh path is OpenRouter-aware:

- Queries `/models?sort=pricing-low-to-high&supported_parameters=tools`.
- Keeps router slugs near the top: `openrouter/free`, `openrouter/auto`, and `openrouter/fusion`.
- Seeds useful current model IDs such as `z-ai/glm-5.2`,
  `nvidia/nemotron-3-ultra-550b-a55b:free`, `qwen/qwen3-coder:free`, and
  `openai/gpt-oss-120b:free`.

## Multi-Model Rooms

Rooms already support multiple models at the TensorFold layer: every seat stores a
`providerId` and `model`. To make a mixed room, add one seat on OpenRouter Free,
another on GLM/Nemotron/Qwen, and another on a local Ollama/LM Studio/oMLX model.
The room engine handles the turn-taking, context compaction, and final gate.
