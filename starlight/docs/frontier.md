# Frontier providers, data policy and frontier-planned local work

Starlight can send requests to approved frontier providers under the same sign-in, limits and audit as local models, or keep
everything inside the network. This document covers providers, the rules that decide where a request may run, the
credential guard, offline mode, and workflows that plan with a frontier model and build locally.

## Providers and routes

A provider holds one organisation account; its API key is sealed at rest and never shown again.

| Kind | Base URL example | Authentication Starlight sends | Protocols by default |
| --- | --- | --- | --- |
| `openai` | `https://api.openai.com/v1` | `Authorization: Bearer` | chat, completions, embeddings, responses |
| `azure_openai` | `https://RESOURCE.openai.azure.com/openai/v1` | `api-key` | chat, completions, embeddings, responses |
| `anthropic` | `https://api.anthropic.com` | `x-api-key` and `anthropic-version` | messages, count_tokens, chat |
| `gemini` | `https://generativelanguage.googleapis.com/v1beta/openai` | `Authorization: Bearer` | chat, embeddings |
| `openai_compatible` | Your server's `/v1` base | `Authorization: Bearer` when a key is set | chat, completions, embeddings |

For OpenAI-style kinds the base URL includes the version path; Starlight appends `/chat/completions`, `/responses` and so on.
Anthropic routes use `/v1/messages`. A client's `anthropic-version` and `anthropic-beta` headers are passed on. Extra
non-secret headers (an organisation id, for example) can be stored with the provider; Starlight never lets them replace
authentication headers.

A provider also has `max_classification` (the highest data classification it may receive, `internal` by default), an
`internal` flag for servers on private addresses, and `enabled`.

A provider route connects a catalogue model to a provider's upstream model, with the protocols it speaks, a priority
(lower first), prices in micro-dollars per million input, output and cached tokens, and a concurrency cap. A model can have
several routes and local deployments together; its `route_policy` decides the order.

## Where a request may run

For each request Starlight computes a classification:

```
classification = max(project default, the key's floor, X-Starlight-Classification)
```

Levels are `public` < `internal` < `confidential` < `restricted`. A provider route is eligible only when the installation
is online, the model's policy allows cloud routes, the request is not from a local-only task, the route speaks the
request's protocol, and the provider's `max_classification` is at least the request's. Local replicas are eligible only on
nodes whose `max_classification` is at least the request's. If nothing is eligible the request answers `503 no_route`;
Starlight never relaxes a rule to find capacity.

On a cloud route Starlight drops TensorFold-only fields that providers would reject: `priority`, `draft`, `thinking_budget`,
`grammar_budget`, `return_token_ids`, `chat_template_kwargs`, `guided_json`, `guided_regex`, `guided_choice`,
`guided_grammar`, `structured_outputs`, `ignore_eos`, `min_p`, and `top_k` except on Anthropic routes.

## The credential guard

Starlight scans request bodies for credentials, in `local` mode only when a cloud route is possible: private keys, AWS
access and secret keys, GitHub, GitLab and Slack tokens, Google API keys, Stripe live keys, Anthropic and OpenAI keys,
JWTs, database connection strings with passwords, and Starlight's own tokens. Each project chooses what happens:

| Mode | Effect |
| --- | --- |
| `local` (default) | The request is treated as `restricted`: it runs on a local replica, or on a provider explicitly trusted with `restricted` data, or is refused with `400 credentials_detected` |
| `block` | The request is refused wherever it would run |
| `off` | No scanning |

Values are never logged; each detection writes an audit event naming the kinds found.

## Offline mode

`starlight init --offline`, or the switch under Settings in the console, makes the installation offline. Every cloud route
becomes ineligible, and the only client that can reach providers refuses all requests whatever the database says. Nodes
receive the flag and start engines without Hugging Face access. See [operations.md](operations.md#offline-and-air-gapped-installations)
for artifacts and bundles.

## Frontier-planned local work

There are two ways to plan with a frontier model and execute on local models.

### Plans

The plans API runs the whole workflow under policy:

1. A member submits a brief and the context they choose to share. The request goes to the planner model through the
   gateway with the member's own credential, so the credential guard, classification, limits and budgets all apply.
2. Starlight records the SHA-256 and size of the exact messages the planner received, and an audit event.
3. The plan is sealed at rest. A project administrator approves it by its SHA-256, so what is approved is exactly what was
   reviewed.
4. Approval issues the plan's owner a task token for the executor model only: local-only, pinned to the executor's current
   deployment revisions, with the plan's budget, concurrency and delegation depth.
5. The owner's agent runs with that token. Subagents get child tokens that can never widen it.

The API is in [api.md](api.md#plans).

### Aliases

Clients that use one model name for planning and another for execution can be split by aliases alone. For example, with
illustrative names, in a project:

| Alias | Catalogue model | Route policy |
| --- | --- | --- |
| `claude-opus-4-1` | `frontier-planner` | `cloud_only`, an Anthropic route |
| `claude-sonnet-4-5` | `qwen3.8-27b` | `local_only` |

A client configured to plan with the first name and build with the second then plans in the cloud and builds locally,
with every request still under the project's grants, limits and guard. Give the project a classification and guard mode
that match what may leave.

## Client setup

With an API key from the console:

```bash
export OPENAI_BASE_URL=https://ai.example.com/v1
export OPENAI_API_KEY=slr_...
```

```bash
export ANTHROPIC_BASE_URL=https://ai.example.com
export ANTHROPIC_AUTH_TOKEN=slr_...
claude --model claude-sonnet-4-5
```

Codex reads a model provider from `~/.codex/config.toml`, with its top-level keys above any table:

```toml
model_provider = "starlight"
model = "qwen3.8-27b"

[model_providers.starlight]
name = "TensorFold Starlight"
base_url = "https://ai.example.com/v1"
env_key = "STARLIGHT_API_KEY"
wire_api = "responses"
```

People with Connect do not need keys: `tensorfold-connect env claude-code`, `env codex` or `env openai` prints the same
settings for Connect's local endpoint, with an app key that acts as the signed-in person.

## Seat subscriptions

Frontier tools that are licensed per seat rather than per token can be tracked alongside usage: each subscription records
vendor, product, seats, seat price, billing period and renewal date, and seats are assigned to people. Because seats
belong to users, deactivating someone through SCIM or the console shows which seats to reclaim.
