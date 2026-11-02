# Starlight API

Starlight speaks the OpenAI and Anthropic HTTP APIs on its public listener, adds a few headers of its own, and serves the OAuth,
task and plan endpoints its clients use. The base URL is `public_url`, for example `https://ai.example.com`.

## Authentication

Send a credential as `Authorization: Bearer <token>` or, for Anthropic clients, `x-api-key: <token>`.

| Credential | Comes from | Acts in |
| --- | --- | --- |
| API key (`slr_`) | Console, for a person or a service account | The key's project |
| Access token (`sla_`) | Connect or a CLI through OAuth | The project in `X-Starlight-Project`, else the person's only project or the everyone project |
| Task token (`slt_`) | The tasks or plans API | The task's project, models and route rules |
| IdP JWT | An OIDC connection with bearer tokens enabled and an audience set | As an access token |

A credential that is missing or invalid answers `401 invalid_api_key` with `WWW-Authenticate: Bearer`.

## Inference endpoints

| Method and path | Protocol | Notes |
| --- | --- | --- |
| `GET /v1/models` | both | Models and aliases the caller may use in its project |
| `POST /v1/chat/completions` | OpenAI | Streaming and non-streaming |
| `POST /v1/completions` | OpenAI | Raw prompts |
| `POST /v1/embeddings` | OpenAI | Only models whose kind is `embedding` |
| `POST /v1/responses` | OpenAI Responses | Streaming and non-streaming, stored responses |
| `GET /v1/responses/{id}`, `DELETE /v1/responses/{id}` | OpenAI Responses | Owner only |
| `POST /v1/messages` | Anthropic Messages | Streaming and non-streaming |
| `POST /v1/messages/count_tokens` | Anthropic Messages | Counted by the serving engine or provider |

The `model` field may name a catalogue model or an alias. Starlight resolves a project alias first, then a global alias, then
a model name, and requires a grant for the caller's project. Local TensorFold engines serve every protocol above; a
provider route serves only the protocols listed on it (see [frontier.md](frontier.md)). Starlight never translates between
protocols.

Starlight rewrites a request in four places before it leaves:

- `model` becomes the engine's served name or the provider's upstream model.
- A missing output limit (`max_completion_tokens` or `max_tokens`, `max_output_tokens`, Anthropic `max_tokens`) is set to
  `gateway.default_max_tokens` or the project's grant cap, whichever is lower. A limit above the grant cap is refused.
- Streaming chat completions get `stream_options.include_usage` so usage can be metered; the extra usage chunk is hidden
  unless the client asked for it.
- On a cloud route, TensorFold-only fields are dropped: `priority`, `draft`, `thinking_budget`, `grammar_budget`,
  `return_token_ids`, `chat_template_kwargs`, `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`,
  `structured_outputs`, `ignore_eos`, `min_p`, and `top_k` (kept for Anthropic routes). On a local route at batch
  priority, Starlight sets the engine's `priority: "background"`.

### Stored responses

`POST /v1/responses` stores the response unless the body sends `"store": false`. Starlight records who created it, in which
project and on which replica or provider route, for seven days. `GET` and `DELETE /v1/responses/{id}` and
`previous_response_id` work only for that principal in that project, and a continuation always runs where the response is
stored. Anyone else gets `404`.

### `GET /v1/models`

```json
{"object": "list", "data": [
  {"id": "qwen3.8-27b", "object": "model", "created": 1767225600, "owned_by": "starlight",
   "starlight": {"model": "qwen3.8-27b", "display_name": "Qwen3.8 27B", "kind": "chat", "route_policy": "local_first"}},
  {"id": "claude-sonnet-4-5", "object": "model", "created": 1767225600, "owned_by": "starlight",
   "starlight": {"model": "qwen3.8-27b", "display_name": "Qwen3.8 27B", "kind": "chat", "route_policy": "local_first"}}
]}
```

An alias appears with the model it resolves to in `starlight.model`. With an `anthropic-version` header the list uses the
Anthropic shape (`data[].type`, `id`, `display_name`, `created_at`, `has_more`, `first_id`, `last_id`).

## Starlight headers

Request headers:

| Header | Values | Effect |
| --- | --- | --- |
| `X-Starlight-Project` | Project slug or id | Chooses the project for an access token or IdP JWT; it must be one the person belongs to |
| `X-Starlight-Priority` | `batch`, `standard`, `interactive` | Requests a service class, capped by the credential and the project |
| `X-Starlight-Classification` | `public`, `internal`, `confidential`, `restricted` | Raises the request's data classification; it can never lower it |
| `X-Starlight-Client` | Free text, 80 printable characters | Labels the request in usage records; Connect sends `connect/<app>` |

A body with `"priority": "background"` is treated as batch.

Response headers:

| Header | Meaning |
| --- | --- |
| `X-Request-Id` | Starlight's id for the request, also in usage and audit records; client-sent ids are ignored |
| `X-Starlight-Route` | `local` or `cloud`: where the request ran |
| `X-Starlight-Model` | The catalogue model that served it |
| `Retry-After` | Seconds to wait, on 429 and 503 answers |

## Errors

Errors use the dialect of the endpoint. OpenAI-style endpoints answer
`{"error": {"message": "...", "type": "...", "code": "...", "param": null}}`; Anthropic endpoints answer
`{"type": "error", "error": {"type": "...", "message": "..."}}`.

| Status | Code | When |
| --- | --- | --- |
| 400 | `invalid_body` | The body could not be read |
| 400 | `invalid_json` | The body is not one JSON object, or a key appears twice |
| 400 | `invalid_request_error` | A field Starlight checks is malformed (`param` names it), the model does not serve this endpoint, the output limit exceeds the project's cap or 1,048,576, or `n` or `best_of` exceeds 16 |
| 400 | `context_length_exceeded` | The estimated prompt plus output exceeds the project's context cap |
| 400 | `credentials_detected` | The prompt appears to contain credentials and the project refuses them, or only cloud routes exist |
| 401 | `invalid_api_key` | Missing, invalid, expired or revoked credential |
| 403 | `permission_denied` | The credential may not run inference in this project |
| 403 | `project_required` | Choose a project with `X-Starlight-Project`, or the named project is not yours |
| 404 | `model_not_found` | Unknown model, or not granted to this project or credential |
| 404 | `previous_response_not_found`, `not_found` | Unknown stored response, or not yours |
| 413 | `request_too_large` | Body larger than `gateway.max_body_bytes` (32 MiB by default) |
| 429 | `concurrency_limit`, `rate_limit`, `budget_limit` | A limit for one of the request's scopes is reached; `Retry-After` is set |
| 502 | `upstream_unreachable`, `upstream_error` | The node or provider did not answer, or its answer could not be read |
| 503 | `capacity_unavailable` | Every eligible replica stayed busy until the queue timeout; `Retry-After: 2` |
| 503 | `no_route` | No deployment or provider may serve this request under its data policy |
| 500 | `internal_error` | An unexpected failure; the message is deliberately generic |

Errors from the engine or provider are passed through unchanged. When a stream breaks after it has started, OpenAI-style
streams end with an error event (`code: stream_interrupted`) and `[DONE]`; Anthropic streams end with an `error` event.

## Discovery

`GET /.well-known/tensorfold-starlight` needs no credential:

```json
{"issuer": "https://ai.example.com", "org_name": "Example Corp", "version": "1.0.0", "installation_id": "0190f5a0-...",
 "offline": false,
 "authorization_endpoint": "https://ai.example.com/oauth/authorize",
 "token_endpoint": "https://ai.example.com/oauth/token",
 "device_authorization_endpoint": "https://ai.example.com/oauth/device/code",
 "revocation_endpoint": "https://ai.example.com/oauth/revoke",
 "userinfo_endpoint": "https://ai.example.com/oauth/userinfo",
 "api_base": "https://ai.example.com/v1", "console_url": "https://ai.example.com/console"}
```

## OAuth for clients

Starlight's own public clients are `tensorfold-connect` (authorization code with PKCE on a loopback redirect, or the device
grant) and `tensorfold-cli` (device grant). Scopes are `inference`, `offline_access` (a refresh token) and `profile`.

| Endpoint | Purpose |
| --- | --- |
| `GET /oauth/authorize` | Signed-in approval page; `redirect_uri` must be `http://127.0.0.1:<port>/...` or `http://[::1]:<port>/...`, `code_challenge_method=S256` |
| `POST /oauth/token` | `authorization_code` (with `code_verifier`), `refresh_token`, `urn:ietf:params:oauth:grant-type:device_code` |
| `POST /oauth/device/code` | Starts a device sign-in; returns `user_code`, `verification_uri` (`/device`) and `interval` |
| `POST /oauth/revoke` | RFC 7009; revoking a refresh token revokes its family |
| `GET /oauth/userinfo` | The person, organisation and usable projects with their models |

Codes are single use and expire after two minutes; a code presented with a wrong verifier is burned. Token errors follow
RFC 6749 (`invalid_grant`, `authorization_pending`, `slow_down`, `access_denied`, `expired_token`).

```json
{"user_id": "0190f5a0-...", "email": "ada@example.com", "display_name": "Ada", "org_name": "Example Corp",
 "projects": [{"id": "0190f5a1-...", "slug": "eng", "name": "Engineering", "role": "member", "classification": "internal",
   "default": true, "models": ["qwen3.8-27b"]}]}
```

## Tasks

A task token is a credential for one agent run. `POST /v1/starlight/tasks` with a person's or service account's credential
creates a root task; the same call made with a task token creates a child of that task. Only a credential with the
`inference` scope can mint one, and the tree stays bound to it: a task never outlives the API key that minted it, never
gains a model, address or priority that key lacked, and stops working when that key is revoked or the Connect sign-in
that minted it is signed out.

```bash
curl -s https://ai.example.com/v1/starlight/tasks -H "Authorization: Bearer $STARLIGHT_API_KEY" -H 'Content-Type: application/json' \
  -d '{"name": "refactor", "models": ["qwen3.8-27b"], "max_concurrency": 4, "max_depth": 2, "budget_tokens": 2000000,
       "ttl_seconds": 86400, "pin_revisions": true}'
```

```json
{"id": "0190f5b2-...", "root_id": "0190f5b2-...", "parent_id": null, "depth": 0, "name": "refactor", "models": ["qwen3.8-27b"],
 "local_only": true, "max_concurrency": 4, "max_depth": 2, "budget_tokens": 2000000,
 "pins": {"qwen3.8-27b": [{"deployment": "0190f5a9-...", "revision": 3}]}, "expires_at": "2026-11-02T10:00:00Z",
 "revoked": false, "token": "slt_...", "used_tokens": 0}
```

| Field | Rule |
| --- | --- |
| `models` | 1 to 16 models granted to the project; a child's must be a subset of its parent's |
| `local_only` | Defaults to true; a child of a local-only task is always local-only |
| `max_concurrency` | Default 4, at most 64; a child gets at most its parent's |
| `max_depth` | Root only, at most 5; a child is refused beyond it (`403 delegation_depth_exceeded`) |
| `budget_tokens` | Root only; shared by the whole tree |
| `ttl_seconds` | Default one day, at most seven days; a child never outlives its parent |
| `pin_revisions` | Pins each model with a local deployment to its current revisions, so local requests run only on those revisions and never on a newer checkpoint |

`GET /v1/starlight/tasks/{id}` returns the task and the tokens its subtree has used; `DELETE /v1/starlight/tasks/{id}` revokes it
and every task below it. Both work for the owner and for a task token at or above the task.

## Plans

A plan is drafted by a planner model, approved by its SHA-256, and executed by a local model with a task token.

```bash
curl -s https://ai.example.com/v1/starlight/plans -H "Authorization: Bearer $STARLIGHT_API_KEY" -H 'Content-Type: application/json' \
  -d '{"title": "Add retries", "planner_model": "frontier-planner", "executor_model": "qwen3.8-27b",
       "brief": "Add retries with backoff to the HTTP client.",
       "context": [{"name": "client.go", "content": "package client\n..."}],
       "budget_tokens": 500000, "max_concurrency": 2, "max_depth": 1, "ttl_seconds": 86400}'
```

```json
{"id": "0190f5c0-...", "project_id": "0190f5a1-...", "title": "Add retries", "planner_model": "frontier-planner",
 "executor_model": "qwen3.8-27b", "status": "pending_approval", "plan": "# Goal\n...", "plan_sha256": "9f2c...",
 "egress_sha256": "41d7...", "egress_bytes": 1840}
```

| Endpoint | Who | Effect |
| --- | --- | --- |
| `POST /v1/starlight/plans` | A member with the plans permission | Drafts through the gateway with the caller's own credential and policy |
| `POST /v1/starlight/plans/{id}/approve` with `{"plan_sha256": "..."}` | A project administrator | Approves only the plan with that hash (`409 plan_changed` otherwise) and issues the owner a task |
| `POST /v1/starlight/plans/{id}/reject` | A project administrator | Rejects a pending plan |
| `GET /v1/starlight/plans/{id}` | The owner or an approver | The plan; the owner receives `task_token` once, on the first read after approval |

`max_tokens` sets the planner's output limit (8,192 by default). Projects with `plan_approval: auto` approve drafts as
soon as they are written. `egress_sha256` is the SHA-256 of the exact `messages` bytes the planner received; routing
never rewrites them.

## Health

`GET /healthz` answers `ok` while the process runs; `GET /readyz` answers `ready` once the database schema matches the
build. Both are served on the public and the metrics listener.

The SCIM API is described in [scim.md](scim.md) and the node protocol in [nodes.md](nodes.md).
