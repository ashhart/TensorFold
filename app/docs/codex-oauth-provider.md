# CLI OAuth Providers

TensorFold's first OAuth-backed frontier integrations use local CLI transports.
That matches Zora's `codex_cli` and `claude_cli` patterns: the app does not store an API key;
it delegates auth to the operator's existing local CLI login.

## Provider Shape

The provider lives in Settings alongside normal OpenAI-compatible endpoints:

```json
{
  "name": "Codex CLI OAuth",
  "transport": "codex-cli",
  "auth": "codex-oauth",
  "models": ["codex-cli-default"],
  "cliPath": "codex",
  "cliSandbox": "workspace-write",
  "cliApprovalPolicy": "never"
}
```

Claude CLI uses the same provider family:

```json
{
  "name": "Claude CLI OAuth",
  "transport": "claude-cli",
  "auth": "claude-oauth",
  "models": ["claude-code-default", "sonnet", "opus"],
  "cliPath": "claude",
  "cliPermissionMode": "default"
}
```

Normal HTTP providers continue to use:

```json
{
  "transport": "chat-completions",
  "auth": "api-key",
  "baseUrl": "https://api.openai.com/v1"
}
```

## Runtime Behavior

`streamChat()` now dispatches by provider transport:

- `chat-completions` calls `POST /chat/completions`.
- `codex-cli` runs `codex exec --json -`, flattens TensorFold's message history into stdin, and parses JSONL `agent_message` output.
- `claude-cli` runs `claude -p --output-format stream-json`, sends TensorFold's message history through stdin, and parses Claude Code stream-json text deltas.

The CLI bridges intentionally do not forward TensorFold's OpenAI tool schema. Codex CLI and Claude
CLI use their own native tool and permission surfaces. The Settings cards expose the auth source
and CLI controls so this is visible at provider-selection time.
