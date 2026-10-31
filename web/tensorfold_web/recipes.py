"""Connect's recipes: how each tool reaches TensorFold, the settings to paste, and the checks that prove it works.

A recipe is data: the tool, the kind of tool, the protocol it speaks to the engine, its settings for a given
address, key and model, and the calls the "Test connection" button makes, the same calls the tool makes. A recipe
also says where and when it was checked end to end with the real tool, or why it could not be.
"""

from __future__ import annotations

import json
import re
import shlex
import time
from dataclasses import dataclass, field

import httpx

from . import httpread

KINDS = {"chat": "Chat front ends", "agent": "Coding agents", "assistant": "Personal agents", "sdk": "Code"}
CHECK_LABELS = {
    "models": "The model is listed at /v1/models",
    "chat_stream": "A streamed chat reply (OpenAI chat completions)",
    "tool_call": "A tool call comes back as tool_calls",
    "anthropic": "A streamed reply through Anthropic Messages (/v1/messages)",
    "anthropic_tools": "A tool call through Anthropic Messages (tool_use)",
    "responses": "A reply through the Responses API (/v1/responses)",
    "context": "The served window holds the tool's prompts",
}
WEATHER_TOOL = {"name": "get_weather", "description": "Current weather for a city.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}


@dataclass
class Ctx:
    """What a recipe is written for: the address as the tool sees it, the key, the model."""

    base: str                       # without /v1
    key: str                        # "local" when the engine needs none
    model: str
    docker_base: str = ""           # the same engine as a container on this machine reaches it
    window: int | None = None       # the context window the engine serves now, when it says
    away: bool = False              # written for a tool on another computer, through TensorFold Web's address


@dataclass
class Recipe:
    id: str
    name: str
    kind: str
    protocol: str                   # "openai", "anthropic" or "responses"
    note: str
    checks: tuple[str, ...]
    checked: dict | None = None     # {"date", "machine", "how"} once checked with the real tool, or {"not": why}
    extra: dict = field(default_factory=dict)

    def snippet(self, c: Ctx) -> str:
        return SNIPPETS[self.id](c)

    def to_dict(self, c: Ctx) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind, "protocol": self.protocol, "note": self.note,
                "snippet": self.snippet(c), "checks": [{"id": k, "label": CHECK_LABELS[k]} for k in self.checks],
                "checked": self.checked, "min_context": self.extra.get("min_context"),
                "form": FORMS.get(self.id, ""), "where_note": (AWAY_NOTES if c.away else HERE_NOTES).get(self.id),
                **({"compose": True} if self.id == "open-webui" and not c.away else {})}


def _sh(value: str) -> str:
    """A value pasted into a shell, quoted: a model's name comes from the engine, which may not be TensorFold Web's."""

    return shlex.quote(value)


def _yaml(value: str) -> str:
    """A YAML scalar: plain when it reads the same, quoted otherwise."""

    return value if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@ ()+-]*", value) and ": " not in value \
        and not value.endswith(" ") else json.dumps(value)


def _key_line(c: Ctx, line: str) -> str:
    return line if c.key != "local" else ""


SNIPPETS = {
    "open-webui": lambda c: (
        "# Open WebUI in a container on that computer, using this AI through TensorFold Web\n"
        "docker run -d --name tensorfold-open-webui -p 3000:8080 \\\n"
        f"  -e {_sh('OPENAI_API_BASE_URL=' + c.base + '/v1')} -e {_sh('OPENAI_API_KEY=' + c.key)} \\\n"
        "  -e ENABLE_OLLAMA_API=false -v tensorfold-open-webui:/app/backend/data ghcr.io/open-webui/open-webui:main\n"
        "# then http://localhost:3000 on that computer") if c.away else (
        "# Open WebUI in a container on this machine, using the engine (compose file: Download below)\n"
        f"docker run -d --name tensorfold-open-webui --network host -e PORT=3000 \\\n"
        f"  -e {_sh('OPENAI_API_BASE_URL=' + c.base + '/v1')} -e {_sh('OPENAI_API_KEY=' + c.key)} \\\n"
        "  -e ENABLE_OLLAMA_API=false -v tensorfold-open-webui:/app/backend/data ghcr.io/open-webui/open-webui:main\n"
        "# then http://localhost:3000; on a Mac, use -p 3000:8080 and http://host.docker.internal:"
        f"{c.base.rsplit(':', 1)[-1]}/v1 instead of --network host"),
    "librechat": lambda c: (
        "# librechat.yaml, in the custom endpoints\nendpoints:\n  custom:\n    - name: \"TensorFold\"\n"
        f"      apiKey: {json.dumps(c.key)}\n      baseURL: {json.dumps((c.docker_base or c.base) + '/v1')}\n"
        f"      models:\n        default: [{json.dumps(c.model)}]\n        fetch: true\n"
        "      titleConvo: true\n      titleModel: \"current_model\"\n      modelDisplayLabel: \"TensorFold\""),
    "anythingllm": lambda c: (
        "# AnythingLLM: Settings > LLM Preference > Generic OpenAI, or in its .env\nLLM_PROVIDER=generic-openai\n"
        f"GENERIC_OPEN_AI_BASE_PATH={c.docker_base or c.base}/v1\nGENERIC_OPEN_AI_MODEL_PREF={c.model}\n"
        f"GENERIC_OPEN_AI_API_KEY={c.key}\nGENERIC_OPEN_AI_MODEL_TOKEN_LIMIT={c.window or 16384}\nGENERIC_OPEN_AI_MAX_TOKENS=4096\n"
        "DISABLE_TELEMETRY=true\n# then set the workspace's chat mode to Chat: new workspaces start in agent mode"),
    "jan": lambda c: (
        "Jan: Settings > Model Providers > Add provider (OpenAI-compatible)\n"
        f"Base URL   {c.base}/v1\nAPI key    {c.key}\nModel      {c.model}  (or Refresh to list it)"),
    "claude-code": lambda c: (
        f"export ANTHROPIC_BASE_URL={_sh(c.base)}\nexport ANTHROPIC_AUTH_TOKEN={_sh(c.key)}\n"
        f"export ANTHROPIC_MODEL={_sh(c.model)}\n"
        "export ANTHROPIC_SMALL_FAST_MODEL=$ANTHROPIC_MODEL\nexport CLAUDE_CODE_ATTRIBUTION_HEADER=0\n"
        "export CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192   # its default asks 32k reply tokens on top of a 16k-token prompt\n"
        f"export CLAUDE_CODE_MAX_CONTEXT_TOKENS={c.window or 65536}  # the engine's window, for a model Claude Code "
        "does not know\n"
        "claude"),
    "opencode": lambda c: json.dumps(
        {"$schema": "https://opencode.ai/config.json",
         "provider": {"tensorfold": {"npm": "@ai-sdk/openai-compatible", "name": "TensorFold",
                                     "options": {"baseURL": f"{c.base}/v1", **({"apiKey": c.key} if c.key != "local" else {})},
                                     "models": {c.model: {"name": f"{c.model} (local)"}}}},
         "model": f"tensorfold/{c.model}"}, indent=2),
    "codex": lambda c: (
        f"# ~/.codex/config.toml\nmodel = {json.dumps(c.model)}\nmodel_provider = \"tensorfold\"\n\n"
        f"[model_providers.tensorfold]\nname = \"TensorFold\"\nbase_url = {json.dumps(c.base + '/v1')}\n"
        "wire_api = \"responses\"" + ("\nenv_key = \"TENSORFOLD_API_KEY\"   # export TENSORFOLD_API_KEY first: Connect shows "
                                       "its value (it is not written in this file)" if c.key != "local" else "")),
    "cline": lambda c: (
        "Cline (VS Code): Settings > API Provider: OpenAI Compatible\n"
        f"Base URL         {c.base}/v1\nAPI Key          {c.key}\nModel ID         {c.model}\n"
        f"Context Window   {c.window or 65536}   (Model Configuration)"),
    "pi": lambda c: json.dumps(
        {"$schema": "https://pi.dev/schemas/models.schema.json",
         "providers": {"tensorfold": {"baseUrl": f"{c.base}/v1", "api": "openai-completions", "apiKey": c.key,
                                      "models": [{"id": c.model, "name": f"{c.model} (TensorFold)",
                                                  "contextWindow": c.window or 65536, "maxTokens": 8192}]}}},
        indent=2),
    "omp": lambda c: (
        "# ~/.omp/agent/models.yml\nproviders:\n  tensorfold:\n"
        f"    baseUrl: {_yaml(c.base + '/v1')}\n    api: openai-completions\n    apiKey: {_yaml(c.key)}\n    models:\n"
        f"      - id: {_yaml(c.model)}\n        name: {_yaml(c.model + ' (TensorFold)')}\n"
        f"        contextWindow: {c.window or 65536}\n        maxTokens: 8192\n\n"
        f"# ~/.omp/agent/config.yml\nmodelRoles:\n  default: {_yaml('tensorfold/' + c.model)}"),
    "openfox": lambda c: json.dumps(
        {"providers": [{"id": "tensorfold", "name": "TensorFold", "url": f"{c.base}/v1", "backend": "vllm",
                        "apiKey": c.key, "isLocal": not c.away,
                        "models": [{"id": c.model, "contextWindow": c.window or 65536}], "isActive": True}],
         "defaultModelSelection": f"tensorfold/{c.model}"}, indent=2),
    "cursor": lambda c: (
        "# Cursor cannot use this AI, here or from another computer.\n"
        "# Cursor sends every request from its own servers (Cursor's cloud), even with\n"
        "# Override OpenAI Base URL: they cannot reach an AI on a private network.\n"
        "# It would take a public HTTPS address for this machine, which opens the AI to the internet.\n"
        "# For the same work in an editor, take Cline (VS Code), or a coding agent in the terminal."),
    "openclaw": lambda c: (
        "// ~/.openclaw/openclaw.json (JSON5), merged with what is there\n{\n"
        f"  agents: {{ defaults: {{ model: {{ primary: \"tensorfold/{c.model}\" }} }} }},\n"
        "  models: {\n    providers: {\n      tensorfold: {\n"
        f"        baseUrl: \"{c.base}/v1\",\n        apiKey: \"{c.key}\",\n        api: \"openai-completions\",\n"
        f"        models: [{{ id: \"{c.model}\", name: \"{c.model} (TensorFold)\", "
        f"contextWindow: {c.window or 65536}, maxTokens: 8192 }}]\n"
        "      }\n    }\n  }\n}\n// then: openclaw agent --local -m \"Hello\", or its gateway as usual"),
    "hermes": lambda c: (
        "# ~/.hermes/config.yaml\nmodel:\n"
        f"  default: {c.model}\n  provider: custom\n  base_url: {c.base}/v1\n"
        f"  api_key: {'' if c.key == 'local' else c.key}\n  context_length: {c.window or 65536}\n"
        "# then: hermes   (or hermes -z \"a one-shot task\")"),
    "python": lambda c: (
        "from openai import OpenAI\n"
        f"client = OpenAI(base_url={json.dumps(c.base + '/v1')}, api_key={json.dumps(c.key)})\n"
        f"r = client.chat.completions.create(model={json.dumps(c.model)},\n"
        "    messages=[{\"role\": \"user\", \"content\": \"Hello\"}])\nprint(r.choices[0].message.content)"),
}

# what kind of thing each tool is, said on its card: where it runs decides the address it is given
FORMS = {"open-webui": "web app in a container", "librechat": "web app in containers",
         "anythingllm": "desktop app or container", "jan": "desktop app", "cline": "VS Code extension",
         "pi": "command line", "omp": "command line", "openfox": "web app", "cursor": "editor",
         "openclaw": "command line and gateway", "hermes": "command line and gateway",
         "claude-code": "command line", "opencode": "command line", "codex": "command line", "python": "library"}
# what changes with the place: on this machine a container reaches the AI through the host; from another computer
# every tool uses TensorFold Web's address and its key, and nothing changes here
HERE_NOTES = {
    "librechat": "On this machine LibreChat's containers reach the AI through the host (host.docker.internal): let "
                 "the AI answer on the local network (at the bottom of this page) and use its key.",
    "anythingllm": "In a container on this machine it reaches the AI through the host (host.docker.internal); the "
                   "desktop app uses the address on this computer instead.",
}
AWAY_NOTES = {
    "librechat": "Its containers on that computer reach TensorFold Web's address directly.",
    "anythingllm": "The desktop app or a container on that computer, with the same lines.",
}

RECIPES = [
    Recipe("open-webui", "Open WebUI", "chat", "openai",
           "A chat front end in a container, talking to the engine as an OpenAI connection.",
           ("models", "chat_stream"),
           checked={"date": "2026-10-10", "machine": "DGX Spark node 1 (GX10)",
                    "how": "the compose file from this page, an account, the model listed and a chat answered through "
                           "Open WebUI's API"}),
    Recipe("librechat", "LibreChat", "chat", "openai",
           "A custom endpoint in librechat.yaml.", ("models", "chat_stream"),
           checked={"date": "2026-10-12", "machine": "DGX Spark node 2 (GX10), TensorFold 1.0.7",
                    "how": "LibreChat 0.8.7 and MongoDB in throwaway containers on the host network, this endpoint "
                           "block as is: TensorFold and the model listed, a reply streamed with its thinking folded, "
                           "the title written; removed afterwards (the owner's LibreChat on node 1 untouched)"}),
    Recipe("anythingllm", "AnythingLLM", "chat", "openai",
           "The Generic OpenAI provider, for chat and its document workspaces. New workspaces start in agent mode: "
           "set them to Chat. A reasoning model's thinking shows in the reply unless thinking is off in Settings.",
           ("models", "chat_stream"),
           checked={"date": "2026-10-10", "machine": "DGX Spark node 1 (GX10)",
                    "how": "AnythingLLM in a container (host network), a workspace in chat mode, a reply streamed from "
                           "Nemotron"}),
    Recipe("jan", "Jan", "chat", "openai",
           "An OpenAI-compatible provider. Jan adds a system prompt of its own while its web search is on, and "
           "leaves thinking to the "
           "model's default: the same question gives another reply than in TensorFold Web's Chat unless both send the "
           "same.",
           ("models", "chat_stream"),
           checked={"date": "2026-10-12", "machine": "Windows PC through node 2's TensorFold Web (GX10), TensorFold 1.0.7",
                    "how": "Jan on Windows with the relay's address and key: the model listed and replies streamed; "
                           "its requests captured and replayed, the same reply token for token as the same request "
                           "sent from here"}),
    Recipe("claude-code", "Claude Code", "agent", "anthropic",
           "The Anthropic-compatible endpoint. The attribution header is turned off so the prompt cache survives "
           "between turns. Claude Code's own prompt is about 16k tokens: the engine needs a window of 32k or more "
           "(65,536 is comfortable); the CUDA engine serves 16,384 unless Longest conversation is set in Settings.",
           ("context", "anthropic", "anthropic_tools"), extra={"min_context": 32768},
           checked={"date": "2026-10-12", "machine": "DGX Spark node 2 (GX10), TensorFold 1.0.7",
                    "how": "claude -p (Claude Code 2.1.296), Nemotron with a 65,536-token window: it wrote the file "
                           "asked, 6 s; with the default 16,384 the engine refuses its prompt. Claude Code now warns "
                           "about a model it does not know: CLAUDE_CODE_MAX_CONTEXT_TOKENS gives it the window"}),
    Recipe("opencode", "OpenCode", "agent", "openai", "A provider block in opencode.json (the project's or "
           "~/.config/opencode/). Its prompts need a window of 32k or more.",
           ("models", "chat_stream", "tool_call", "context"), extra={"min_context": 32768},
           checked={"date": "2026-10-12", "machine": "DGX Spark node 2 (GX10), TensorFold 1.0.7",
                    "how": "opencode run 1.19.0 with this provider block, Nemotron with a 65,536-token window: it "
                           "wrote the file asked through its write tool, 15 s (also on the Mac mini M6, 10 Oct)"}),
    Recipe("codex", "Codex CLI", "agent", "responses",
           "Codex speaks only the Responses API (wire_api = \"chat\" is gone), and asks for "
           "include: [\"reasoning.encrypted_content\"] on every request: TensorFold accepts and ignores it from 1.0.9 "
           "(earlier versions refuse it, and Codex stops at its first request). It needs a window of 32k tokens or "
           "more (Longest conversation in Settings). The test sends Codex's own request.",
           ("context", "responses"), extra={"min_context": 32768, "codex_shape": True},
           checked={"not": "Codex itself not run here; its own request (stateless, include "
                           "reasoning.encrypted_content) is answered by TensorFold 1.0.9 on node 2 (13 Oct), where "
                           "1.0.2 to 1.0.7 refused it with a 400"}),
    Recipe("pi", "Pi", "agent", "openai",
           "A small coding agent for the terminal: this provider in ~/.pi/agent/models.json, then pi --model "
           "tensorfold/<the model>. Its prompts need a window of 32k or more.", ("models", "chat_stream", "tool_call", "context"), extra={"min_context": 32768},
           checked={"date": "2026-10-12", "machine": "DGX Spark node 2 (GX10), TensorFold 1.0.7",
                    "how": "pi 1.1.0 -p with this provider, Nemotron with a 65,536-token window: it wrote the file "
                           "asked, 3 s"}),
    Recipe("omp", "OMP (Oh My Pi)", "agent", "openai",
           "Pi's fork with the editor's tools wired in (LSP, debugger): a provider in ~/.omp/agent/models.yml and the "
           "default role in config.yml. It runs on Bun. Its prompts need a window of 32k or more.",
           ("models", "chat_stream", "tool_call", "context"), extra={"min_context": 32768},
           checked={"date": "2026-10-12", "machine": "DGX Spark node 2 (GX10), TensorFold 1.0.7",
                    "how": "omp 18.10.3 -p with this provider, Nemotron with a 65,536-token window: it wrote the "
                           "file asked, 5 s"}),
    Recipe("openfox", "OpenFox", "agent", "openai",
           "A coding agent made for local AIs, in the browser (port 10369): this provider in "
           "~/.config/openfox/config.json (openfox provider add asks the same), then openfox. Name the model as the default, or a session has no model. It starts in Planner "
           "mode, which plans and writes nothing.", ("models", "chat_stream", "tool_call", "context"),
           extra={"min_context": 32768},
           checked={"date": "2026-10-12", "machine": "DGX Spark node 2 (GX10), TensorFold 1.0.7",
                    "how": "OpenFox 2.0.161 with this config: a session planned the task asked (7.6k tokens of its "
                           "prompt, 97 tokens a second)"}),
    Recipe("cline", "Cline", "agent", "openai",
           "VS Code: the OpenAI Compatible provider, for chat, plans and edits in the editor. Its prompts need a "
           "window of 32k or more.",
           ("models", "chat_stream", "tool_call", "context"), extra={"min_context": 32768},
           checked={"not": "an editor extension (VS Code); no editor session on these machines"}),
    Recipe("cursor", "Cursor", "agent", "openai",
           "An editor with its own AI service: it does not reach a private AI, so there is nothing to set.", (),
           checked={"not": "Cursor's requests leave from its own servers, which cannot reach this machine"}),
    Recipe("openclaw", "OpenClaw", "assistant", "openai",
           "A personal agent that acts for you (files, shell, messaging channels through its gateway): a provider in "
           "~/.openclaw/openclaw.json, made its primary model. Its prompts need a window of 32k or more.",
           ("models", "chat_stream", "tool_call", "context"), extra={"min_context": 32768},
           checked={"date": "2026-10-12", "machine": "DGX Spark node 2 (GX10), TensorFold 1.0.7",
                    "how": "openclaw 2026.10.2, agent --local with this provider, Nemotron with a 65,536-token "
                           "window: it wrote the file asked, 25 s"}),
    Recipe("hermes", "Hermes Agent", "assistant", "openai",
           "Nous Research's personal agent, in the terminal or through its messaging gateway, with memory and "
           "skills: a custom endpoint in ~/.hermes/config.yaml. Its prompts need a window of 32k or more.",
           ("models", "chat_stream", "tool_call", "context"), extra={"min_context": 32768},
           checked={"date": "2026-10-12", "machine": "DGX Spark node 2 (GX10), TensorFold 1.0.7",
                    "how": "hermes -z with this endpoint, Nemotron with a 65,536-token window: it wrote the file "
                           "asked and ran it to check, 17 s"}),
    Recipe("python", "Python", "sdk", "openai", "The official OpenAI client, pointed here.", ("models", "chat_stream")),
]


def compose(c: Ctx) -> str:
    """Open WebUI next to the engine, as a compose file: host networking on Linux so it reaches the engine's
    loopback address; Open WebUI on port 3000 so it does not take the engine's. Its project, container and volume
    are named for TensorFold Web: a plain "open-webui" would reuse an existing install's data, and Open WebUI migrates
    a database to its own version at start, which an older Open WebUI then cannot read."""

    return (
        "# Open WebUI for the TensorFold engine on this machine (Linux; on a Mac see the note at the end).\n"
        "# Its own names, so it never starts on another Open WebUI's data (which it would migrate).\n"
        "name: tensorfold-open-webui\n"
        "services:\n  open-webui:\n    image: ghcr.io/open-webui/open-webui:main\n"
        "    container_name: tensorfold-open-webui\n"
        "    network_mode: host\n    restart: unless-stopped\n    environment:\n      PORT: \"3000\"\n"
        f"      OPENAI_API_BASE_URL: {json.dumps(c.base + '/v1')}\n"
        # the key stays out of a file that may be shared: compose reads it from the shell that starts it
        + ("      OPENAI_API_KEY: \"${TENSORFOLD_API_KEY:?export TENSORFOLD_API_KEY first: Connect shows it}\"\n"
           if c.key != "local" else "      OPENAI_API_KEY: \"local\"\n") +
        "      ENABLE_OLLAMA_API: \"false\"\n      WEBUI_AUTH: \"true\"\n    volumes:\n      - data:/app/backend/data\n"
        "volumes:\n  data:\n    name: tensorfold-open-webui\n\n"
        + ("# export TENSORFOLD_API_KEY=<the key Connect shows>, then docker compose up -d; then http://localhost:3000\n"
           "# (the first account made is the admin).\n" if c.key != "local" else
           "# docker compose up -d, then http://localhost:3000 (the first account made is the admin).\n") +
        "# On a Mac (Docker Desktop has no host networking): drop network_mode, add ports: [\"3000:8080\"], drop PORT,\n"
        f"# and use http://host.docker.internal:{c.base.rsplit(':', 1)[-1]}/v1 as OPENAI_API_BASE_URL.\n")


# -- the checks: the calls a tool makes ------------------------------------------------------------------------------

def _headers(key: str, anthropic: bool = False) -> dict:
    if anthropic:
        return {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    return {"Authorization": f"Bearer {key}"} if key and key != "local" else {}


def _stream_text(r: httpx.Response) -> tuple[str, list[str]]:
    """The text and the event types of an SSE stream, OpenAI's or Anthropic's."""

    text, kinds = "", []
    for line in r.iter_lines():
        if line.startswith("event:"):
            kinds.append(line[6:].strip())
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            msg = json.loads(data)
        except ValueError:
            continue
        for ch in msg.get("choices") or []:
            d = ch.get("delta") or {}
            text += (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
        if msg.get("type") == "content_block_delta":
            d = msg.get("delta") or {}
            text += d.get("text") or d.get("thinking") or ""
    return text, kinds


def run_check(check: str, c: Ctx, client: httpx.Client, extra: dict | None = None) -> dict:
    """One check, as the tool would make it; passed or failed, with the reason in words."""

    t0 = time.perf_counter()
    base = c.base.rstrip("/")
    try:
        if check == "models":
            r = client.get(base + "/v1/models", headers=_headers(c.key))
            if r.status_code == 401:
                return _out(check, False, "the engine wants a key this recipe does not give", t0)
            r.raise_for_status()
            ids = [m.get("id") for m in r.json().get("data", [])]
            ok = c.model in ids
            return _out(check, ok, f"lists {', '.join(ids) or 'nothing'}" if not ok else f"{c.model} is listed", t0)
        if check == "context":
            need = int((extra or {}).get("min_context", 16384))
            r = client.get(base + "/v1/models", headers=_headers(c.key))
            r.raise_for_status()
            window = _window(client, base, c, r.json())
            if not window:
                return _out(check, True, "the engine does not say its window; not checked", t0, skipped=True)
            ok = int(window) >= need
            return _out(check, ok, f"{int(window):,} tokens served" + ("" if ok else f"; this tool needs {need:,} or more: "
                        "set Longest conversation in Settings"), t0)
        if check == "chat_stream":
            body = {"model": c.model, "stream": True, "max_tokens": 256,
                    "messages": [{"role": "user", "content": "Reply with the word ready."}]}
            with client.stream("POST", base + "/v1/chat/completions", json=body, headers=_headers(c.key)) as r:
                if r.status_code != 200:                 # at most 4 KB and 10 s of it: an error that trickles ends here
                    return _out(check, False, f"{r.status_code}: {httpread.error_text(r)[:160]}", t0)
                text, _ = _stream_text(r)
            return _out(check, bool(text.strip()), f"{len(text)} characters streamed" if text else "an empty reply", t0)
        if check == "tool_call":
            body = {"model": c.model, "max_tokens": 1024, "tool_choice": "auto",
                    "tools": [{"type": "function", "function": WEATHER_TOOL}],
                    "messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]}
            r = client.post(base + "/v1/chat/completions", json=body, headers=_headers(c.key))
            if r.status_code != 200:
                return _out(check, False, f"{r.status_code}: {r.text[:160]}", t0)
            msg = (r.json().get("choices") or [{}])[0].get("message") or {}
            calls = msg.get("tool_calls") or []
            if not calls:
                return _out(check, False, "the reply named no tool: " + (msg.get("content") or "")[:120], t0)
            fn = calls[0].get("function") or {}
            return _out(check, fn.get("name") == "get_weather",
                        f"{fn.get('name')}({fn.get('arguments')})"[:160], t0)
        if check == "anthropic":
            body = {"model": c.model, "max_tokens": 256, "stream": True,
                    "messages": [{"role": "user", "content": "Reply with the word ready."}]}
            with client.stream("POST", base + "/v1/messages", json=body, headers=_headers(c.key, anthropic=True)) as r:
                if r.status_code != 200:                 # at most 4 KB and 10 s of it: an error that trickles ends here
                    return _out(check, False, f"{r.status_code}: {httpread.error_text(r)[:160]}", t0)
                text, kinds = _stream_text(r)
            ok = "message_start" in kinds and "message_stop" in kinds and bool(text.strip())
            return _out(check, ok, f"{len(text)} characters, events {', '.join(dict.fromkeys(kinds))}"[:200], t0)
        if check == "anthropic_tools":
            body = {"model": c.model, "max_tokens": 1024,
                    "tools": [{"name": WEATHER_TOOL["name"], "description": WEATHER_TOOL["description"],
                               "input_schema": WEATHER_TOOL["parameters"]}],
                    "messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}]}
            r = client.post(base + "/v1/messages", json=body, headers=_headers(c.key, anthropic=True))
            if r.status_code != 200:
                return _out(check, False, f"{r.status_code}: {r.text[:160]}", t0)
            uses = [b for b in r.json().get("content", []) if b.get("type") == "tool_use"]
            if not uses:
                return _out(check, False, "no tool_use block in the reply", t0)
            return _out(check, uses[0].get("name") == "get_weather",
                        f"{uses[0].get('name')}({json.dumps(uses[0].get('input'))})"[:160], t0)
        if check == "responses":
            body = {"model": c.model, "input": "Reply with the word ready.", "max_output_tokens": 256,
                    # as Codex asks (0.162): stateless, with the reasoning carried back encrypted
                    **({"store": False, "include": ["reasoning.encrypted_content"]}
                       if (extra or {}).get("codex_shape") else {})}
            r = client.post(base + "/v1/responses", json=body, headers=_headers(c.key))
            if r.status_code != 200:
                return _out(check, False, f"{r.status_code}: {r.text[:160]}", t0)
            d = r.json()
            text = d.get("output_text") or "".join(
                p.get("text", "") for o in d.get("output", []) for p in (o.get("content") or []) if isinstance(p, dict))
            return _out(check, bool(text.strip()),
                        f"{len(text)} characters" if text else "no output text", t0)
    except httpx.HTTPError as exc:
        return _out(check, False, f"nothing answers: {exc}"[:200], t0)
    except ValueError as exc:
        return _out(check, False, f"not the expected JSON: {exc}"[:200], t0)
    return _out(check, False, "unknown check", t0)


def _out(check: str, ok: bool, detail: str, t0: float, *, skipped: bool = False) -> dict:
    """One check's result; ``skipped``: it could not be made, which is neither a pass nor a failure (the test as a
    whole is not failed by it, and the page does not tick it)."""

    return {"id": check, "label": CHECK_LABELS.get(check, check), "ok": bool(ok), "skipped": skipped, "detail": detail,
            "seconds": round(time.perf_counter() - t0, 2)}


def _window(client: httpx.Client, base: str, c: Ctx, models: dict) -> int | None:
    """The window the engine serves ``c.model`` with: its /v1/models row, else what /tokenize says."""

    row = next((m for m in models.get("data", []) if m.get("id") == c.model), {})
    window = row.get("context_length") or row.get("max_model_len")
    if not window:
        t = client.post(base + "/tokenize", json={"model": c.model, "prompt": ""}, headers=_headers(c.key))
        window = t.json().get("max_model_len") if t.status_code == 200 else None
    return int(window) if window else None


def served_window(c: Ctx) -> int | None:
    """The engine's window now, for the recipes and the page; None when it does not say or cannot be reached."""

    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(c.base + "/v1/models", headers=_headers(c.key))
            r.raise_for_status()
            return _window(client, c.base, c, r.json())
    except (httpx.HTTPError, ValueError):
        return None


def test(recipe: Recipe, c: Ctx) -> dict:
    with httpx.Client(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
        results = [run_check(k, c, client, recipe.extra) for k in recipe.checks]
    return {"recipe": recipe.id, "ok": all(r["ok"] for r in results), "checks": results, "at": time.time()}
