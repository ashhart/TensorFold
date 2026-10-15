# Bots

Open **Bots** in the sidebar, choose a starting point, and save a name, responsibility, instructions and model.
The built-in TensorFold model appears first when it is available.
Bots reuse the app's saved agent presets and ordinary encrypted chat history.

**Read only** is the default and refuses actions that would change files or connected services.
**Review actions with workspace rules** uses TensorFold's existing approval rules, including approvals you previously saved.
Instructions do not grant permissions.
Keep passwords and API keys in their dedicated Settings fields.

Each new bot chat captures its model, model account, responsibility, instructions, permissions and execution target.
Editing a bot affects new chats.
Forking a bot conversation without a worktree keeps an independent copy of its captured model, tool selection and permissions.
Bot conversations cannot fork into host worktrees or hand off to a different engine; edit the bot and start a new chat to change those choices.
All bots also refuse child agents and linked missions because those runtimes do not yet inherit the bot's captured model, approvals and tool selection.
Existing chats keep their captured configuration; they refuse to silently switch to another model, account or execution target when the saved one becomes unavailable.
The primary Open chat action returns to the most recent conversation with a matching captured configuration; New chat remains available separately.
Pause stops active work and blocks further turns until you resume the bot.
Archive stops its work and keeps its chat history and VM files; restore it from the Archived filter.

## Execution

**This Mac** uses the chat tools and sources configured in TensorFold.
External API models follow the app's planning-only policy.
Third-party coding CLIs cannot be selected for bots because their internal tools cannot enforce the bot's execution boundary.

**Separate Linux VM** uses a private guest workspace for that bot's files and commands.
It requires a supported Apple silicon Mac, the verified Apple container runtime, and explicit VM setup from the bot card.
The model must run on this Mac; turn Cortheon off before starting the VM bot.
Guest networking is disabled, and host files, accounts, browsers, phones, plugins, connections and delegation tools are unavailable inside the VM run.
TensorFold refuses to fall back to host execution when the VM is unavailable.
One chat at a time owns each bot VM, including during startup and cancellation.
Rebuilding the guest image preserves its private `/workspace` files.

The desktop app and local model must remain running for bot work to continue.
Saving a bot does not create an always-on cloud service or a recurring automation.

## Selected tools

The optional **Tool access** section lets you capture exact app and connected MCP tool names for a bot.
Turning it on starts with no tools selected; the normal creator stays read only by default.
Select the tools the bot needs, then start a new chat.
The model's schemas, direct executor and deferred MCP catalog all use the captured selection.
The MCP search and call transports expose only selected targets.
Editing a preset does not broaden an existing chat or an in-flight run, and a changed selection starts a separate compatible conversation.
Corrupt or unsupported scope records remain blocked and retain their original saved value for review.
Resetting a damaged selection in the editor starts with no tools and does not rewrite existing chat snapshots.

Tool selection does not grant approvals, enable disabled capabilities, connect a device or bypass the external-model planning policy.
MCP actions still need the normal review rules, even when a server labels a tool read only.
VM execution intersects the selection with its existing guest-only tool list.
Commands and selected MCP servers may perform broader operations; this limits TensorFold dispatch and is not an OS sandbox.

Restricted bots do not use plugin tools or hooks, child agents, alternate model helpers, global orchestration or automatic skill review.
These helpers do not yet inherit a captured scope safely, so they fail closed.
The built-in engine's Cortheon mode must be off before startup and throughout restricted or VM bot work, including custom API aliases of that server.
TensorFold refuses hidden server research instead of changing the global setting automatically.
It also checks research policy before each model request and refuses enabling Cortheon while an isolated or restricted run is active.
VM and restricted bots use a local deterministic compaction summary, without auxiliary summarizer or verification requests.
Automatic titles and skill self-improvement are disabled for these guarded conversations.
Ordinary chats keep their existing model-assisted compaction behavior.
