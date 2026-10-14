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
