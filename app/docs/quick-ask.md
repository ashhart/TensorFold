# Ask TensorFold

Open Ask TensorFold from the workspace header, the command palette, Runtime settings or the system tray.
Its settings enable an optional global shortcut. The default is Command/Ctrl+Shift+Space.
The shortcut recorder temporarily releases TensorFold's binding and reports conflicts without replacing a working binding.
Enabling it keeps TensorFold available in the tray after the main window closes.

The floating window shares the app's colours, display preferences and bundled languages.
It keeps an encrypted scratch conversation in the active local profile.
Hide, Escape and the global shortcut preserve the draft. Quit waits for the renderer to flush it.
Starting a new ask clears the scratch history and keeps any unfinished question.

## Models and context

Choose a configured chat-completions provider, account and model, or an installed TensorFold model.
Delegated coding CLIs are excluded because this window has no workspace or coding permissions.
Loading a model is a separate explicit action. Asking never starts or switches the built-in engine automatically.
Model, account, instructions and destination stay pinned for the conversation.
A changed endpoint or host connection requires a new ask before history can be shared with it.

Attach clipboard text, images, text files or a display screenshot explicitly.
Attachments have previews and remove controls. Clipboard reads and screen capture never run on opening the window.
Unverified image support is labelled; models known to reject images cannot receive them.
External and remote requests require a destination review before each turn.

Presets have editable names, slash commands, instructions and optional model choices.
Changing presets requires a new ask. Web research is opt-in and honours disabled capabilities.
Only bounded web search and public-page reads are available. Default search covers news.
There is no shell, MCP, messaging, account connector or general desktop-control bridge in this window.

## Remote hosts and handoff

Enabled, configured TensorFold hosts publish model choices over authenticated owner connections.
Both clients and hosts need this version. Streams use bounded NDJSON, cancellation and completion receipts.
Disconnecting or revoking the connection stops generation. Host settings and credentials stay in the main process.
The remote API cannot capture the host screen or read its clipboard.

Continue in TensorFold saves an ordinary conversation with the model history and visible partial answers.
The handoff also saves an unfinished draft before opening the main window.
Retries do not overwrite a newer full-app draft. Unsent attachments stay on the client when handing off a remote conversation.
File payloads remain available in the current renderer session; durable composer references require reselection after restart.

## Speech and platform qualification

Dictation uses the configured system speech service when the desktop runtime supports it.
The question box also accepts the operating system's own dictation shortcut.
Read aloud uses configured local speech or an available offline system voice.
This is turn-based dictation and playback, not a realtime voice conversation.

The shared renderer and host paths build for macOS, Windows and Linux.
Linux uses a stable desktop identity for the Wayland global-shortcut portal.
Compositor shortcut consent, screenshot selection, native focus and speech availability need testing on each target desktop.
Browser tests with platform configuration values do not establish native Windows or Linux qualification.

## Verification

The agent suite covers payload bounds, destination pins, explicit completion, tool restrictions, cancellation,
encrypted-state handoff, draft retries, sender identity, window sandboxing and shortcut lifecycle.
The QA scripts under `scripts/qa/quick-ask-*.mjs` cover the browser journey, isolated native Electron round trip
and a real CPU model through the production generation and model transport.
