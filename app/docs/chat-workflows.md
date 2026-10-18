# Conversation and document workflows

These additions use the existing protected profile, conversation storage and semantic UI components, with no new runtime dependency.

- Quote visible user or assistant text into the same local API conversation, edit or remove the draft passage, and jump to its exact source message. A source hash is checked again before sending; changed or removed sources refuse the send and retain the draft. Quoted text is JSON context data, with user edits marked, and is stored alongside the visible user message and matching model history.
- Edit a waiting follow-up with Save and Cancel. Saving preserves its position and never sends it. An open editor pauses automatic draining; Stop or switching draft owners clears the session queue, while unfinished composer text remains separately saved. Duplicate clicks and late replies cannot target another chat.
- Choose Enter or Command/Control+Enter in Preferences. Shift+Enter remains a newline, including with mention or skill suggestions open; IME confirmation is not a send gesture.
- Filter chat search by local-calendar updated dates, current provider/model and attached files. These are current conversation metadata, not claims about historical model use.
- Open Outline or use Command/Control+Shift+O for exact message jumps, pinned decisions and recorded work outcomes. It scans at most 50,000 items, paginates and discloses truncation; active streamed answers remain excluded unless explicitly pinned.
- Recall recent direct prompts from the same workspace and model account, append them without replacing a draft, and clear their listing without deleting chats. Private, external, connector, bot, phone, paired and remote inputs, file payloads and quoted context are excluded. Prompt text is projected from existing protected conversations; only hashed scope/cutoff metadata is added to storage.
- Select document collections per chat. New chats start with none; older unspecified policies visibly retain their enabled collections until explicitly saved. Imported records start with none. Host validation, tool retrieval, cached connector excerpts and helper search enforce the same live allowlist.
- Process documents with a protected per-file queue, one extraction at a time, independent errors and retry, pause, resume and cancel. Restart requires explicit continuation; in-flight results are discarded when paused, cancelled or their source changes.
- View the checked original revision and extracted text, parser, truncation and cited passage. HTML is escaped. PDF image/page rendering and exact OCR overlays are not provided; uncertain PDF extraction locations are identified rather than invented.
- Expand the live context estimate to inspect history, instructions, draft, files, quotes and the reported output reserve. Character estimates are approximate; unknown image costs, missing limits and changing history are disclosed and do not guarantee engine admission.

## Boundaries and checks

Quotation editing changes its draft snapshot, not the original source. Editing a sent message retains its quotation snapshot and rebuilds matching model content through the existing explicit history-edit flow. Separate CLI-owned sessions and remote quotations are refused rather than silently dropping context. Selecting text across rendered Markdown decorations can fail exact-source validation; quoting the whole message or an exact passage remains available within the 8,000-character passage limit.

The regression suite exercises production handlers and components, protected storage and fresh-process draft recovery, fake provider requests, stale owner/selection changes, queue restart/cancellation, malformed sources, refusals and portable restore. GUI qualification uses synthetic data and transports; real accounts and local models are not required for these checks.
