# Assistant inbox acceptance receipt

## Connection health and Gmail Inbox, 28 October 2026

Assistant Connections now separates account authorization, read/send permissions and a verified
mailbox read. Explicit tests read at most one message's metadata and persist only a safe result
and timestamps. Stale account checks and failed protected writes cannot record a healthy verdict.

Gmail has explicit per-account Inbox selection, bounded metadata sync, native conversation reads
and reviewed threaded replies. Its cache and drafts remain in protected Google storage; only
owner-only IPC can access them. The renderer merges Gmail with existing Inbox items without
adding Google data to generic connector events, remote views or unrelated model tasks. Removing
and re-adding a source fences late reads, syncs and reply preparation. Same-account reconnects
preserve text drafts while removing old message content. Native send receipts remain one-shot,
and Gmail acceptance is not presented as recipient delivery.

Gmail assistant handoff requires local execution, a local API model and external research off.
The host persists its Google privacy marker before inference. Renderer launch and native remote
admission reject Gmail context before transmitting it to an Orchestrator. The affected remote
controller was split into project, run and room/ticket modules instead of adding to its old
989-line file. All 231 changed or new TypeScript and JavaScript files are at most 593 lines;
the final AST comment audit finds no multiline comments or comments over 120 columns.

The final local verification passes all 906 test files, desktop and mobile typechecking, mobile
security and lint checks, privacy scanning, whitespace checks and production compilation.
The actual renderer passes the new five-part mail journey, all nine existing Inbox journeys,
Google Workspace checks and three-platform connection fixtures, with no page errors. These
journeys include 720 by 520 window checks. Fixture mail is synthetic and no message was sent.

The unsigned Mac development package passes isolated headless startup, UI response and
unauthenticated API refusal. The rebuilt Linux ARM64 Debian app passes native sandboxed startup,
fresh Full-mode onboarding, Assistant Connections and the actual health/Gmail preload against
an empty isolated protected store. Unknown-account operations refuse before provider access.
Both packages retain engine 1.1.6. Test profiles, the browser server and the Linux VM are closed.

Live Google/Microsoft account access and message sending still need qualification with the
intended account configuration. Windows native execution, Linux x64 native execution and the
Mac window check remain pending. Earlier Windows and Linux x64 installers predate this change.
The signed installed Mac app and published releases were not replaced. Nothing was committed,
pushed, merged or deployed.

## Maintenance pass, 28 October 2026

The local refactor splits seven oversized changed modules by responsibility. Connector contracts
now live in channel-specific type modules. Email parsing, source citations and outbound actions
are separate. Codex process transport, protocol sessions and pooling are separate. The engine
coordinator delegates inventory, downloads, feature changes, performance and process records.
Settings components, preload namespaces and browser fixtures have their own modules. Microsoft
account authorization, Graph mail operations and HTTP transport are separate as well.

The read-only structure audit covers all 174 changed or new TypeScript and JavaScript source and
test files. Each is at most 593 lines, with no multiline comments, comments over 120 columns or
trailing whitespace. Comment cleanup preserves the parsed executable code. Public connector
exports and the renderer API contract remain intact.

The actual-renderer Inbox, Google Workspace and three-platform connection fixtures pass after
the split. The engine IPC test now invokes the extracted preload bridge and checks all existing
routes, structured arguments and subscription cleanup. All 897 test files, desktop and mobile
typechecking, mobile checks, privacy scanning, whitespace checks and the production build pass.
The unsigned development Mac package passes isolated startup, UI and authentication checks.
No live account or native window check was performed. Earlier Windows and Linux installers
predate this source refactor.

## Follow-up acceptance, 28 October 2026

The updated inbox retains the nine earlier browser journeys and their minimum-window checks.
`scripts/qa/assistant-google-journey.mjs` adds actual-renderer coverage for the Assistant workspace:

- A connected Google account is reachable from Assistant, without automatically reading Gmail.
- Known Google rate-limit failures retain specific, static recovery guidance.
- A changed authorization revision removes previously read Gmail content.
- Unsaved Google email text survives workspace-tab switches and same-account reauthorization.
- A failed Overview launch retains the general request and its selected Assistant model.
- The Overview launch does not silently include connected account data.
- The 720 by 520 window has no horizontal document overflow.

Both browser suites use clearly scoped synthetic accounts and fixtures. They send no messages or
Google actions. The expired-token host-service regression separately exercises invalid client,
revoked grant and invalid scope responses through the real Google transport and account state.
It asserts both the public error and account recovery state, without displaying server descriptions.

The rebuilt Linux ARM64 Debian app completed fresh Full-mode onboarding and opened the actual
Assistant Overview and Google Workspace panes through its sandboxed preload. Native Google status
reported the expected empty isolated credential store and engine 1.1.6. Linux packaged startup
and the native Node expired-token regression passed. Mac packaged startup passed; native Mac
window inspection was blocked because the Studio desktop was locked. Windows installers were
cross-built locally, with native execution still deferred until the user's VM exists.

No live Google client or account was imported, signed in, disconnected or modified in this pass.
These receipts do not establish the user's current Google Cloud configuration or a real-account
end-to-end result. The original Mac close-window flashing remains unresolved.

## Earlier acceptance, 26 October 2026

Date: 2026-10-26. Scope: the application renderer on the `app-improvements` branch.

The Assistant journey was exercised in a fresh Chromium profile against the running Vite app.
The screenshots use synthetic browser fixtures and are visibly labelled. They demonstrate the
actual application components and event handlers; they do not demonstrate a native connector,
account connection, outbound delivery, or GPU inference. The fixture provides a configured email
source identity so same-account draft continuity is tested without weakening account isolation.

## Acceptance matrix

| Journey | Result | Evidence exercised |
| --- | --- | --- |
| Find a conversation | Pass | Filter by source, search sender/subject/preview, recover from no matches, select with ArrowUp/ArrowDown. |
| Recover a failed refresh | Pass | The real snapshot hook retains the last inbox, selected conversation and exact draft text. Retry clears the failed-read state. |
| Receive live updates while writing | Pass | The unmodified 15-second hook reads a new fixture message; the same conversation, draft text and textarea focus remain. |
| Prepare assistant work | Pass | Cancelling a suggested replacement preserves the draft. Accepting selects Ask TensorFold intent and its task action, instead of a direct reply. |
| Review and retry an approval | Pass | The queue limits its initial rows and expands; the modal shows the exact account, recipient, subject and body. Opening it does not dispatch. Tab stays inside; Escape restores focus. A simulated failure retains the payload; explicit retry passes the exact review digest and records one fixture delivery. |
| Use a smaller dark window | Pass | At 1024×768, the conversation and draft remain actionable without horizontal document overflow. |
| Use the minimum window | Pass | At 720×520, with the sidebar both expanded and collapsed, compact source/conversation selectors and the reply action remain reachable. The reading area stays at least 160 pixels high; vertical scrolling handles constrained height. Draft text remains intact. |
| Recover an initial read | Pass | An unresolved read shows loading, a rejected read shows its diagnostic and Retry inbox, and retry restores the conversation. |
| Recover an empty inbox | Pass | A successful empty result is distinct from loading or error and opens the existing Connectors settings through Manage sources. |

The final run produced nine passing journeys and eleven labelled screenshots, with no browser page
errors. The evidence manifest records each viewport and state. The minimum-size check also verifies
that the send button is within the viewport and receives the pointer at its centre, after the
temporary theme-change notification has disappeared.

## Focused regression coverage

The following existing test entry points passed after the changes:

- `src/components/assistant/snapshot-refresh.test.ts`
- `src/components/assistant/useAssistantSnapshot.test.ts`
- `src/components/assistant/AssistantHeader.test.tsx`
- `src/components/assistant/Composer.test.mjs`
- `src/components/assistant/drafts.test.ts`
- `src/components/assistant/OutboundApprovalDialog.test.tsx`
- `src/components/assistant/message-delivery-status.test.ts`
- `src/components/preferences/language-grammar.test.tsx`

These checks cover coalescing and trailing reads, last-good retention, deadline/late-result fencing,
visible/focus/network reconciliation, unmount cleanup, draft/account boundaries, replacement
confirmation, explicit task-versus-reply keyboard dispatch, IME handling and exact-payload review.
Full TypeScript checking and the affected-files whitespace check also passed. The branch-wide
receipt reports the complete suite and production build separately.

## Reproducing the browser checks

Run `scripts/qa/assistant-journey.mjs` from `app`, with the evidence directory and running development
URL as arguments. The shared browser helper accepts `TENSORFOLD_QA_CHROMIUM` for a Chromium binary
and `TENSORFOLD_QA_PLAYWRIGHT` for a Playwright module path. Use a fresh profile; finish source and
language-pack writes first so Vite does not replace the page during an interaction.

The script installs fixture responses only in that browser page's in-memory API object. The normal
snapshot controller, polling, drafts, rendering, keyboard handlers and approval UI remain active.
No account is signed in and no message leaves the fixture. Generated screenshots and the
`acceptance.json` manifest belong in the operator-supplied evidence directory, outside the source
archive.

## Boundaries

A failed refresh retains an already-loaded conversation and its draft. An unconfigured or changed
account remains a different draft identity by design; the UI does not merge drafts across accounts
to make a continuity check pass. Browser drafts remain session-only, and their neutral notice does
not claim durable persistence. Native connector and host sandbox qualification must be performed
on their supported platforms.

## Messaging and mail follow-up, 28 October 2026

- Check Assistant → Connections → Outlook and iMessage on the first navigation after startup. React rendering must not consume the destination before the screen mounts.
- A Microsoft device-code request starts only on explicit Connect. Read-only is the default, the sign-in URL is fixed, host polling respects the provider interval, and cancellation or disconnect prevents a late token exchange from creating an account.
- Microsoft Inbox/list/read use the existing Email connector and account provenance. Modern sign-in tokens stay in a separate encrypted host collection. A changed authorization revision invalidates any earlier send approval.
- Microsoft sends and replies require durable exact-content approval, never retry automatically, and show HTTP 202 as queued. Replies use the selected account's native Graph reply endpoint.
- BlueBubbles reads only the selected exact chat GUIDs. Empty scope receives nothing. The Mac server prerequisite and text-only support are stated. Server, credential or scope changes invalidate previous conversation identities and approvals.
- Existing allowed iMessage group conversations can pass exact-content review. WhatsApp remote instructions refer to the host computer, and existing QR/account controls remain local on Windows, Linux and Mac.
- All real account credentials and actual external sends remain outside the synthetic acceptance tests. Do not call the integrations live-verified until native sign-in, read, reply, reconnect, restart and disconnect have been tested with appropriate accounts.
