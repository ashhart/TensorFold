# App workflow research and implementation decisions

## Messaging and mail follow-up, 28 October 2026

This pass reviewed OpenClaw and Hermes as reference designs. TensorFold uses its own host adapters and Assistant inbox. It does not embed or run either gateway, and no reference code was copied into the app.

Pinned references:

- [OpenClaw iMessage](https://github.com/openclaw/openclaw/blob/b0966db12046d114774506425aab1e6ae29755e6/docs/channels/imessage.md) and [BlueBubbles migration announcement](https://github.com/openclaw/openclaw/blob/b0966db12046d114774506425aab1e6ae29755e6/docs/announcements/bluebubbles-imessage.md). This revision uses `imsg` on a signed-in Mac and has removed its BlueBubbles adapter. Its account routing, recovery and capability-specific controls are useful ideas. A Mac remains necessary for iMessage.
- [Hermes BlueBubbles](https://github.com/NousResearch/hermes-agent/blob/0431192d69d4afd2443a7da15bb9e129a424414c/website/docs/user-guide/messaging/bluebubbles.md), [WhatsApp](https://github.com/NousResearch/hermes-agent/blob/0431192d69d4afd2443a7da15bb9e129a424414c/website/docs/user-guide/messaging/whatsapp.md) and [email](https://github.com/NousResearch/hermes-agent/blob/0431192d69d4afd2443a7da15bb9e129a424414c/website/docs/user-guide/messaging/email.md). Separate accounts, explicit sender grants, loop prevention and reconnect visibility fit TensorFold. Its gateway accepts commands from authorized messaging users; TensorFold's personal inbox continues to treat received text as untrusted context rather than command authorization.
- [BlueBubbles API](https://docs.bluebubbles.app/server/developer-guides/rest-api-and-webhooks), checked against server source `76d2b2e786c04d7516de089f31be978a68da8083`. The new adapter was written against the public protocol. It polls only explicit chat GUIDs and needs no public inbound webhook.
- [Microsoft device sign-in](https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-device-code) and [Microsoft Graph mail](https://learn.microsoft.com/en-us/graph/api/user-list-messages?view=graph-rest-1.0). Modern delegated auth supplies a portable Outlook path without guessing another application's private database schema.

The previous integration was incomplete. The Outlook connector explicitly documented its guessed Mac-only database schema. It was not a portable Microsoft account connection. iMessage had a native Mac read/send path but no bridge for Windows or Linux. The reviewed-send validator also rejected existing iMessage groups, despite the Mac sender supporting them. A red regression captured that failure before the validator was corrected.

| Service | Local app implementation after this pass | Prerequisites and limits |
| --- | --- | --- |
| Gmail and Google Calendar | Existing account-scoped OAuth tools, explicit reads, local drafts and exact owner action reviews in Assistant's Google Workspace tab | User must configure a Desktop OAuth client and authorize the account. Live-account sign-in is still unverified. The local follow-up adds explicit per-account Inbox selection, bounded metadata sync, native thread reads and reviewed threaded replies. Attachments remain metadata only; live-account qualification is still pending. |
| Outlook / Microsoft 365 | New device sign-in and protected host tokens, Microsoft Graph Inbox reads, full text reads, native reply endpoint and reviewed sending through the existing Email/Assistant path | User supplies a registered public-client application ID. Delegated User.Read/Mail.Read, offline access and optional Mail.Send. Organization policy can require admin consent. No live Microsoft account tested. Recent Inbox text is supported; attachments, other folders, Microsoft calendar and contacts remain future work. |
| Direct iMessage | Existing read-only Mac Messages database and AppleScript text send; reviewed existing-group replies now pass | Signed-in Mac, Full Disk Access for reads, Messages Automation permission for sends. Native window/permission testing remains pending. |
| BlueBubbles / iMessage | New shared adapter, exact allowed chat GUIDs, selected text reads and approved text replies from any desktop OS | A separate reachable signed-in Mac running BlueBubbles Server is required. Text only. No private API enablement, SIP changes, media, reactions, edits or unsend. Polling is explicit through Inbox sync. |
| WhatsApp Linked Devices | Existing shared Baileys adapter, multiple protected accounts, explicit contact/group grants, media ingestion, QR expiry/reconnect state and local remove/disconnect | Phone linking required. The linked-device route uses an unofficial protocol. Native Windows qualification and live phone pairing/delivery are still unverified. |
| WhatsApp Business Cloud | Existing separate signed ingress and account-scoped route | Meta setup, verified public webhook and explicit sender scope required. Not exercised against a live business account. |
| Other connections | Existing Slack, Discord, Telegram, Matrix, Mattermost, Google Chat, documents and selected browser context | Their presence is not evidence that a user's service configuration works. This pass does not claim live validation of these accounts. |

Assistant now has a Connections tab leading to each service. Microsoft sign-in defaults to read-only, never displays tokens, and requires a separate exact-content owner review before sending. Graph's HTTP 202 is shown as queued rather than recipient delivery. Tokens are refreshed serially; canceled sign-ins and disconnects cannot resurrect an account. Off-origin continuations, redirects and automatic retries of ambiguous sends are refused. Credentials remain in the existing protected local storage and never enter renderer snapshots.

BlueBubbles credentials are encrypted, masked in settings and bound to the saved origin. Conversation keys and approval digests include a host-authored connection identity, so changing servers, passwords or scope invalidates an old conversation's send approval. Incoming text never grants command authority. The host checks selected scope on read, sync and send.

The browser journey also reproduced a first-navigation failure: one-shot Settings requests could be consumed by a discarded React render. Selection is now read without consuming during render and consumed after mount. The actual first click from Assistant to Outlook and iMessage is covered in the browser journey.

Validation receipts are kept outside the repository. Browser fixtures exercise all three platform views with synthetic accounts and no external writes. Protocol tests exercise a real loopback BlueBubbles fixture and fixed-origin Microsoft response fixtures; they do not establish live account delivery. Development packages retain app version 1.0.0-beta.10 and engine 1.1.6. Nothing in this pass is published. The unresolved Full-mode Mac window flashing is unchanged.


## Follow-up review, 28 October 2026

This pass reviews the shared Mac, Windows and Linux application against complete user journeys.
The observations below combine TensorFold's actual renderer and host tests with official product
documentation and selected public bug reports. They are not a user survey, a competitor usability
trial or evidence that one product is universally better. The three TensorFold packages share the
renderer; native permission, process and GPU behavior need separate platform receipts.

### Workflow references

| Product | Useful documented behavior | TensorFold decision |
| --- | --- | --- |
| [LM Studio runtimes](https://lmstudio.ai/docs/cli/runtime/runtime) | Install and select an inference runtime independently of the app. | Keep the new signed runtime manager and explicit rollback. Publishing a runtime catalog is a separate release action. |
| [Codex engineering workflows](https://developers.openai.com/blog/mastering-codex-remote-for-engineering) | Make scope, approvals, review and recovery accessible while work continues on its host. | Keep the existing project/worktree controls and reuse Work activity in Assistant Overview. |
| [Claude Google Workspace](https://support.claude.com/en/articles/10166901-use-google-workspace-connectors) | Mail, calendar and document work are reachable from a conversation. | Make Gmail and Calendar reachable inside Assistant. Document work and grounded assistant handoffs remain separate follow-up work. |
| [super.engineering notifications](https://super.engineering/docs/notifications-and-approvals/) | Separate work needing a response from ongoing activity, with source context. | Reuse the existing lifecycle-based attention component rather than infer completion from a transcript or spinner. This is the product assumed for the user's “Super” reference. |
| [T3 Code FAQ](https://github.com/pingdotgg/t3code/blob/main/docs/user/faq.md) | A shared interface can use existing agent harnesses. | Keep provider/account/model identity visible and preserve the dedicated Assistant model. Do not add another agent runtime for the UI redesign. |
| [oMLX](https://omlx.ai/) | Local server management, per-model controls and client configuration are central workflows. | Prioritize readiness, memory and recovery clarity. Its published speed and cache claims are not measurements of TensorFold. |

Public issues illustrate failure cases worth testing. They do not establish how common those failures
are. Examples include [T3 reconnect state showing stale Working rows](https://github.com/pingdotgg/t3code/issues/25449),
[LM Studio treating a partial model download as complete](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2538),
and [Hermes losing refreshed Google credentials across SSH sessions](https://github.com/NousResearch/hermes-agent/issues/128233).
These support tests for recovery and truthful state, rather than more unsupported success indicators.

### Hermes Google study

The read-only study pins NousResearch/hermes-agent commit
`8acb7f65801fa7548b223244dee57465adfda8ec`.
[Setup](https://github.com/NousResearch/hermes-agent/blob/8acb7f65801fa7548b223244dee57465adfda8ec/skills/productivity/google-workspace/scripts/setup.py)
distinguishes valid, partial, revoked and invalid-client authorization. It preserves actual granted
scopes when refreshing. Its API wrapper prefers `gws` and has a Python fallback. This can broaden
service coverage but also creates two execution and credential-refresh paths.

The reviewed setup uses a manual localhost redirect and JSON token files. TensorFold retains its
existing PKCE loopback listener, encrypted host store, account revisions and exact owner reviews.
The useful adaptation is explicit recovery guidance and verified permissions. No Hermes code,
skill instructions, credential files or execution backend were imported into the app.

Google's [desktop OAuth contract](https://developers.google.com/identity/protocols/oauth2/native-app)
and [refresh-token policy](https://developers.google.com/identity/protocols/oauth2) remain authoritative.
For an external OAuth project in Testing, broader Workspace scopes can mean seven-day refresh-token
expiry. That can require reconnection even when an implementation is correct; this review does not
establish the user's current project configuration.

### Reproduced problems and local changes

| Finding | User impact | Change and evidence |
| --- | --- | --- |
| A connected Google account had no entry point in Assistant. The legacy inbox snapshot contains other connector sources. | Connecting Google in Settings did not make its mail/calendar tools discoverable in the Assistant workflow. | Add Overview, Inbox and Google Workspace tabs. Reuse one protected Google state controller and the existing account, draft and review components. The real-renderer journey now reaches the connected account. |
| OAuth token HTTP failures discarded the provider error category, and renderer hooks replaced known failures with generic text. | An invalid Desktop client looked like a generic request failure; retrying sign-in did not explain the required fix. | Parse only a bounded, allowlisted token error code. Preserve static client/grant/permission recovery guidance through IPC and renderer hooks. Never display raw server descriptions or arbitrary exceptions. The expired-token host-service test failed before the change and passes afterward. |
| Read panes survived authorization revision changes. | Previously read Gmail content remained visible after reauthorization. | Remount Gmail and Calendar readers by account and revision, while retaining account-scoped draft edits. A browser check caught the old message remaining and now verifies its removal. |
| Assistant work started from the inbox had no separate general request surface. | A task not tied to a message was hard to place, and its model was less visible. | Add an Overview with the chosen Assistant model, a saved general draft, recent work and Work activity. Failed launches retain the request; no connected account data is attached silently. |
| Google copy assumed a Mac even on other desktop hosts. | Windows/Linux instructions named the wrong machine. | Use host-computer wording in Google connection and privacy messages. Add the new authored UI phrases to all twelve bundled language packs. |

Tabs retain mounted draft panes. Visiting another Assistant tab does not discard unsaved Google
email edits or the general request draft. Gmail reads remain explicit. Status reconciliation reads
the local account snapshot, not the remote inbox. Existing host checks still keep Google tool data
on an approved local model and require owner review before external writes.

### User needs to validate

These are product hypotheses to test with users, not measured demand:

| User and job | Observable success criterion | Next priority |
| --- | --- | --- |
| Local-model newcomer wants a first answer. | Find, download, load and answer without mistaking downloaded files for a running model. Recover a failed download without restarting setup. | High: first-answer and interrupted-download tests on each platform. |
| Daily assistant user wants to handle mail and commitments. | Identify the account, read the actual thread/event, draft a grounded response, inspect the exact proposed action and recover from expired consent. | High: message-to-assistant handoff, correctly threaded replies and a grounded daily agenda. |
| Engineer wants a bounded change. | See project, branch/worktree, model and permissions before starting; review the actual diff and test evidence afterward. | High: retain the current worktree/review path and test interruption/resume. |
| Assistant-first user starts in the compact chat layout. | Reach Assistant directly without discovering the Complete workspace transition first. | High: review primary navigation. The native Linux walkthrough required opening Complete workspace before Assistant became reachable. |
| Background-work user needs the next decision. | One attention entry identifies approval, question, failure or uncertain outcome and opens the relevant work. | High: exercise real stalled/reconnected lifecycle events, not only fixtures. |
| Privacy-conscious user mixes local and hosted models. | Know the data route; a model change cannot move Google history to a cloud, CLI, LAN or remote run. | Release gate: keep the existing host-enforced privacy and account-bound review tests. |
| Windows/Linux user expects the same core workflow. | Unsupported OS integrations have a clear explanation; common controls and keyboard shortcuts match the OS. | Release gate: native Windows qualification, command/MCP isolation, desktop/OCR and VM backends remain incomplete. |

Recommended research is five task-based sessions spanning these workflows, with consented observation
of task completion, recovery, account mistakes and lost drafts. Do not collect mailbox contents,
prompts, credentials or model files as product telemetry. Ask which concrete jobs people repeat
before expanding integrations to Drive, Docs or additional write scopes.

The Windows VM and real Google account qualification are still outstanding. Synthetic fixtures and
local host-service tests cannot prove that a live Google Cloud client, consent audience or account
policy is configured correctly. The original Full-mode Mac close flashing is also still unresolved.
The native Mac GUI check was blocked by the locked Studio desktop; packaged headless startup passed.
This pass does not benchmark assistant reasoning quality or prove a live mail/calendar task completes.

## Earlier review, 26 October 2026

Reviewed on **2026-10-26** against TensorFold commit `c9d9f005c48ed9f88f5085d2e8df50a3a1ed5ea5`.
The sources below describe the products at retrieval time. They are product documentation, not independent performance or reliability measurements.

## Official sources

| Product | Source | Relevant documented behavior |
| --- | --- | --- |
| LM Studio | [Getting started](https://lmstudio.ai/docs/app/basics) | The onboarding path distinguishes finding/downloading a model, loading it into memory, and starting a chat. |
| LM Studio | [Loading local models](https://lmstudio.ai/docs/cli/local-models/load) | Load configuration and estimated memory requirements are explicit. Estimates depend on configuration and are not measured usage. |
| LM Studio | [User or Developer modes](https://lmstudio.ai/docs/app/user-interface/modes) | The interface can expose different levels of detail for different workflows. |
| T3 Code | [User keybindings](https://github.com/pingdotgg/t3code/blob/main/docs/user/keybindings.md) | Composer controls keep model, host, access, workspace and Git branch choices near the request. Workspace choices include the current checkout and worktrees. |
| T3 Code | [Appearance preferences](https://github.com/pingdotgg/t3code/blob/main/docs/user/appearance.md) | Composer context visibility and motion preferences make important context accessible without permanently exposing every control. |
| super.engineering | [Notifications and approvals](https://super.engineering/docs/notifications-and-approvals/) | The inbox separates ongoing work from work needing a response. Rows retain source context, and structured lifecycle signals drive approval/question/error state. |
| super.engineering | [Review changes](https://super.engineering/docs/review-changes/) | The review workflow keeps branch/target context, changed files and follow-up review work together. |
| super.engineering | [Git actions and pull requests](https://super.engineering/docs/pull-requests-and-review/) | Available Git actions depend on the current repository state. A workflow shortcut is not evidence that an implementation passed its checks. |

## What TensorFold already implements

The app already has model discovery, downloads, load configuration and preflight checks, local runtime supervision, chat/task layouts, worktrees, change review, a command centre, and a connected-source assistant. Adding parallel replacements would increase the navigation burden.

The implementation opportunities are the connections between those capabilities:

| Observation in TensorFold | User impact | Implemented response |
| --- | --- | --- |
| `Assistant.tsx` initially fetched its private snapshot on mount and after user actions. Existing subscriptions did not refresh that snapshot when new inbox data arrived. | A visible inbox could remain old until the user refreshed or left and returned. | Extract a read-only snapshot controller/hook, reconcile while visible and on return, retain usable data on failure, and show refresh freshness and retry state. |
| The compact work menu lived inside `ChatWorkspaceHeader.tsx`, while standard workspace navigation had no equivalent shared attention surface. Its generic status labels did not distinguish questions or uncertain run outcomes. | Attention was harder to find across layouts, and the user had less information about why to open an item. | Reuse one work activity component in both layouts and derive reasons from the existing store. |
| Model files, loaded state and engine health are available in existing model/runtime interfaces. | A user needs a clear next action when a model is downloaded but the runtime is unavailable or not ready. | Add a concise runtime summary to the existing model page; retain the existing detailed controls and measured/estimated distinctions. |
| Task worktree and review information already exists in thread and panel APIs. | Repository scope and the next review action can be less visible than the conversation itself. | Put relevant task context and review entry points next to the active task without changing repository or approval semantics. |
| Download recovery and runtime load cancellation were buried in detailed controls; starting a new chat could apply a different saved surface model. | Setup could strand a user or silently replace their explicit local-model selection. | Inline download/load recovery and an explicit-model chat transition, including the onboarding return path. |
| The empty task surface left project selection in overflow and gave optional starters more prominence than the working scope. | Starting a scoped task required exploring unrelated controls. | A focused New task surface with project/folder/branch beside the composer, readable model selection and recent-task continuity. |
| Transcript reading position and expanded history were lost when a view unmounted; approvals could force a scroll. | Users lost their place during long work. | Bounded session memory for position/ranges, anchored restoration, explicit Jump to latest and no forced approval scrolling. |
| Tool logs and diffs had weak failure/provenance context and awkward long-line handling. | Review required too much searching and horizontal scrolling. | Grounded terminal states, copy/wrap/expand tools, changed-file filtering, progressive diff rendering and retained-data retry. |

## Decisions for this branch

### Connected-source assistant freshness

`src/components/assistant/useAssistantSnapshot.ts` reads the existing `getAssistantSnapshot` API. It reconciles every 15 seconds after a settled read while the document is visible, and when visibility, focus or connectivity returns. It does not call connector synchronization APIs automatically.

`snapshot-refresh.ts` coalesces concurrent background reads. An explicit refresh after a mutation requests one trailing read, so an older in-flight snapshot cannot immediately overwrite the action's outcome. A failed read retains the last good snapshot and its timestamp. A 20-second logical deadline unlocks retry if the bridge does not settle; an eventual expired response is ignored. Unmounting removes timers and listeners and suppresses late updates.

The header distinguishes an update in progress, the last successful read and a delayed update. A separate error message explains when retained data is being shown. The existing sync and outbound approval controls retain their meanings.

### Inbox reading, drafting and review

The Assistant now makes its existing journey explicit: choose a source or search a conversation,
read the selected conversation, then prepare a reply or start assistant work. The conversation rail
supports keyboard selection and an actual pending-review filter. A compact native selector replaces
the rail at narrower widths. The reading header keeps the subject and contact visible even when
the latest event is an outgoing draft.

The composer has separate **Reply** and **Ask TensorFold** intents, with one corresponding submit
action. Preparing a briefing or suggested task selects assistant intent. Replacing an existing,
different draft requires confirmation; cancelling preserves the text and its current intent.
The existing account-scoped draft storage and external-send confirmation remain in force.
Browser session-only draft storage is presented as a capability note, while actual save failures
retain an error and retry action.

The pending approval queue initially shows two actions and can expand within a bounded scrolling
area. Outbound review still uses the exact host-sealed account, recipient, body and digest. Opening
review never sends; a failed attempt leaves the review available. The modal traps Tab, supports
Escape while idle, and restores focus to its trigger when that trigger still exists.

Loading, a failed initial read, a failed refresh with retained data, an empty inbox and an empty
search are separate states. Source synchronization is an explicit secondary action, with labels
that name the currently supported email/iMessage sync path. Refresh only checks the inbox snapshot.

### Shared work attention

`src/components/work-activity/WorkActivityMenu.tsx` uses the existing store and `src/lib/full-workspace-activity.ts`. It has two groups: **Needs you** and **In progress**. Rows explain whether the recorded cause is approval, a pending question, an outcome review, an error, a stalled job, a queue or a wait.

Rows retain task/chat/room/routine identity and project names where available. Duplicate thread/job or room/job projections merge without losing an actionable reason. A handed-off job routes to its current thread; room supervision issues route to the owning room. Approvals in a worker thread can open that specific thread. Archived projections are excluded.

The compact list initially limits each group to 12 rows and can expand in place. This preserves access to pending questions and outcome reviews that are not necessarily listed by the command centre. Opening a row only navigates. It does not grant approval, dismiss a request, stop work or manufacture a completion receipt.

### Context and model clarity

The branch uses the existing design tokens, navigation and APIs. Task context is grounded in the selected thread/workspace. Runtime state is grounded in the host snapshot. Action labels must describe what the existing handler actually does; an engine being online is distinct from a model being available for inference, and a completed run is distinct from verified correctness.

### Composer maintenance

The composer has been split by responsibility while retaining its send/approval guards. `Composer.tsx` retains acceptance, attachment, queue, quote and keyboard behavior. `useComposerWorkspace.tsx` owns workspace selection and stale-response fences, `useComposerModel.ts` owns model context and readiness, `useComposerInput.ts` owns insertion and input lifecycles, and `ComposerToolbar.tsx` presents secondary controls. Project selection is now visible beside a new task; secondary controls use progressive disclosure. Each module is below the repository's 600-line limit. Existing tests still execute the production send/queue/attachment callbacks; the model-fit integration test loads the extracted hooks and toolbar.

## Independent implementation and limits

The official documentation informed workflow questions and information hierarchy. No competitor source code, screenshots, icons, assets or page layouts were copied into TensorFold. The changes are original implementations using TensorFold's own components and data contracts.

This scope does not add a new notification persistence service, invent a completion history, ship unwired experiments, or replace the current provider, permission or worktree machinery. Native runtime qualification, model quality and signed packaging require their existing platform-specific gates.

Focused regression coverage exercises concurrent and failed snapshot reads, stale-response fencing, visible/focus/network reconciliation, cleanup, retained drafts, truthful freshness, attention grouping, deduplication and source navigation. The branch-level validation receipt records the broader build and test results separately.
