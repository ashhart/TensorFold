# App improvements candidate

This branch continues the supplied TensorFold checkout at
`c9d9f005c48ed9f88f5085d2e8df50a3a1ed5ea5`. It improves the existing desktop app's
model setup, conversation, project/task, review and Assistant journeys. The candidate archive
contains the resulting revision/tree identity, checksums, verification receipts, screenshots,
a binary-safe patch and a Git bundle. Its `START_HERE.md` is the reconstruction entry point.

## Product changes

| Journey | Implemented behavior |
| --- | --- |
| First useful result | Friendly cached-model choices; download progress with cancel/resume/retry; visible load cancellation and recovery; an explicit model selection stays selected through onboarding and the first chat. The setup guide continues after loading from Models. |
| Everyday conversation | A quieter composer keeps the active model readable. Secondary conversation tools are disclosed in one keyboard-accessible menu. Expected browser-only draft limits are distinct from failed persistence. Drafts and attachments survive navigation. |
| Project to task | New task puts the chosen project, folder and branch beside the composer. Task views show their actual workspace and a direct review action. Home scope survives navigation; stale folder/branch/browse results cannot overwrite a newer selection. |
| Long work | Work activity separates actionable Needs you items from ongoing work using existing runtime state. Activity rows distinguish recorded terminal outcomes without claiming that finishing proves correctness. Steering and cancellation keep their existing runtime paths. |
| Review | Changed-file filtering, wrapping, progressive large-diff disclosure, loading/retry and explicit diff provenance make real review easier. Tool output has copy/wrap/expand controls and visible failure state. Existing consequential actions retain their approval and Git semantics. |
| Assistant | Searchable source-filtered conversations, selected-conversation context, explicit Reply/Ask modes, bounded pending approvals and exact-payload review. A visible inbox refreshes automatically, retaining data/drafts on failure with an actionable retry. |

The implementation uses TensorFold's existing brand art, icon system, theme tokens, typography
and production services. TensorFold, Navy and Glass retain their light and dark variants.
The supported minimum window is 720×520; large panes and long transcripts remain scrollable.
Menus and new actions have accessible names, visible focus and keyboard paths. Reduced-motion
preferences suppress decorative spin/pulse motion as well as transcript scrolling animation.

### Reading position and drafts

A bounded renderer-session cache stores transcript row IDs, viewport position and loaded ranges,
never message text. Returning to a conversation restores its visible anchor. Streaming follows
the end only while the reader is already there. Incoming approvals do not move someone reading
older content; the docked approval and Needs you entry remain actionable. Jump to latest opts
back into following. This cache does not claim persistence across an app restart.

Drafts retain the existing persistence service and account/host scoping. In a browser-only
preview, a neutral notice explains tab-scoped storage. A real save failure still exposes Retry.
Suggested Assistant prompts ask before replacing a different draft. An account/source identity
change keeps its security boundary instead of copying private drafts into another scope.

## Reliability foundations

- Required Loop checks are explicit, validated before work starts, and authoritative at completion.
  Missing, failed, errored or cancelled verification cannot become a successful completion because
  a critic approves the result. Late cancellation still vetoes success.
- Local engine startup is tied to a live child generation. Readiness probes cannot resurrect a
  process that exited, was replaced or failed to spawn. Renderer readiness also requires an
  available, ready engine with an identified model; stale metrics are not presented as current.
- Approved npm dependency setup stages permitted inputs, disables lifecycle scripts and publishes
  persistent dependencies transactionally into the selected project. Final manifest checks,
  guarded local CLI links and owned recovery paths protect concurrent work. See
  [dependency setup](../dependency-setup.md) for the supported commands and precise limits.
- Retrieval recognizes Zig, ZON, Metal and CUDA sources. Bounded breadth-first discovery spreads
  coverage across the repository; deterministic file selection, visited/budget counts and visible
  partial coverage prevent an incomplete index from looking complete. Existing privacy, symlink,
  size and path filters remain in force.
- Main-process composition, agent-loop phases, tool dispatch, sandbox, retrieval and composer
  responsibilities have focused modules. Every public tool still enters the common policy guard
  before domain dispatch. Unknown names, including inherited object properties, fail closed.

## Verification layers

The archive's acceptance matrix distinguishes these layers for each material change:

1. **Implementation:** production components and real state/service paths changed in this branch.
2. **Host verification:** regression tests, typecheck, production build and mobile checks in an
   isolated Linux development environment. Process/socket fixtures run with explicit container
   permissions; this does not qualify a Linux product command sandbox.
3. **Browser verification:** actual application renderer interactions and screenshots with the
   repository's browser-demo backend and explicitly labeled synthetic failures/receipts. These
   demonstrate UI behavior, not a real model inference, connected account or native permission.
4. **Native/device verification:** remains required on supported hardware before release.

Screenshots are application captures, not mockups. Baseline and candidate captures use matching
sizes where available. Test fixture content, its sample timings and the demo update banner are
synthetic; they are not model benchmarks, live updates or production task receipts.

## Reproduce host checks

Use Node 22 and npm 10.9.8, as pinned by the existing package policy. From `app/`:

```sh
npm ci --ignore-scripts
npm run deps:trusted-scripts
npm test
npm run typecheck
npm run build
npm --prefix mobile ci --ignore-scripts
npm run verify:mobile
git diff --check
```

The dependency preparation runs only the repository's existing trusted Electron/node-pty setup.
Native build prerequisites vary by platform. Do not relax the sandbox to turn an unavailable
production platform into a claimed supported one.

## Reproduce renderer evidence

Start `npm run dev:web -- --port 5173`. The QA scripts use a fresh profile and synthetic demo data;
they never reuse a signed-in browser. They require Playwright supplied separately to the QA
environment, not a new app dependency. The receipt records the exact browser and harness versions.

```sh
TENSORFOLD_QA_URL=http://127.0.0.1:5173 \
TENSORFOLD_QA_OUTPUT=/absolute/path/qa-workspace \
TENSORFOLD_QA_PLAYWRIGHT=/absolute/path/to/playwright \
node scripts/qa/workspace-journey.mjs
```

Other checked-in scripts exercise model setup, Assistant refresh/recovery and transcript/review.
`capture-app.mjs` takes `--url`, `--output` and `--label` to produce matching-size captures and a
fixture manifest. `TENSORFOLD_QA_CHROMIUM` optionally selects the verified browser executable.
`TENSORFOLD_QA_NO_SANDBOX=1` exists only for the isolated root container used by this receipt;
normal local/browser verification should retain Chromium's sandbox.

## Release and integration boundaries

The package version, dependency/release pins, engine bundle pin, signing/notarization settings,
update protections and release feature policy are unchanged. Native inference remains Zig.
Native kernels, Living Weights and Cortheon implementation belong to their existing workstreams.
This work did not fetch, pull, rebase or inspect later upstream state after the requested freeze,
and it did not publish, push, merge or install anything on the user's computer.

The following checks remain release gates:

- Run a supported Apple Silicon/macOS packaged candidate with the pinned native engine. Exercise
  model download interruption/resumption, actual load/unload, process exit during both readiness
  probes, first real response, cancellation, reconnect and memory/context reporting.
- Execute approved dependency setup under the actual macOS sandbox, then a separate offline
  native rebuild where appropriate. Confirm the networked stage cannot read the project or user
  credentials and dependencies are usable from the real project afterwards.
- Verify desktop draft flush/restart, multiple windows, OS focus restoration, native file pickers,
  real clipboard/attachments and native vibrancy at the minimum window size.
- Exercise connector inbox updates and exact-payload approval with test accounts. Verify deny,
  cancellation, reconnect, failed dispatch and scope changes without cross-account draft leakage.
- Complete packaged signing, notarization, update/rollback and device smoke checks through the
  existing release pipeline. No signed installer is implied by this branch-source ZIP.
- When upstream integration is authorized, compare the frozen base with the then-current branch,
  review overlapping app changes and rerun the same checks. A later integration is not covered by
  this candidate's revision identity.

Host and renderer evidence supports a reviewable release candidate. It is not a claim of
production qualification or a performance improvement benchmark.
