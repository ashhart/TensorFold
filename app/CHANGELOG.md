# TensorFold Desktop changelog

Desktop installers ship under `app-v*` tags. Engine updates follow a separate release schedule.
See the [engine changelog](../CHANGELOG.md) for engine releases.

## Unreleased

These are highlights from changes merged after Desktop 1.0.0. They are available in builds from
current source and are awaiting a packaged Desktop release.

- Search in **Current conversation** ignores the project filter from **All saved conversations**.
  Switching back restores the selected project. [#3900](https://github.com/ashhart/TensorFold/pull/3900)
- **Planning only** opens the model picker on **Local**. The regular model button opens **All**.
  [#3357](https://github.com/ashhart/TensorFold/pull/3357)
- Rooms show completed tickets out of the total. An agent can release its own idle ticket claim
  so another seat can take it. [#3698](https://github.com/ashhart/TensorFold/pull/3698) · [#4081](https://github.com/ashhart/TensorFold/pull/4081)
- Room health shows the Room's worktree path and Git branch.
  [#3463](https://github.com/ashhart/TensorFold/pull/3463)
- Room settings save when you select **Supervisor** or apply the **Supervised build** preset.
  [#3472](https://github.com/ashhart/TensorFold/pull/3472)
- Use **Details** to view and edit a Kanban ticket's title, description, type, priority and effort.
  Archived tickets can be restored. [#3464](https://github.com/ashhart/TensorFold/pull/3464)
- Moving a parked ticket to **Ready** clears its assignee so another Room seat can take it.
  [#3469](https://github.com/ashhart/TensorFold/pull/3469)
- New Solo and Room ticket worker chats appear in the sidebar as soon as they're created.
  [#3461](https://github.com/ashhart/TensorFold/pull/3461)
- Each saved bot can have its own reasoning effort. Bots without a saved effort use the app default.
  [#3500](https://github.com/ashhart/TensorFold/pull/3500)
- **Replay as new task** accepts one launch at a time, so a second click during a launch
  creates no duplicate task. [#3848](https://github.com/ashhart/TensorFold/pull/3848)
- **Export Run** waits for a preview that matches the selected run, mode and redaction settings
  before enabling copy or save. [#3834](https://github.com/ashhart/TensorFold/pull/3834)
- Model downloads honor proxy settings and offer a retry when a download stalls or fails.
  [#4141](https://github.com/ashhart/TensorFold/pull/4141)
- Benchmarks accept community-v2 results and preserve optional Mac product-family labels in
  saved history and exports. [#4031](https://github.com/ashhart/TensorFold/pull/4031) · [#2693](https://github.com/ashhart/TensorFold/pull/2693)

## 1.0.0 · 2026-11-02

[TensorFold for Mac 1.0.0 release notes and installer](https://github.com/ashhart/TensorFold/releases/tag/app-v1.0.0)

- First stable Desktop release for Apple silicon Macs running macOS 13.3 or later.
- Ask TensorFold adds a compact input bar with window selection and capture preview. The main
  window can close into the background and reopen from the menu bar.
- Connection health and the unified inbox show account status and provider errors. The interface
  includes Portuguese.
- The installer bundles engine 1.2.0. The app can download and activate engine 1.2.1 separately
  from **Models → Details → Engine updates**.
