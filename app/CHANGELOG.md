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
