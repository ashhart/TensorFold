# TensorFold — notes for coding agents

Agentic AI coding harness (Electron + React + Vite + Tailwind v4). See `README.md` for the
full picture. This file captures the non-obvious bits.

## Run / build
- `npm run dev` — Vite dev server + auto-launches Electron (via `vite-plugin-electron`).
- `npm run build` — renderer → `dist/`, main+preload → `dist-electron/`.
- `npm run typecheck` — typecheck (Vite/esbuild does not typecheck on its own).
- `npm test` — guarded agent/runtime suite; fails if any `*.test.ts(x)` / `*.test.mjs` file is
  not wired into `test:agent`, or source/docs contain non-placeholder email addresses.
- `npm run verify` — non-build preflight: `npm test`, typecheck, and `git diff --check`.

## Things to know
- **Electron main/preload are bundled to CJS** (see `vite.config.ts` rollup output) and
  `package.json` has no `"type": "module"`. Keep it that way or the preload path / `__dirname`
  resolution in `electron/main.ts` will break.
- **Renderer ↔ main type sharing** is type‑only: `src/lib/types.ts` re‑exports from
  `electron/agent/types.ts`. Never import runtime code across that boundary.
- **Browser demo mode**: `src/lib/ipc.ts` falls back to `src/lib/mock.ts` when
  `window.tensorfoldStudio` is absent (plain browser at the Vite URL). Real fs/shell only runs in Electron.
- **The agent loop** (`electron/agent/loop.ts`) streams `/chat/completions`, accumulates tool
  calls, gates them through the approval policy, executes via `tools.ts`, and emits `AgentEvent`s
  that both `main.ts` (persistence) and the renderer store mirror into the timeline.
- **Design tokens** live in `src/styles/index.css` (`@theme inline` + light/`.dark` vars).
  Text uses the `--ink` opacity scale; the runtime theme (`src/lib/theme.ts`) derives surfaces
  and borders from the active TensorFold preset. Use the semantic utilities (`bg-card`,
  `text-ink-2`, `border-line`, `text-accent`…), not raw colors.
