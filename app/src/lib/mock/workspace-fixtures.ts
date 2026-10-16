import type { AgentEvent, DiffLine, GitDiffFile, GitStatus, TimelineItem } from '../types'
import { mockState, uid } from './state'
import { thread } from './fixtures-core'

// Browser demo of the workspace shell: per-folder git state with real-looking diffs, and threads
// in every sidebar state (running, needs you, failed, done with diff stats).

const DASHBOARD = '/Users/example/Projects/web-dashboard'
const API_SERVER = '/Users/example/Projects/api-server'

function file(path: string, status: GitDiffFile['status'], lines: string[]): GitDiffFile {
  const body = lines.filter((l) => !l.startsWith('+++') && !l.startsWith('---'))
  return {
    path,
    status,
    additions: body.filter((l) => l.startsWith('+')).length,
    deletions: body.filter((l) => l.startsWith('-')).length,
    diff: [`diff --git a/${path} b/${path}`, ...lines].join('\n'),
  }
}

const DASHBOARD_FILES: GitDiffFile[] = [
  file('src/App.tsx', 'M', [
    '--- a/src/App.tsx', '+++ b/src/App.tsx',
    '@@ -1,6 +1,8 @@',
    " import { useState } from 'react'",
    " import { Sidebar } from './components/Sidebar'",
    " import { Main } from './pages/Dashboard'",
    "+import { ThemeToggle } from './components/ThemeToggle'",
    "+import { useTheme } from './hooks/useTheme'",
    ' ',
    ' export function App() {',
    '   const [open, setOpen] = useState(false)',
    '@@ -12,9 +14,11 @@ export function App() {',
    '   }, [open])',
    ' ',
    '-  return <div className="app">',
    '+  const [theme, setTheme] = useTheme()',
    '+  return <div className="app" data-theme={theme}>',
    '     <header className="app-header">',
    '       <h1>Dashboard</h1>',
    '+      <ThemeToggle value={theme} onChange={setTheme} />',
    '     </header>',
    '     <Sidebar open={open} onToggle={() => setOpen(!open)} />',
    '     <Main />',
  ]),
  file('src/hooks/useTheme.ts', 'A', [
    'new file mode 100644', '--- /dev/null', '+++ b/src/hooks/useTheme.ts',
    '@@ -0,0 +1,22 @@',
    "+import { useEffect, useState } from 'react'",
    '+',
    "+export type Theme = 'light' | 'dark'",
    '+',
    "+const STORAGE_KEY = 'dashboard.theme'",
    '+',
    '+function initialTheme(): Theme {',
    '+  const saved = localStorage.getItem(STORAGE_KEY)',
    "+  if (saved === 'light' || saved === 'dark') return saved",
    "+  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'",
    '+}',
    '+',
    '+export function useTheme(): [Theme, (next: Theme) => void] {',
    '+  const [theme, setTheme] = useState<Theme>(initialTheme)',
    '+  useEffect(() => {',
    '+    document.documentElement.dataset.theme = theme',
    '+    localStorage.setItem(STORAGE_KEY, theme)',
    '+  }, [theme])',
    '+  return [theme, setTheme]',
    '+}',
  ]),
  file('src/styles/tokens.css', 'M', [
    '--- a/src/styles/tokens.css', '+++ b/src/styles/tokens.css',
    '@@ -1,9 +1,19 @@',
    ' :root {',
    '   --font-sans: system-ui, sans-serif;',
    '-  --panel-radius: 18px;',
    '-  --panel-border: rgba(0, 0, 0, .08);',
    '+  --panel-radius: 12px;',
    '+  --panel-border: rgba(0, 0, 0, .1);',
    '   --bg: #ffffff;',
    '   --fg: #18181b;',
    '   --muted: #71717a;',
    ' }',
    '+',
    "+[data-theme='dark'] {",
    '+  --panel-border: rgba(255, 255, 255, .12);',
    '+  --bg: #0b0b0f;',
    '+  --fg: #f4f4f5;',
    '+  --muted: #a1a1aa;',
    '+}',
  ]),
]

const API_FILES: GitDiffFile[] = [
  file('src/webhooks/stripe.ts', 'M', [
    '--- a/src/webhooks/stripe.ts', '+++ b/src/webhooks/stripe.ts',
    '@@ -31,10 +31,16 @@ export async function handleStripeEvent(event: Stripe.Event) {',
    '   if (event.type !== \'charge.succeeded\') return { ignored: true }',
    '   const charge = event.data.object as Stripe.Charge',
    '-  await db.charges.insert({ id: charge.id, amount: charge.amount })',
    '-  return { ok: true }',
    '+  const seen = await db.webhookEvents.find(event.id)',
    '+  if (seen) return { ok: true, replay: true }',
    '+  await db.transaction(async (tx) => {',
    '+    await tx.webhookEvents.insert({ id: event.id, receivedAt: Date.now() })',
    '+    await tx.charges.insert({ id: charge.id, amount: charge.amount })',
    '+  })',
    '+  return { ok: true, replay: false }',
    ' }',
  ]),
  file('test/webhooks/stripe.test.ts', 'A', [
    'new file mode 100644', '--- /dev/null', '+++ b/test/webhooks/stripe.test.ts',
    '@@ -0,0 +1,9 @@',
    "+import { handleStripeEvent } from '../../src/webhooks/stripe'",
    "+import { chargeEvent } from '../fixtures/stripe'",
    '+',
    "+test('a replayed event charges once', async () => {",
    '+  const first = await handleStripeEvent(chargeEvent)',
    '+  const second = await handleStripeEvent(chargeEvent)',
    '+  expect(first.replay).toBe(false)',
    '+  expect(second.replay).toBe(true)',
    '+})',
  ]),
]

function filesFor(cwd: string): GitDiffFile[] {
  if (cwd.startsWith(API_SERVER)) return API_FILES
  if (cwd.startsWith(DASHBOARD) || cwd.includes('zora-agent')) return DASHBOARD_FILES
  return []
}

export function mockGitStatus(cwd: string): GitStatus {
  const files = filesFor(cwd).map(({ diff: _diff, ...rest }) => rest)
  const branch = cwd.startsWith(API_SERVER) ? 'fix/webhook-replay' : cwd.includes('zora-agent') ? 'main' : 'feat/dark-mode'
  return { branch, files, additions: files.reduce((n, f) => n + f.additions, 0), deletions: files.reduce((n, f) => n + f.deletions, 0) }
}

export function mockGitDiff(cwd: string, paths?: string[]): GitDiffFile[] {
  const all = filesFor(cwd)
  return paths?.length ? all.filter((f) => paths.includes(f.path)) : all
}

// The thread's own edit, as the loop would record it (DiffLine[] from the unified text).
function editItem(path: string, files: GitDiffFile[], tool: 'edit_file' | 'write_file' = 'edit_file'): TimelineItem {
  const diff: DiffLine[] = []
  const f = files.find((x) => x.path === path)!
  let oldNo = 0
  let newNo = 0
  for (const raw of f.diff.split('\n')) {
    const hunk = /^@@ -(\d+)(?:,\d+)? \+(\d+)/.exec(raw)
    if (hunk) { oldNo = +hunk[1]; newNo = +hunk[2]; diff.push({ type: 'hunk', text: raw }); continue }
    if (!diff.length || raw.startsWith('diff ') || raw.startsWith('new file') || raw.startsWith('---') || raw.startsWith('+++')) continue
    if (raw.startsWith('+')) diff.push({ type: 'add', text: raw.slice(1), newNo: newNo++ })
    else if (raw.startsWith('-')) diff.push({ type: 'del', text: raw.slice(1), oldNo: oldNo++ })
    else diff.push({ type: 'ctx', text: raw.slice(1), oldNo: oldNo++, newNo: newNo++ })
  }
  return { id: uid(), kind: 'tool', tool, title: tool === 'write_file' ? 'Create' : 'Edit', path, state: 'ok', meta: `+${f.additions} −${f.deletions}`, diff }
}

function byTitle(title: string) {
  return mockState.threads.find((t) => t.title === title)
}

const darkMode = byTitle('Add dark mode toggle')
if (darkMode) {
  darkMode.timeline = [
    ...darkMode.timeline.filter((i) => !(i.kind === 'tool' && i.tool === 'edit_file')),
    { id: uid(), kind: 'user', text: 'Persist the choice and give dark mode its own tokens.' },
    editItem('src/App.tsx', DASHBOARD_FILES),
    editItem('src/hooks/useTheme.ts', DASHBOARD_FILES, 'write_file'),
    editItem('src/styles/tokens.css', DASHBOARD_FILES),
    { id: uid(), kind: 'assistant', text: 'The theme now persists in `localStorage` and dark mode has its own tokens in `tokens.css`.' },
  ]
}

const webhook = byTitle('Harden the billing webhook')
if (webhook) webhook.timeline = [...webhook.timeline, editItem('src/webhooks/stripe.ts', API_FILES)]

mockState.threads.push(
  thread('Rotate the session signing keys', API_SERVER, 12, 'running', [
    { id: 'demo-rotate-user', kind: 'user', text: 'Rotate the session signing keys and keep old sessions valid for a day.' },
    { id: 'demo-rotate-plan', kind: 'assistant', text: 'I will add the new key next to the old one, then run the migration that re-signs active sessions.' },
    { id: 'demo-rotate-tool', kind: 'tool', tool: 'run_command', title: 'Run', command: 'npm run migrate:sessions -- --resign', state: 'awaiting', output: 'Runs a database migration on the local dev database.' },
  ]),
  thread('Upgrade to Vite 7', DASHBOARD, 200, 'error', [
    { id: uid(), kind: 'user', text: 'Upgrade the dashboard to Vite 7.' },
    { id: uid(), kind: 'tool', tool: 'run_command', title: 'Run', command: 'npm run build', state: 'error', meta: 'exit 1', output: 'error: vite-plugin-legacy does not support Vite 7' },
    { id: uid(), kind: 'error', text: 'The build failed: vite-plugin-legacy does not support Vite 7 yet.' },
  ]),
)

// Live run states the demo replays once the store subscribes, so rows show every status.
export function demoRunStateEvents(): AgentEvent[] {
  const events: AgentEvent[] = []
  const running = byTitle('Harden the billing webhook')
  if (running) events.push({ runId: 'demo-running', threadId: running.id, type: 'status', status: 'tool' })
  const waiting = byTitle('Rotate the session signing keys')
  if (waiting) {
    events.push({ runId: 'demo-waiting', threadId: waiting.id, type: 'status', status: 'awaiting' })
    events.push({ runId: 'demo-waiting', threadId: waiting.id, type: 'approval_request', callId: 'demo-rotate-call', itemId: 'demo-rotate-tool' })
  }
  return events
}

// The demo's one pending approval belongs to the waiting thread.
export function demoApprovalThread(callId: string): string | null {
  return callId === 'demo-rotate-call' ? byTitle('Rotate the session signing keys')?.id ?? null : null
}
