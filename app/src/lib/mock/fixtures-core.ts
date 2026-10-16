import type { AgentEvent, ArchiveKind, ArchiveSnapshot, DiffLine, Settings, StartRunPayload, Thread, TimelineItem, Project, Room, RoomEventItem, RoomMemoryEntry, RoomStreamEvent, Ticket, Automation, Job, GitCommitResult, GitDiffFile, GitHygieneFinding, GitHygieneReport, GitHygieneSanitizeResult, GitStatus, TokenUsageSnapshot, TokenUsageWindow, TokenUsageWindowKey, ContextPressureSnapshot, ModelRouterSnapshot, AssistantControlSnapshot, ConnectorEvent, ConnectorInteractionInput, AppUpdateStatus, CreateMissionInput, CreatePrLoopInput, CreatePrLoopResult, EmailDraftInput, EmailListOptions, EmailMessage, EmailReadResult, EmailResolveInput, EmailSourceConfig, EmailSyncResult, EmailThreadResult, Mission, MissionCheckpoint, MissionInspectionSnapshot, MissionJournalEvent, MissionPendingApproval, MissionRuntimeSnapshot, MissionStatus, Skill, Command, McpServerConfig, McpServerSnapshot, McpSnapshot, McpToolCallResult, McpToolDefinition, ControllerSnapshot, ControllerStatus, ProjectMemoryCandidate, ProjectMemoryEntry, WebhookDeliveryRecord, WebhookEndpoint, WebhookSnapshot, RemoteControlPairingInfo, RemoteControlStatus, RemoteControlWorkSnapshot, SecurityControlSnapshot, SecurityPurgeResult } from '../types'
import type { BrowserConnectorSettings, BrowserItem, BrowserListOptions, BrowserReadResult, BrowserSnapshotInput, BrowserSourceConfig, BrowserSummaryResult, BrowserSyncResult, CalendarEvent, CalendarListOptions, CalendarMutationInput, CalendarReadResult, CalendarSourceConfig, CalendarSummaryResult, CalendarSyncResult, DocumentItem, DocumentListOptions, DocumentReadResult, DocumentSourceConfig, DocumentSummaryResult, DocumentSyncResult, SlackMessage, SlackListOptions, SlackReadResult, SlackSourceConfig, SlackSummaryResult, SlackSyncResult } from '../types'
import type { Incident, RunExportBundle } from '../types'
import { PRESETS } from '../../../electron/agent/theme-presets'
import { CODEX_CLI_MODELS, CLAUDE_CLI_MODELS } from '../cli-models'
import { ago, mockState, uid } from './state'

mockState.settings = {
  providers: [
    { id: 'openai', name: 'OpenAI', baseUrl: 'https://api.openai.com/v1', apiKey: '', models: ['gpt-4o', 'gpt-4o-mini', 'o4-mini'], kind: 'cloud', source: 'openai' },
    { id: 'codex-cli', name: 'Codex CLI OAuth', transport: 'codex-cli', auth: 'codex-oauth', baseUrl: '', apiKey: '', models: CODEX_CLI_MODELS, kind: 'cloud', cliPath: 'codex', cliSandbox: 'workspace-write', cliApprovalPolicy: 'on-request', defaultAccountId: 'codex-personal', accounts: [
      { id: 'codex-personal', name: 'Personal Codex', kind: 'subscription', enabled: true, cliProfile: 'personal', cliHome: '~/.codex-personal', models: [], autoFailover: true, createdAt: ago(900), updatedAt: ago(120) },
      { id: 'codex-work', name: 'Work Codex', kind: 'subscription', enabled: true, cliProfile: 'work', cliHome: '~/.codex-work', models: [], autoFailover: true, createdAt: ago(860), updatedAt: ago(90) },
    ] },
    { id: 'claude-cli', name: 'Claude CLI OAuth', transport: 'claude-cli', auth: 'claude-oauth', baseUrl: '', apiKey: '', models: CLAUDE_CLI_MODELS, kind: 'cloud', cliPath: 'claude', cliPermissionMode: 'default', defaultAccountId: 'claude-personal', accounts: [
      { id: 'claude-personal', name: 'Personal Claude', kind: 'subscription', enabled: true, cliProfile: '~/.claude-personal/mockState.settings.json', cliHome: '~/.claude-personal', models: [], createdAt: ago(780), updatedAt: ago(80) },
      { id: 'claude-work', name: 'Work Claude', kind: 'subscription', enabled: true, cliProfile: '~/.claude-work/mockState.settings.json', cliHome: '~/.claude-work', models: [], createdAt: ago(760), updatedAt: ago(70) },
    ] },
    { id: 'omlx', name: 'oMLX (local)', transport: 'chat-completions', auth: 'api-key', baseUrl: 'http://127.0.0.1:9000/v1', apiKey: '', models: ['Qwen3-Coder-Next-MLX-8bit', 'Qwen2.5-Coder-14B-Instruct-MLX-8bit'], kind: 'local', source: 'omlx' },
    { id: 'ollama', name: 'Ollama (local)', baseUrl: 'http://127.0.0.1:11434/v1', apiKey: 'ollama', models: ['qwen2.5-coder:14b', 'llama3.1:8b'], kind: 'local', source: 'ollama' },
    { id: 'lmstudio', name: 'LM Studio (local)', baseUrl: 'http://127.0.0.1:1234/v1', apiKey: 'lm-studio', models: ['local-model'], kind: 'local', source: 'lmstudio' },
    { id: 'openrouter', name: 'OpenRouter', baseUrl: 'https://openrouter.ai/api/v1', apiKey: '', models: ['openrouter/free', 'openrouter/auto', 'openrouter/fusion', 'z-ai/glm-5.2', 'nvidia/nemotron-3-ultra-550b-a55b:free', 'qwen/qwen3-coder:free', 'openai/gpt-oss-120b:free'], kind: 'cloud', source: 'openrouter' },
    { id: 'zai', name: 'z.ai GLM', baseUrl: 'https://api.z.ai/api/paas/v4', apiKey: '', models: ['glm-5.2', 'glm-5.1', 'glm-4.6', 'glm-4-flash'], kind: 'cloud' },
    { id: 'tensorfold', name: 'TensorFold (built in)', transport: 'chat-completions', auth: 'api-key', baseUrl: 'http://127.0.0.1/v1', apiKey: '', models: ['mlx-community/Qwen3.5-2B-MLX-4bit'], kind: 'local', source: 'tensorfold' },
  ],
  cliToolOverrides: {},
  activeProviderId: 'codex-cli',
  activeProviderAccountId: 'codex-personal',
  activeModel: 'gpt-5.5',
  agentPresets: [],
  theme: 'system',
  approvalMode: 'auto',
  approvalRules: [
    {
      id: 'rule-demo-tests',
      scope: 'tool',
      key: 'run_command:test:demo',
      decision: 'allow',
      duration: 'always',
      label: 'test in demo workspace',
      explanation: 'Allows this test command signature in the demo workspace, not arbitrary shell commands.',
      legacyTool: 'run_command',
      createdAt: ago(240),
      updatedAt: ago(240),
      lastUsedAt: ago(30),
    },
    {
      id: 'rule-demo-draft',
      scope: 'tool',
      key: 'email_compose:draft:demo',
      decision: 'allow',
      duration: 'always',
      label: 'draft email to team@example.com',
      explanation: 'Allows draft email composition only for team@example.com with this subject signature.',
      legacyTool: 'email_compose',
      createdAt: ago(180),
      updatedAt: ago(180),
      lastUsedAt: null,
    },
  ],
  defaultCwd: '/Users/example/Projects/web-dashboard',
  temperature: 0.4,
  reasoningEffort: 'xhigh',
  autoCompact: true,
  contextLimit: 128_000,
  localContextLimit: 24_000,
  autoCheckUpdates: true,
  updateFeedUrl: '',
  power: {
    keepComputerAlive: false,
    lastStartedAt: null,
    lastError: null,
  },
  remoteControl: {
    enabled: false,
    serverName: 'TensorFold Orchestrator',
    bindHost: '127.0.0.1',
    port: 8765,
    authToken: 'demo_remote_token',
    allowLan: false,
    lastStartedAt: null,
    lastClientAt: null,
    lastError: null,
  },
  remoteClient: {
    enabled: false,
    serverUrl: '',
    authToken: '',
    lastConnectedAt: null,
    lastError: null,
  },
  remoteOrchestrators: [
    {
      id: 'orch_demo_studio',
      name: 'Mac Studio Orchestrator',
      serverUrl: 'http://192.168.1.50:8765',
      authToken: 'demo_remote_token',
      enabled: true,
      lastConnectedAt: ago(14),
      lastError: null,
      createdAt: ago(1800),
      updatedAt: ago(14),
    },
  ],
  email: {
    sources: [
      { id: 'email_demo', label: 'Demo Maildir', path: '/Users/example/Mail/demo', kind: 'maildir', enabled: true, createdAt: ago(360), updatedAt: ago(22), lastSyncAt: ago(22), lastError: null },
    ],
    maxMessages: 300,
    lastSyncAt: ago(22),
  },
  slack: {
    sources: [
      { id: 'slack_demo', label: 'Demo Slack export', path: '/Users/example/Slack/demo-export', kind: 'export_dir', enabled: true, createdAt: ago(360), updatedAt: ago(16), lastSyncAt: ago(16), lastError: null },
    ],
    maxMessages: 300,
    lastSyncAt: ago(16),
  },
  calendar: {
    sources: [
      { id: 'cal_demo', label: 'Demo Calendar', path: '/Users/example/Calendars/demo.ics', kind: 'ics_file', enabled: true, createdAt: ago(360), updatedAt: ago(12), lastSyncAt: ago(12), lastError: null },
    ],
    maxEvents: 500,
    lastSyncAt: ago(12),
  },
  documents: {
    sources: [
      { id: 'docs_demo', label: 'Demo Documents', path: '/Users/example/Documents/TensorFold', kind: 'dir', enabled: true, createdAt: ago(360), updatedAt: ago(18), lastSyncAt: ago(18), lastError: null },
    ],
    maxDocuments: 500,
    maxDocumentBytes: 1_000_000,
    lastSyncAt: ago(18),
  },
  browser: {
    sources: [
      { id: 'browser_tab_demo', label: 'Demo current tab', kind: 'current_tab', enabled: true, createdAt: ago(40), updatedAt: ago(11), lastSyncAt: ago(11), lastError: null, url: 'https://example.com/launch-review', title: 'Launch review tab', excerpt: 'The foreground browser tab mentions auth edge cases, QA evidence, and release-note ownership.', capturedAt: ago(11) },
      { id: 'browser_history_demo', label: 'Demo history export', kind: 'history_export', enabled: true, createdAt: ago(80), updatedAt: ago(20), lastSyncAt: ago(20), lastError: null, path: '/Users/example/Browser/history-export.json' },
    ],
    maxItems: 300,
    lastSyncAt: ago(11),
  },
  imessage: {
    enabled: false,
    allowedChatIds: '',
    maxMessages: 200,
    lastSyncAt: null,
    lastError: null,
  },
  whatsapp: {
    sources: [],
    lastSyncAt: null,
  },
  discord: {
    sources: [],
    lastSyncAt: null,
  },
  mattermost: {
    sources: [],
    lastSyncAt: null,
  },
  matrix: {
    sources: [],
    lastSyncAt: null,
  },
  googlechat: {
    sources: [],
    lastSyncAt: null,
  },
  botControl: {
    telegram: {
      enabled: false,
      botToken: '',
      allowedChatIds: '',
      allowedUserIds: '',
      polling: false,
      lastUpdateId: null,
      lastPollAt: null,
      lastError: null,
    },
    whatsapp: {
      enabled: false,
      accessToken: '',
      phoneNumberId: '',
      verifyToken: '',
      appSecret: '',
      allowedChatIds: '',
      lastEventAt: null,
      lastError: null,
    },
  },
  capabilities: {
    disabledCapabilityIds: [],
    health: {},
    localScripts: [],
    packagedPlugins: [],
  },
  controller: {
    enabled: false,
    includeAccessibilityText: false,
    maxTextChars: 4_000,
    lastCapturedAt: null,
    lastError: null,
  },
  voice: {
    enabled: true,
    ttsEnabled: true,
    sttEnabled: true,
    ttsProvider: 'browser',
    sttProvider: 'browser',
    voiceName: '',
    rate: 1.6,
    pitch: 1,
    volume: 1,
    autoSummarizeBeforeRead: true,
    summaryMaxChars: 900,
    autoReadRoomUpdates: false,
    apiBaseUrl: 'https://api.openai.com/v1',
    apiKey: '',
    ttsModel: 'gpt-4o-mini-tts',
    sttModel: 'gpt-4o-mini-transcribe',
    elevenLabsApiKey: '',
    elevenLabsVoiceId: '',
    elevenLabsModelId: 'eleven_multilingual_v2',
    elevenLabsStability: 0.4,
    elevenLabsStyle: 0.3,
    localTtsBaseUrl: 'http://127.0.0.1:8080',
    localTtsVoice: '',
    localTtsEmotion: '',
  },
  preset: 'tensorfold',
  light: PRESETS.tensorfold.light,
  dark: PRESETS.tensorfold.dark,
  pointerCursors: true,
  doomTodos: true,
  notifyRuns: true,
  sshHosts: [],
  trustedSshHostKeys: [],
}

mockState.sampleDiff = [
  { type: 'hunk', text: '@@ -14,6 +14,9 @@ export function App() {' },
  { type: 'ctx', text: '  const [open, setOpen] = useState(false)', oldNo: 14, newNo: 14 },
  { type: 'ctx', text: '  const [theme, setTheme] = useTheme()', oldNo: 15, newNo: 15 },
  { type: 'del', text: '  return <div className="app">', oldNo: 16 },
  { type: 'add', text: '  return <div className="app" data-theme={theme}>', newNo: 16 },
  { type: 'add', text: '    <ThemeToggle value={theme} onChange={setTheme} />', newNo: 17 },
  { type: 'ctx', text: '      <Sidebar />', oldNo: 17, newNo: 18 },
  { type: 'ctx', text: '      <Main />', oldNo: 18, newNo: 19 },
]

mockState.sampleVisualization = {
  kind: 'flow',
  title: 'One Model, Three Machines, Two Jobs',
  subtitle: 'A compact flow artifact rendered from structured model output.',
  nodes: [
    { id: 'gb10', title: 'GB10', subtitle: 'trains the base', tone: 'accent', column: 0, row: 0 },
    { id: 'base', title: 'base.pt', subtitle: 'the model', tone: 'accent', column: 0, row: 1 },
    { id: 'studio', title: 'Studio', subtitle: 'teacher data', tone: 'muted', column: 1, row: 0 },
    { id: 'broad', title: 'broad.jsonl', subtitle: 'on the Studio', tone: 'muted', column: 1, row: 1 },
    { id: 'macbook', title: 'MacBook', subtitle: 'teacher data', tone: 'muted', column: 2, row: 0 },
    { id: 'code', title: 'code.jsonl', subtitle: 'on the MacBook', tone: 'muted', column: 2, row: 1 },
    { id: 'copy', title: 'Both data files', subtitle: 'copied to GB10', tone: 'muted', column: 0, row: 2 },
    { id: 'distill', title: 'GB10 distills', subtitle: 'into the same model', tone: 'accent', column: 1, row: 2 },
    { id: 'reasoning', title: 'Reasoning model', subtitle: 'same base.pt', tone: 'accent', column: 2, row: 2 },
  ],
  edges: [
    { from: 'gb10', to: 'base' },
    { from: 'studio', to: 'broad' },
    { from: 'macbook', to: 'code' },
    { from: 'copy', to: 'distill' },
    { from: 'distill', to: 'reasoning' },
  ],
  legend: [
    { label: 'the model being built', tone: 'accent' },
    { label: 'data factories and files', tone: 'muted' },
  ],
  caption: 'The model stays singular; the other machines feed future curriculum data.',
}

export const DEMO_ENGINE_MODEL = 'mlx-community/Qwen3.5-2B-MLX-4bit'

const THEME_DIFF: TimelineItem['diff'] = [
  { type: 'hunk', text: '@@ -0,0 +1,14 @@' },
  { type: 'add', text: "import { useEffect, useState } from 'react'", newNo: 1 },
  { type: 'add', text: '', newNo: 2 },
  { type: 'add', text: "export type Theme = 'light' | 'dark'", newNo: 3 },
  { type: 'add', text: '', newNo: 4 },
  { type: 'add', text: 'export function useTheme(): [Theme, (t: Theme) => void] {', newNo: 5 },
  { type: 'add', text: "  const [theme, setTheme] = useState<Theme>(() => (localStorage.getItem('theme') as Theme) || 'light')", newNo: 6 },
  { type: 'add', text: '  useEffect(() => {', newNo: 7 },
  { type: 'add', text: "    localStorage.setItem('theme', theme)", newNo: 8 },
  { type: 'add', text: '    document.documentElement.dataset.theme = theme', newNo: 9 },
  { type: 'add', text: '  }, [theme])', newNo: 10 },
  { type: 'add', text: '  return [theme, setTheme]', newNo: 11 },
  { type: 'add', text: '}', newNo: 12 },
]

// A finished Code turn on the built-in engine: reads, edits, a sandboxed test run, bookkeeping, measured stats.
export function demoTimeline(): TimelineItem[] {
  const t0 = ago(5)
  const at = (s: number) => t0 + s * 1000
  const runId = 'demo-run'
  return [
    { id: uid(), kind: 'user', text: 'Add a dark mode toggle to the dashboard header and persist the choice.', createdAt: at(0) },
    { id: uid(), kind: 'notice', title: 'Run contract', text: `Run contract saved: code on local with ${DEMO_ENGINE_MODEL}.`, state: 'ok', createdAt: at(0) },
    { id: uid(), kind: 'reasoning', text: 'The header lives in src/App.tsx. I should add a ThemeToggle component, wire it to a useTheme hook backed by localStorage, and pass the current theme down via a data attribute so the CSS variables switch.', createdAt: at(1) },
    { id: uid(), kind: 'assistant', text: "I'll add a `ThemeToggle` to the header and persist the preference in `localStorage`. Let me look at the current layout first.", createdAt: at(2) },
    { id: uid(), kind: 'tool', tool: 'read_file', title: 'Read', path: 'src/App.tsx', state: 'ok', meta: '62 lines', output: '    1\timport { useState } from "react"\n    2\timport { Sidebar } from "./Sidebar"\n    3\t…', createdAt: at(3) },
    { id: uid(), kind: 'tool', tool: 'write_file', title: 'Create', path: 'src/theme.ts', state: 'ok', meta: '+12', diff: THEME_DIFF, changedPaths: ['src/theme.ts'], createdAt: at(5) },
    { id: uid(), kind: 'tool', tool: 'edit_file', title: 'Edit', path: 'src/App.tsx', state: 'ok', meta: '+2 −1', diff: mockState.sampleDiff, changedPaths: ['src/App.tsx'], createdAt: at(7) },
    { id: uid(), kind: 'notice', title: 'Verify-on-stop', text: 'Code changes need fresh verification before this run can conclude: src/theme.ts, src/App.tsx', state: 'awaiting', createdAt: at(8) },
    {
      id: uid(), kind: 'tool', tool: 'run_command', title: 'Run', command: 'npm test -- --run', state: 'ok',
      meta: 'exit 0 · 2140ms · sandbox=build-test · macos-sandbox-exec · artifacts=sbox-demo1',
      output: '✓ src/App.test.tsx (4)\n✓ src/theme.test.ts (3)\n\nTest Files  2 passed (2)\n     Tests  7 passed (7)',
      artifacts: { id: 'sbox-demo1', profile: 'build-test', implementation: 'macos-sandbox-exec', inputFiles: 3, outputFiles: 1, screenshots: [], diffs: [{ path: '.vontra-artifacts/coverage.json', status: 'added' }], patch: true, logBytes: 212 },
      createdAt: at(9),
    },
    {
      id: uid(), kind: 'assistant', createdAt: at(12),
      text: "Done. I added a **`ThemeToggle`** to the header, wired it to a `useTheme()` hook that reads and writes `localStorage`, and switched the root element to use `data-theme`. All 7 tests pass.\n\n- `src/App.tsx` renders the toggle and applies `data-theme`\n- `src/theme.ts` holds `useTheme()` with persistence",
      stats: { model: DEMO_ENGINE_MODEL, providerId: 'tensorfold', modelCalls: 4, toolCalls: 4, completionTokens: 1184, decodeTokensPerSecond: 212.4, decodeSeconds: 5.556, prefillTokensPerSecond: 3140.2, promptTokens: 18406, cachedTokens: 16210, firstTokenMs: 380 },
    },
    {
      id: uid(), kind: 'notice', title: 'Run receipt', text: 'Run receipt: verified. 2 files changed, 1 check passed.', state: 'ok', createdAt: at(14),
      runReceipt: { id: 'receipt-demo', runId, threadId: 'demo', verdict: 'verified', summary: '2 files changed, 1 check passed.', changedFiles: ['src/App.tsx', 'src/theme.ts'], commands: ['npm test -- --run'], tests: ['npm test -- --run'], approvals: 0, diffs: 2, artifacts: ['sbox-demo1'], errors: 0, eventReceipts: [], gaps: [], warnings: [], startedAt: at(0), finishedAt: at(14), needsReview: false },
    },
  ]
}

export function thread(title: string, cwd: string, min: number, status: Thread['status'], timeline: TimelineItem[], kind: Thread['kind'] = 'task', projectId: string | null = null): Thread {
  return { id: uid(), title, kind, projectId, cwd, providerId: 'openai', model: 'gpt-4o', createdAt: ago(min + 30), updatedAt: ago(min), status, apiMessages: [], timeline }
}

mockState.threads = [
  {
    ...thread('Add dark mode toggle', '/Users/example/Projects/web-dashboard', 4, 'idle', demoTimeline(), 'task', 'proj-demo'),
    providerId: 'tensorfold',
    model: DEMO_ENGINE_MODEL,
    goal: 'Add a persistent dark-mode toggle to the dashboard header',
    goalStatus: 'complete',
    plan: [
      { id: 'p1', text: 'Add a ThemeToggle to the dashboard header', done: true, status: 'done' },
      { id: 'p2', text: 'Back it with a useTheme() hook + localStorage', done: true, status: 'done' },
      { id: 'p3', text: 'Switch the root to data-theme and wire CSS variables', done: true, status: 'done' },
      { id: 'p4', text: 'Verify with the existing theme tests', done: true, status: 'done' },
    ],
  },
  {
    ...thread('Harden the billing webhook', '/Users/example/Projects/api-server', 95, 'running', [
      { id: uid(), kind: 'user', text: 'The Stripe webhook dropped a duplicate charge last night. Trace it, fix it, and add coverage.' },
      { id: uid(), kind: 'assistant', text: 'On it — tracing the replay path now.' },
    ]),
    goal: 'Make the billing webhook idempotent and prove it with tests',
    goalStatus: 'active',
    plan: [
      { id: 'w1', text: 'Trace the webhook replay path end to end', done: true, status: 'done' },
      { id: 'w2', text: 'Reproduce the duplicate charge locally', done: false, status: 'in_progress' },
      { id: 'w2a', text: 'Replay the fixture twice against the local handler', done: false, status: 'pending', depth: 1 },
      { id: 'w3', text: 'Guard with an idempotency key check', done: false, status: 'blocked', blocker: 'Stripe test-mode key rotation', depth: 0 },
      { id: 'w4', text: 'Port the old retry shim away', done: false, status: 'dropped' },
    ],
  },
  thread('Fix flaky login test', '/Users/example/Projects/api-server', 95, 'idle', [
    { id: uid(), kind: 'user', text: 'The login test fails intermittently in CI. Find out why and fix it.' },
    { id: uid(), kind: 'assistant', text: 'It was a race on the async session write. I awaited the persistence call and the flake is gone across 50 runs.' },
  ]),
  thread('Explain the codebase', '/Users/example/Projects/web-dashboard', 1500, 'idle', [
    { id: uid(), kind: 'user', text: 'Give me a tour of this project.' },
    { id: uid(), kind: 'tool', tool: 'list_dir', title: 'List', path: '.', state: 'ok', meta: '14 entries', output: 'src/\npackage.json\nREADME.md' },
    { id: uid(), kind: 'visualization', tool: 'visualize', title: 'One Model, Three Machines, Two Jobs', state: 'ok', meta: '9 nodes', visualization: mockState.sampleVisualization },
    { id: uid(), kind: 'assistant', text: 'The flow above is the whole system: one model, fed by two data jobs on separate machines.' },
  ], 'task', 'proj-demo'),
  thread('Async/await, simply', '', 30, 'idle', [
    { id: uid(), kind: 'user', text: 'Explain async/await in JavaScript, simply.' },
    { id: uid(), kind: 'assistant', text: '`async/await` lets you write asynchronous code that reads top-to-bottom like synchronous code. An `async` function always returns a promise, and `await` pauses inside it until a promise settles — without blocking the main thread.' },
  ], 'chat'),
]

mockState.listener = null
export const emit = (e: AgentEvent) => mockState.listener?.(e)
const simulations = new Map<string, () => void>()

export function cancelSimulation(threadId: string): void {
  const cancel = simulations.get(threadId)
  if (cancel) cancel()
  else emit({ runId: 'demo', threadId, type: 'done' })
}

export function simulate(threadId: string, userText: string) {
  simulations.get(threadId)?.()
  const reasoningId = uid()
  const assistantId = uid()
  const th = mockState.threads.find((t) => t.id === threadId)
  const isTask = th?.kind === 'task'
  const steps: Array<[() => void, number]> = []
  const step = (fn: () => void, delay = 45) => steps.push([fn, delay])
  const ev = (e: Record<string, unknown>) => {
    if (th) {
      if (e.type === 'status') th.status = e.status === 'error' ? 'error' : e.status === 'idle' || e.status === 'done' ? 'idle' : 'running'
      if (e.type === 'done') th.status = 'idle'
      th.updatedAt = Date.now()
    }
    emit({ runId: 'demo', threadId, ...e } as AgentEvent)
  }
  const startedAt = Date.now()
  if (th) th.status = 'running'
  step(() => ev({ type: 'status', status: 'thinking' }))
  step(() => ev({ type: 'item_add', item: { id: reasoningId, kind: 'reasoning', text: '', createdAt: Date.now() } }))
  const thought = 'This is TensorFold running in browser demo mode, so I am simulating a response. In the desktop app this streams from your configured model and can read, edit, and run code.'
  thought.match(/.{1,28}/g)!.forEach((c) => step(() => ev({ type: 'item_delta', id: reasoningId, field: 'text', chunk: c })))

  // Code tasks show a tool-using run with realistic pacing.
  if (isTask) {
    const tool = (item: TimelineItem, done: Partial<TimelineItem>, runMs: number) => {
      step(() => { ev({ type: 'status', status: 'tool' }); ev({ type: 'item_add', item: { ...item, state: 'running', createdAt: Date.now() } }) }, 120)
      step(() => ev({ type: 'item_update', id: item.id, patch: { state: 'ok', ...done } }), runMs)
    }
    tool({ id: uid(), kind: 'tool', tool: 'read_file', title: 'Read', path: 'src/App.tsx' }, { meta: '62 lines', output: '    1\timport { useState } from "react"' }, 900)
    tool({ id: uid(), kind: 'tool', tool: 'edit_file', title: 'Edit', path: 'src/App.tsx' }, { meta: '+2 −1', diff: mockState.sampleDiff, changedPaths: ['src/App.tsx'] }, 1600)
    tool({ id: uid(), kind: 'tool', tool: 'run_command', title: 'Run', command: 'npm test -- --run' }, { meta: 'exit 0 · 2140ms', output: 'Tests  7 passed (7)' }, 4000)
  }

  step(() => ev({ type: 'status', status: 'streaming' }))
  step(() => ev({ type: 'item_add', item: { id: assistantId, kind: 'assistant', text: '', createdAt: Date.now() } }))
  const reply = `You said: *"${userText}"*.\n\nThis is **demo mode**. Open TensorFold as the desktop app and add a provider (OpenAI, a local Ollama/LM Studio endpoint, or any OpenAI-compatible URL) to run real agentic tasks with live tool calls and diffs.`
  reply.match(/.{1,24}/g)!.forEach((c) => step(() => ev({ type: 'item_delta', id: assistantId, field: 'text', chunk: c })))
  if (isTask) step(() => ev({ type: 'item_update', id: assistantId, patch: { stats: { model: th?.model ?? DEMO_ENGINE_MODEL, providerId: th?.providerId ?? null, modelCalls: 4, toolCalls: 3, completionTokens: 640, decodeTokensPerSecond: 198, decodeSeconds: 3.212, prefillTokensPerSecond: 2870.5, promptTokens: 9812, cachedTokens: 8640, firstTokenMs: Math.min(900, Date.now() - startedAt) } } }))

  // Goal Mode demo: build a plan, tick steps off, then verify complete.
  if (th?.goal && th.goalStatus === 'active') {
    const plan = [
      { id: 'g1', text: 'Understand the request', done: false },
      { id: 'g2', text: 'Make the change', done: false },
      { id: 'g3', text: 'Verify it works', done: false },
    ]
    step(() => ev({ type: 'plan', plan: plan.map((p) => ({ ...p })) }))
    step(() => ev({ type: 'plan', plan: plan.map((p, i) => ({ ...p, done: i < 1 })) }))
    step(() => ev({ type: 'plan', plan: plan.map((p, i) => ({ ...p, done: i < 2 })) }))
    step(() => ev({ type: 'plan', plan: plan.map((p) => ({ ...p, done: true })), goalStatus: 'complete' }))
    step(() => ev({ type: 'item_add', item: { id: uid(), kind: 'notice', text: 'Goal complete. Verified.', createdAt: Date.now() } }))
  }

  step(() => ev({ type: 'done' }))

  let i = 0
  let timer: ReturnType<typeof setTimeout>
  let cancelled = false
  const tick = () => {
    if (cancelled) return
    if (i < steps.length) { const [fn, delay] = steps[i++]; fn(); timer = setTimeout(tick, delay) }
    else simulations.delete(threadId)
  }
  simulations.set(threadId, () => {
    cancelled = true
    clearTimeout(timer)
    simulations.delete(threadId)
    const receipt: import('../types').AgentRunReceiptSnapshot = {
      id: uid(), runId: 'demo', threadId, verdict: 'cancelled', summary: 'Demo worker stopped; its worktree is preserved.',
      changedFiles: [], commands: [], tests: [], approvals: 0, diffs: 0, artifacts: [], errors: 0, eventReceipts: [], gaps: [], warnings: [],
      startedAt, finishedAt: Date.now(), needsReview: true,
    }
    if (th) th.lastRunReceipt = receipt
    ev({ type: 'thread_update', patch: { activeRunContract: null, lastRunReceipt: receipt } })
    ev({ type: 'done' })
  })
  timer = setTimeout(tick, 250)
}

// ---- Projects & rooms (demo) ----
mockState.projects = [
  { id: 'proj-demo', name: 'Zora-Agent', cwd: '/Users/example/Projects/zora-agent', memory: 'Goal: ship the v2 API. Stack: Node + Postgres. Keep endpoints REST.', team: 'Architect designs the surface, Engineer implements, Reviewer checks correctness.', kanban: { modelMode: 'single' }, createdAt: ago(3000), updatedAt: ago(60) },
]
mockState.mockRooms = [
  {
    id: 'room-demo', projectId: 'proj-demo', name: 'API design huddle',
    seats: [
      { id: 's1', name: 'planner', role: 'Planner - API surface and acceptance checks', lane: 'planning', providerId: 'openai', model: 'gpt-4o' },
      { id: 's2', name: 'builder', role: 'Builder - local implementation details', lane: 'development', providerId: 'lmstudio', model: 'local-model' },
      { id: 's3', name: 'reviewer', role: 'Reviewer - evidence and risk checks', lane: 'review', providerId: 'codex-cli', providerAccountId: 'codex-work', model: 'codex-cli-default' },
    ],
    collaborationMode: 'frontier_plans_local_builds',
    deliberationEffort: 'normal',
    workerBudget: { maxActive: 2, maxStarts: 4 },
    memory: [
      { id: 'mem-1', kind: 'decision', text: 'Keep the public API REST-first unless the dashboard needs streaming updates.', source: 'manual', createdAt: ago(120), updatedAt: ago(120) },
    ],
    memorySummary: 'Previous room decided to keep REST as the default API surface and use local seats for implementation detail.',
    problem: '', state: 'ready', events: [], finalText: '', createdAt: ago(200), updatedAt: ago(40),
  },
]
mockState.roomListener = null
export const roomEmit = (e: RoomStreamEvent) => mockState.roomListener?.(e)
const mkItem = (tick: number, kind: string, seat: string, target: string, text: string): RoomEventItem => ({ id: uid(), tick, kind, seat, target, text, createdAt: Date.now() })

export function simulateRoom(roomId: string, problem: string) {
  const room = mockState.mockRooms.find((r) => r.id === roomId)
  const seats = room && room.seats.length ? room.seats : [{ name: 'architect' } as any, { name: 'engineer' } as any]
  const a = seats[0].name
  const b = (seats[1]?.name) || 'engineer'
  const script: Array<{ seat: string; kind: string; target: string; prose: string; full: string; tool?: string; path?: string }> = [
    { seat: a, kind: 'assign', target: b, prose: `Let's split it. I'll define the API surface; ${b}, you take the handlers.`, full: `Let's split it. I'll define the API surface; ${b}, you take the handlers.` },
    { seat: a, kind: 'tool', target: 'all', prose: '', full: `claim_ticket — Claimed "Define API surface" so ${b} knows it's mine.` },
    { seat: a, kind: 'tool', target: 'all', tool: 'edit_file', path: 'src/api/spec.ts', prose: '', full: `edit_file — Wrote the OpenAPI surface in src/api/spec.ts (+48 −2).` },
    { seat: b, kind: 'ask', target: a, prose: `On it. Quick call, ${a}: REST or RPC for the dashboard?`, full: `On it. Quick call, ${a}: REST or RPC for the dashboard?` },
    { seat: a, kind: 'speak', target: 'all', prose: `REST — simpler for the dashboard to consume. The spec's in src/api/spec.ts; build the handlers off that.`, full: `REST — simpler. The spec is in src/api/spec.ts; build the handlers off that.` },
    { seat: b, kind: 'tool', target: 'all', tool: 'write_file', path: 'src/api/handlers/records.ts', prose: '', full: `write_file — Created src/api/handlers/records.ts.` },
    { seat: b, kind: 'ask', target: a, prose: `Done — ${a}, can you review src/api/handlers/records.ts before I open the PR?`, full: `Done — ${a}, can you review src/api/handlers/records.ts?` },
    { seat: a, kind: 'speak', target: 'all', prose: `Reviewed — one nit on error handling, otherwise it's solid. Ship it.`, full: `Reviewed — one nit on error handling, otherwise it's solid. Ship it.` },
    { seat: b, kind: 'final', target: 'all', prose: `Plan locked & built: REST API, spec from ${a}, handlers from me. Shipping it.`, full: `Plan locked & built: REST API, spec from ${a}, handlers from me.` },
  ]
  const steps: Array<() => void> = []
  let tick = 0
  steps.push(() => roomEmit({ roomId, type: 'status', state: 'queued' }))
  steps.push(() => roomEmit({ roomId, type: 'status', state: 'running' }))
  steps.push(() => roomEmit({ roomId, type: 'event', item: mkItem(++tick, 'task', 'you', 'all', problem) }))
  for (const s of script) {
    steps.push(() => roomEmit({ roomId, type: 'turn', seat: s.seat }))
    ;(s.prose.match(/.{1,16}/g) || []).forEach((c) => steps.push(() => roomEmit({ roomId, type: 'delta', seat: s.seat, chunk: c })))
    steps.push(() => roomEmit({ roomId, type: 'event', item: { ...mkItem(++tick, s.kind, s.seat, s.target, s.full), tool: s.tool, path: s.path } }))
    if (s.kind === 'final') {
      steps.push(() => {
        const current = mockState.mockRooms.find((r) => r.id === roomId)
        if (current) {
          const now = Date.now()
          const entry: RoomMemoryEntry = { id: uid(), kind: 'summary', text: `Request: ${problem}\nOutcome: ${s.full}`, source: 'room', createdAt: now, updatedAt: now }
          current.memory = [entry, ...(current.memory ?? [])].slice(0, 24)
          current.memorySummary = `- ${entry.text}\n${current.memorySummary ?? ''}`.slice(0, 5000)
          roomEmit({ roomId, type: 'memory', memory: current.memory, memorySummary: current.memorySummary })
        }
      })
      steps.push(() => roomEmit({ roomId, type: 'final', text: s.full }))
      steps.push(() => roomEmit({ roomId, type: 'status', state: 'complete' }))
    }
  }
  steps.push(() => roomEmit({ roomId, type: 'done' }))
  let i = 0
  const tickFn = () => { if (i < steps.length) { steps[i++](); setTimeout(tickFn, 38) } }
  setTimeout(tickFn, 220)
}
