import { validateDocumentSourceIds } from '../../../shared/document-collections'
import { parseThreadMemoryPolicy, threadMemoryPolicyRevision, assertThreadMemoryTransport } from '../../../shared/thread-memory-policy'
import type { AgentEvent, StartRunPayload, Thread, GitCommitResult, GitDiffFile, GitHygieneReport, GitHygieneSanitizeResult, GitStatus } from '../types'
import { mockState, uid } from './state'
import { cancelSimulation, emit, simulate, thread } from './fixtures-core'
import { mockContextPressure, mockTokenUsage } from './fixtures-data'
import { mockAssistantSnapshot, mockModelRouterSnapshot } from './connectors'
import { activeOnly, mockArchive, mockArchiveSnapshot, mockBlockedCommandReason, mockGitHygieneReport, mockRestore } from './runtime'
import { demoApprovalThread, demoRunStateEvents, mockGitDiff, mockGitStatus } from './workspace-fixtures'
import { captureBotConfig } from '../../../shared/bot-policy'
import type { TensorFoldStudioApi } from '../ipc'

export function createMockWorkspaceApi(): Pick<TensorFoldStudioApi, 'getThreads' | 'getArchives' | 'archiveItem' | 'restoreItem' | 'createThread' | 'createBotChat' | 'setThreadProject' | 'deleteThread' | 'renameThread' | 'setThreadEngine' | 'ompStatus' | 'configureThread' | 'setThreadMemoryPolicy' | 'patchThread' | 'setThreadWorkspace' | 'appendThreadNotice' | 'compactThread' | 'startRun' | 'cueRun' | 'cancelRun' | 'approve' | 'pickDirectory' | 'getTokenUsage' | 'getContextPressure' | 'getModelRouterSnapshot' | 'getAssistantSnapshot' | 'getBranch' | 'inspectGitRepository' | 'initGitRepository' | 'scanGitHygiene' | 'sanitizeGitHygiene' | 'listFiles' | 'resolveInstructions' | 'gitStatus' | 'gitDiff' | 'revertFiles' | 'checkpointList' | 'checkpointUndoLast' | 'checkpointDiffLast' | 'checkpointTurns' | 'checkpointRevertTurn' | 'checkpointForkTurn' | 'previewWithheld' | 'handoffThread' | 'commit' | 'exec' | 'onAgentEvent' | 'listSshHosts' | 'sshStatus' | 'saveSshHost' | 'removeSshHost' | 'testSshHost' | 'connectSshHost' | 'disconnectSshHost' | 'resolveSshSecret' | 'resolveSshTrust' | 'listRemoteDir' | 'onSshStatus' | 'onPersistenceFailure' | 'onSshSecretPrompt' | 'onSshTrustPrompt' | 'windowAction' | 'openExternal'> {
  let demoRunStatesSent = false
  return {
    getThreads: async () => activeOnly(mockState.threads),
    getArchives: async () => mockArchiveSnapshot(),
    archiveItem: async (kind, id) => mockArchive(kind, id),
    restoreItem: async (kind, id) => mockRestore(kind, id),
    createThread: async (cwd, kind = 'task', projectId = null, _commandId, incognito = false) => {
      const proj = projectId ? mockState.projects.find((p) => p.id === projectId) : null
      const t = thread(kind === 'chat' ? 'New chat' : 'New task', cwd || proj?.cwd || mockState.settings.defaultCwd || '~', 0, 'idle', [], kind, projectId)
      t.documentSourceIds = []
      if (incognito) t.incognito = true
      mockState.threads = [t, ...mockState.threads]
      return t
    },
    createBotChat: async (presetId) => {
      const preset = mockState.settings.agentPresets.find((candidate) => candidate.id === presetId)
      if (!preset) throw new Error('Bot not found.')
      const bot = captureBotConfig(preset, mockState.settings.providers)
      const t = thread(`${bot.name} chat`, bot.executionTarget === 'vm' ? '/workspace' : '', 0, 'idle', [], 'chat', null)
      Object.assign(t, { documentSourceIds: [], bot, providerId: bot.providerId, providerAccountId: bot.providerAccountId, model: bot.model, workspace: null })
      mockState.threads = [t, ...mockState.threads]
      return t
    },
    setThreadProject: async (id, projectId) => { const t = mockState.threads.find((x) => x.id === id); if (t) { t.projectId = projectId; const proj = projectId ? mockState.projects.find((p) => p.id === projectId) : null; if (proj?.cwd) t.cwd = proj.cwd } },
    deleteThread: async (id) => { mockArchive('thread', id) },
    renameThread: async (id, title) => { const t = mockState.threads.find((x) => x.id === id); if (t) { t.title = title; t.updatedAt = Date.now() } },
    setThreadEngine: async (id, engine) => { const t = mockState.threads.find((x) => x.id === id); if (t) t.engine = engine === 'native' ? undefined : engine },
    ompStatus: async () => ({ available: true, binary: '/usr/local/bin/omp', version: 'omp v18.1.18 (demo)' }),
    configureThread: async (id, providerId, model, providerAccountId = null) => {
      const t = mockState.threads.find((x) => x.id === id)
      if (t) { t.providerId = providerId; t.providerAccountId = providerAccountId; t.model = model }
    },
    setThreadMemoryPolicy: async (id, policy, expectedRevision) => {
      const thread = mockState.threads.find((t) => t.id === id)
      if (!thread || thread.archivedAt) throw new Error('Choose an active chat.')
      if (thread.status === 'running') throw new Error('Stop this run before changing memory choices.')
      if (threadMemoryPolicyRevision(thread) !== expectedRevision) throw new Error('Memory choices changed in another window.')
      const requested = parseThreadMemoryPolicy(policy)
      assertThreadMemoryTransport({ ...thread, memoryPolicy: requested }, mockState.settings.providers.find((p) => p.id === thread.providerId))
      Object.assign(thread, { memoryPolicy: requested, memoryPolicyRevision: expectedRevision + 1 })
      return thread
    },
    patchThread: async (id, patch) => {
      const t = mockState.threads.find((x) => x.id === id)!
      if (t && Object.prototype.hasOwnProperty.call(patch, 'documentSourceIds')) {
        const ids = validateDocumentSourceIds(patch.documentSourceIds)
        if (!ids || t.status === 'running' || t.archivedAt || t.bot || t.orchestratorHostId || t.workspace?.kind === 'ssh' || t.phoneLocalOnly || t.googleLocalOnly) throw new Error('Stop the eligible local chat before choosing document collections.')
        if (ids.some(id => !mockState.settings.documents.sources.some(source => source.id === id && source.enabled))) throw new Error('Choose enabled document collections configured in this profile.')
      }
      if (t) Object.assign(t, patch)
      return t
    },
    setThreadWorkspace: async (id, workspace, cwd) => {
      const t = mockState.threads.find((x) => x.id === id)!
      if (t) { t.workspace = workspace; t.cwd = workspace?.path ?? cwd ?? t.cwd }
      return t
    },
    appendThreadNotice: async (id, text) => { const t = mockState.threads.find((x) => x.id === id); if (t) t.timeline.push({ id: `notice-${t.timeline.length}`, kind: 'notice', text, createdAt: Date.now() } as any) },
    compactThread: async (id) => {
      const t = mockState.threads.find((x) => x.id === id)
      if (!t) throw new Error('Thread not found')
      const item = { id: uid(), kind: 'notice' as const, text: 'Context compacted. Folded earlier messages into a running summary.', createdAt: Date.now() }
      t.timeline = [...t.timeline, item]
      t.compactions = [...(t.compactions ?? []), { id: uid(), at: Date.now(), trigger: 'manual', messagesFolded: 4, tokensBefore: 9200, tokensAfter: 2100, summaryExcerpt: 'Demo context summary.', kindsDropped: ['assistant', 'tool'] }]
      emit({ runId: 'demo', threadId: id, type: 'item_add', item } as AgentEvent)
      return { ok: true, thread: t }
    },
    startRun: async ({ threadId, userText }: StartRunPayload) => {
      emit({ runId: 'demo', threadId, type: 'item_add', item: { id: uid(), kind: 'user', text: userText, createdAt: Date.now() } })
      simulate(threadId, userText)
      return { runId: 'demo' }
    },
    cueRun: async ({ threadId, userText }) => {
      const id = uid()
      emit({ runId: 'demo', threadId, type: 'item_add', item: { id, kind: 'cue', title: 'Guide', text: userText, state: 'pending', meta: 'queued', createdAt: Date.now() } })
      setTimeout(() => emit({ runId: 'demo', threadId, type: 'item_update', id, patch: { state: 'ok', meta: 'applied' } }), 350)
      return { queued: true, queueDepth: 1 }
    },
    cancelRun: async (threadId) => { cancelSimulation(threadId) },
    approve: async (callId) => {
      const threadId = demoApprovalThread(callId)
      if (!threadId) return
      emit({ runId: 'demo-waiting', threadId, type: 'status', status: 'done' })
      emit({ runId: 'demo-waiting', threadId, type: 'done' })
    },
    pickDirectory: async () => '/Users/example/Projects/web-dashboard',
    getTokenUsage: async () => mockTokenUsage(),
    getContextPressure: async () => mockContextPressure(),
    getModelRouterSnapshot: async () => mockModelRouterSnapshot(),
    getAssistantSnapshot: async () => mockAssistantSnapshot(),
    getBranch: async () => 'feat/dark-mode',
    inspectGitRepository: async (cwd) => ({
      cwd,
      exists: !cwd.includes('new-folder-without-git'),
      insideWorkTree: !cwd.includes('new-folder-without-git'),
      topLevel: cwd.includes('new-folder-without-git') ? null : cwd,
      branch: cwd.includes('new-folder-without-git') ? null : 'main',
      hasCommits: !cwd.includes('new-folder-without-git'),
      message: cwd.includes('new-folder-without-git') ? 'This folder is not initialized as a Git repository.' : undefined,
    }),
    initGitRepository: async (cwd) => ({ cwd, exists: true, insideWorkTree: true, topLevel: cwd, branch: 'main', hasCommits: true }),
    scanGitHygiene: async (cwd, message = ''): Promise<GitHygieneReport> => mockGitHygieneReport(cwd, message),
    sanitizeGitHygiene: async (_cwd, findings): Promise<GitHygieneSanitizeResult> => ({
      ok: true,
      changedFiles: (() => { mockState.mockGitHygieneSanitized = true; return [...new Set(findings.filter((f) => f.autoFixable && f.file).map((f) => f.file!))] })(),
      skippedFindings: findings.filter((f) => !f.autoFixable).map((f) => f.id),
    }),
    listFiles: async () => [
      'package.json', 'tsconfig.json', 'vite.config.ts', 'README.md', 'index.html',
      'src/App.tsx', 'src/main.tsx', 'src/index.css',
      'src/components/Sidebar.tsx', 'src/components/Header.tsx', 'src/components/ThemeToggle.tsx', 'src/components/Composer.tsx',
      'src/lib/store.ts', 'src/lib/theme.ts', 'src/lib/api.ts',
      'src/hooks/useTheme.ts', 'src/pages/Dashboard.tsx', 'src/pages/Settings.tsx',
      'src/styles/tokens.css', 'tests/App.test.tsx', 'tests/theme.test.ts',
    ],
    resolveInstructions: async (cwd, providerId = null, workspace = null) => {
      if (workspace?.kind === 'ssh') return { rootPath: workspace.path || cwd, providerId, nodes: [], cycles: [], resolvedContent: '', tokenEstimate: 0 }
      const shared = '# AGENTS.md (demo)\nProject conventions: use the design tokens, keep PRs small.'
      const overlay = 'Provider overlay: prefer concise diffs and cite file:line.'
      const resolvedContent = `<!-- shared: AGENTS.md -->\n${shared}\n\n<!-- imported: docs/architecture.md -->\nThe app is Electron + React; main process owns tools.${providerId ? `\n\n<!-- provider: .vontra/${providerId}.md -->\n${overlay}` : ''}`
      return {
        rootPath: cwd || '~/demo', providerId,
        nodes: [
          { id: '1', kind: 'shared', path: 'AGENTS.md', content: shared, imports: ['docs/architecture.md'], ignored: false, tokenEstimate: 28 },
          { id: '2', kind: 'imported', path: 'docs/architecture.md', content: 'The app is Electron + React; main process owns tools.', imports: [], ignored: false, tokenEstimate: 18 },
          ...(providerId ? [{ id: '3', kind: 'provider' as const, path: `.vontra/${providerId}.md`, content: overlay, imports: [], ignored: false, tokenEstimate: 14 }] : []),
        ],
        cycles: [],
        resolvedContent,
        tokenEstimate: Math.round(resolvedContent.length / 4),
      }
    },
    gitStatus: async (cwd): Promise<GitStatus> => mockGitStatus(cwd),
    gitDiff: async (cwd, paths): Promise<GitDiffFile[]> => mockGitDiff(cwd, paths),
    revertFiles: async () => { },
    checkpointList: async () => [],
    checkpointUndoLast: async () => ({ ok: false, error: 'Checkpoints require the desktop app.' }),
    checkpointDiffLast: async () => ({ ok: false, error: 'Checkpoints require the desktop app.' }),

    checkpointTurns: async ({ threadId }) => {
      const t = mockState.threads.find((x) => x.id === threadId)
      if (!t) return []
      const users = t.timeline.filter((i) => i.kind === 'user')
      return users.map((item, idx) => ({
        threadId,
        turn: idx + 1,
        createdAt: Date.now() - (users.length - idx) * 120_000,
        fileRef: t.kind === 'chat' ? null : `refs/vontra/checkpoints/${threadId}/turn/${idx + 1}`,
        fileSha: t.kind === 'chat' ? null : 'demo1234',
        conversationId: `demo-cvk-${threadId}-${idx + 1}`,
        fingerprint: `demo-fp-${idx + 1}`,
        messageCount: Math.max(0, t.timeline.indexOf(item)),
        branch: 'main',
        planId: null,
        label: `turn ${idx + 1}`,
      }))
    },
    checkpointRevertTurn: async ({ threadId, turn }) => {
      const t = mockState.threads.find((x) => x.id === threadId)
      if (!t) return { ok: false, turn, error: 'Thread not found.' }
      let seen = 0
      const cut = t.timeline.findIndex((i) => i.kind === 'user' && ++seen === turn)
      if (cut < 0) return { ok: false, turn, error: `No demo checkpoint for turn ${turn}.` }
      t.timeline = t.timeline.slice(0, cut)
      t.updatedAt = Date.now()
      return { ok: true, turn }
    },
    checkpointForkTurn: async ({ threadId, turn, worktree }) => {
      const src = mockState.threads.find((x) => x.id === threadId)
      if (!src) return { ok: false, error: 'Thread not found.' }
      let seen = 0
      const cut = src.timeline.findIndex((i) => i.kind === 'user' && ++seen === turn)
      if (cut < 0) return { ok: false, error: `No demo checkpoint for turn ${turn}.` }
      const demoWorktree = worktree ? `~/.vontra/worktrees/${threadId.slice(0, 8)}-turn${turn}-fork-1` : null
      const t: Thread = {
        ...src,
        id: uid(),
        title: `Fork of ${src.title}`,
        ...(demoWorktree ? { cwd: demoWorktree } : {}),
        apiMessages: [],
        timeline: [
          ...src.timeline.slice(0, cut),
          { id: uid(), kind: 'notice', text: `Forked from “${src.title}” at turn ${turn} (branch fork-1). This thread continues independently — the original is untouched.${demoWorktree ? `\nThis fork lives in its own worktree: ${demoWorktree} (simulated in the browser demo).` : ''}`, createdAt: Date.now() },
        ],
        createdAt: Date.now(),
        updatedAt: Date.now(),
        status: 'idle',
      }
      mockState.threads = [t, ...mockState.threads]
      return { ok: true, branch: 'fork-1', fileRef: null, thread: t, worktreePath: demoWorktree, worktreeError: null }
    },

    previewWithheld: async ({ text, terms }) => {
      let total = 0
      let out = String(text ?? '').replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g, () => { total++; return `[EMAIL_${total}]` })
      for (const [i, term] of (terms ?? []).filter((t) => t.trim()).entries()) {
        out = out.replace(new RegExp(`\\b${term.trim().replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'gi'), () => { total++; return `[TERM_${i + 1}]` })
      }
      return { text: out, withheld: total ? 'terms and emails (demo preview)' : '', total }
    },
    handoffThread: async ({ threadId, toProviderId, toModel }) => {
      const src = mockState.threads.find((x) => x.id === threadId)
      if (!src) throw new Error('The source thread no longer exists.')
      if (src.providerId === toProviderId) throw new Error('Handoff requires a different target engine.')
      const toProvider = mockState.settings.providers.find((p) => p.id === toProviderId)
      const t: Thread = {
        ...src,
        id: uid(),
        title: `${src.title} → ${toProvider?.name ?? toProviderId}`,
        providerId: toProviderId,
        model: toModel?.trim() || src.model,
        apiMessages: [],
        timeline: [{ id: uid(), kind: 'notice', text: `Handed off from “${src.title}” to ${toProvider?.name ?? toProviderId}. In the desktop app the flattened transcript runs as the first turn (simulated here).`, createdAt: Date.now() }],
        createdAt: Date.now(),
        updatedAt: Date.now(),
        status: 'idle',
      }
      mockState.threads = [t, ...mockState.threads]
      return t
    },
    commit: async (cwd, message): Promise<GitCommitResult> => ({ ok: true, hygiene: mockGitHygieneReport(cwd, `${message} clean`), message: message.replace(/^co-authored-by:.*$/gim, '').trim() || 'TensorFold changes' }),
    exec: async (_cwd, command) => {
      await new Promise((r) => setTimeout(r, 250))
      const blocked = mockBlockedCommandReason(command)
      if (blocked) return { output: `Blocked by TensorFold safety policy: ${blocked}.`, code: 126 }
      if (/^ls/.test(command)) return { output: 'src/        electron/    package.json\nindex.html  README.md    tsconfig.json', code: 0 }
      if (/git status/.test(command)) return { output: 'On branch main\nnothing to commit, working tree clean', code: 0 }
      return { output: `(demo) ran: ${command}\nThe desktop app runs this in your workspace and streams real output.`, code: 0 }
    },
    onAgentEvent: (cb) => {
      mockState.listener = cb
      // Replay the demo's live run states once, so the sidebar shows running and waiting rows.
      if (!demoRunStatesSent) { demoRunStatesSent = true; setTimeout(() => demoRunStateEvents().forEach(emit), 300) }
      return () => { mockState.listener = null }
    },
    // SSH remote workspaces — demo stubs (real SSH only runs in the desktop app)
    listSshHosts: async () => mockState.settings.sshHosts ?? [],
    sshStatus: async (hostId) => ({ hostId, phase: 'idle' as const }),
    saveSshHost: async (host) => {
      const list = mockState.settings.sshHosts ?? []
      const idx = list.findIndex((h) => h.id === host.id)
      mockState.settings.sshHosts = idx >= 0 ? list.map((h) => (h.id === host.id ? host : h)) : [...list, host]
      return mockState.settings.sshHosts
    },
    removeSshHost: async (hostId) => {
      mockState.settings.sshHosts = (mockState.settings.sshHosts ?? []).filter((h) => h.id !== hostId)
      return mockState.settings.sshHosts
    },
    testSshHost: async () => ({ ok: false, message: 'SSH connections only run in the desktop app.' }),
    connectSshHost: async (hostId) => ({ hostId, phase: 'error' as const, lastError: 'SSH only runs in the desktop app.' }),
    disconnectSshHost: async () => ({ ok: true }),
    resolveSshSecret: async () => ({ ok: true }),
    resolveSshTrust: async () => ({ ok: true }),
    listRemoteDir: async () => [],
    onSshStatus: () => () => { },
    onPersistenceFailure: () => () => { },
    onSshSecretPrompt: () => () => { },
    onSshTrustPrompt: () => () => { },
    windowAction: async () => { },
    openExternal: async () => { },
  }
}
