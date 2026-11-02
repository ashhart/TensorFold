import type { Thread, CliToolStatus } from '../types'
import { mockState, uid } from './state'
import type { TensorFoldStudioApi } from '../ipc'

export function createMockToolsApi(): Pick<TensorFoldStudioApi, 'listClis' | 'listCliTools' | 'spineReadSince' | 'spineReceipt' | 'spineAdvanceCursor' | 'listImportableSessions' | 'importExternalSession' | 'startTerminal' | 'writeTerminal' | 'resizeTerminal' | 'killTerminal' | 'renameTerminal' | 'listTerminals' | 'restorableTerminals' | 'restoreTerminal' | 'terminalBacklog' | 'terminalSince' | 'onTerminalData' | 'onTerminalStatus' | 'onTerminalSessions'> {

  return {
    // Browser demo: pretend the plain shell + Claude Code CLIs are installed (no real PATH probe here).
    listClis: async () => [
      { id: 'shell', name: 'Shell', short: 'zsh', command: null, resumeCommand: null, color: '#8B8B97', icon: 'terminal', builtin: true },
      { id: 'claude', name: 'Claude Code', short: 'claude', command: 'claude', resumeCommand: 'claude --continue', color: '#D97757', icon: 'spark', builtin: true },
    ],
    listCliTools: async (): Promise<CliToolStatus[]> => [
      { id: 'shell', name: 'Shell', short: 'zsh', command: null, resumeCommand: null, color: '#8B8B97', icon: 'terminal', builtin: true, installed: true, version: null, effectiveCommand: null, configuredPath: null, providerIds: [] },
      { id: 'codex', name: 'Codex', short: 'codex', command: 'codex', resumeCommand: 'codex resume --last', color: '#2FB9B0', icon: 'code', builtin: true, installed: true, version: 'codex-cli 0.142.5', effectiveCommand: 'codex', configuredPath: null, providerIds: ['codex-cli'], latestVersion: '0.150.0', updateCommand: 'npm i -g @openai/codex', advisory: { status: 'update_available', currentVersion: '0.142.5', latestVersion: '0.150.0', updateCommand: 'npm i -g @openai/codex', canUpdate: true, message: 'Codex 0.150.0 is available (you have 0.142.5).' } },
      { id: 'claude', name: 'Claude Code', short: 'claude', command: 'claude', resumeCommand: 'claude --resume', color: '#D97757', icon: 'spark', builtin: true, installed: true, version: '2.1.280 (Claude Code)', effectiveCommand: 'claude', configuredPath: null, providerIds: ['claude-cli'] },
      { id: 'cursor', name: 'Cursor Agent', short: 'cursor', command: 'cursor-agent', resumeCommand: null, color: '#8B8B97', icon: 'spark', builtin: true, installed: false, version: null, effectiveCommand: 'cursor-agent', configuredPath: null, providerIds: [] },
      { id: 'gemini', name: 'Gemini', short: 'gemini', command: 'gemini', resumeCommand: null, color: '#6E7BF2', icon: 'globe', builtin: true, installed: false, version: null, effectiveCommand: 'gemini', configuredPath: null, providerIds: [] },
      { id: 'grok', name: 'Grok', short: 'grok', command: 'grok', resumeCommand: null, color: '#4DA3FF', icon: 'spark', builtin: true, installed: false, version: null, effectiveCommand: 'grok', configuredPath: null, providerIds: [] },
      // A wrapper script whose Python venv is gone: the binary is there but --version fails.
      { id: 'hermes', name: 'Hermes', short: 'hermes', command: 'hermes', resumeCommand: 'hermes --resume', color: '#C9A54E', icon: 'spark', builtin: true, installed: true, state: 'broken', problem: '~/.local/bin/hermes: line 4: ~/.hermes/venv/bin/python: No such file or directory', version: null, effectiveCommand: 'hermes', configuredPath: null, providerIds: [] },
      { id: 'opencode', name: 'OpenCode', short: 'opencode', command: 'opencode', resumeCommand: 'opencode --continue', color: '#E06C9F', icon: 'code', builtin: true, installed: true, version: '1.14.20', effectiveCommand: 'opencode', configuredPath: null, providerIds: [] },
      { id: 'kilo', name: 'Kilo', short: 'kilo', command: 'kilo', resumeCommand: null, color: '#B48AF7', icon: 'code', builtin: true, installed: false, version: null, effectiveCommand: 'kilo', configuredPath: null, providerIds: [] },
      { id: 'pi', name: 'Pi', short: 'pi', command: 'pi', resumeCommand: 'pi --resume', color: '#5BC873', icon: 'cpu', builtin: true, installed: true, version: '0.80.3', effectiveCommand: 'pi', configuredPath: null, providerIds: [] },
    ],

    spineReadSince: async () => ({ events: [], hasMore: false, snapshotRequired: false }),
    spineReceipt: async () => null,
    spineAdvanceCursor: async (input) => input.sequence,
    // Browser demo: two importable external-harness sessions so the import flow is visible.
    listImportableSessions: async () => [
      { sessionId: 'demo-codex-session-1', source: 'codex' as const, title: 'Wire up the payments webhook', cwd: '/Users/demo/work/api', model: 'gpt-5.5', createdAt: Date.now() - 3 * 60 * 60_000, updatedAt: Date.now() - 20 * 60_000, messageCount: 34, resumeCursor: { threadId: 'demo-codex-session-1' }, truncated: false },
      { sessionId: 'demo-claude-session-1', source: 'claude' as const, title: 'Refactor auth middleware', cwd: '/Users/demo/work/api', model: 'claude-fable-5', createdAt: Date.now() - 26 * 60 * 60_000, updatedAt: Date.now() - 2 * 60 * 60_000, messageCount: 58, resumeCursor: { sessionId: 'demo-claude-session-1' }, truncated: false },
    ],
    importExternalSession: async (input) => {
      const sourceName = input.source === 'codex' ? 'Codex' : 'Claude Code'
      const thread: Thread = {
        id: `mock-import-${uid()}`,
        title: input.title || `Imported ${sourceName} session`,
        kind: 'task',
        projectId: null,
        cwd: input.cwd ?? '',
        workspace: null,
        providerId: null,
        providerAccountId: null,
        model: input.model ?? null,
        createdAt: Date.now(),
        updatedAt: Date.now(),
        status: 'idle',
        apiMessages: [],
        timeline: [{ id: `mock-import-notice-${uid()}`, kind: 'notice', text: `Imported from a ${sourceName} session (${input.sessionId.slice(0, 8)}…). The next message resumes that session.`, createdAt: Date.now() }],
        ...(input.source === 'codex' ? { codexResumeCursor: { threadId: input.sessionId } } : { claudeResumeCursor: { sessionId: input.sessionId } }),
      }
      mockState.threads.unshift(thread)
      return thread
    },
    startTerminal: async (projectId, cliId, opts) => ({ id: `mock-term-${uid()}`, projectId, cliId, title: opts?.label?.trim() || (cliId === 'shell' ? 'Shell' : cliId), status: 'idle', createdAt: Date.now() }),
    writeTerminal: async () => { },
    resizeTerminal: async () => { },
    killTerminal: async () => { },
    renameTerminal: async () => { },
    listTerminals: async () => [],
    // Browser demo has no persisted PTYs, so nothing is ever restorable; restore is a no-op echo.
    restorableTerminals: async () => [],
    restoreTerminal: async (sessionId) => ({ id: sessionId, projectId: '', cliId: 'shell', title: 'Shell', status: 'idle', createdAt: Date.now() }),
    terminalBacklog: async () => '$ npm test\n\x1b[32mPASS\x1b[0m src/app.test.ts (2.1s)\n\x1b[31mFAIL\x1b[0m src/auth.test.ts — 1 failing\n  ● login › rejects a bad password\n',
    terminalSince: async () => ({ chunks: [], gap: false, head: 0 }),
    onTerminalData: () => () => { },
    onTerminalStatus: () => () => { },
    onTerminalSessions: () => () => { },
  }
}
