import { randomUUID } from 'node:crypto'
import { promises as fs } from 'node:fs'
import { resolveProviderModel } from '../../../agent/provider-accounts'
import { scanImportableSessions, type SessionFileSystem } from '../../../agent/session-import'
import type { EventSpineHost } from '../../../agent/event-spine-host'
import type { ImportableSession, ImportSessionSource, Thread } from '../../../agent/types'
import type { RemoteControlEvent } from '../../../agent/types'
import { auditPreview } from '../../../agent/security'
import { store } from '../../state'
import { auditEvent, requireSecurityRole } from '../audit'
import { dirExists } from '../workspace'
import { handle } from '../ipc'
import { persistThreads } from '../persistence'
import { selectedProviderAccountId } from '../provider-runtime'

interface RemoteServerLike {
  publish(event: Pick<RemoteControlEvent, 'kind' | 'data'>): void
}

export interface SessionIpcDeps {
  spine: () => EventSpineHost | null
  remoteServer: RemoteServerLike
}

export function registerSessionIpc(deps: SessionIpcDeps) {
  const { spine, remoteServer } = deps

// ---- external harness session import (VST-033) ----
// Scan ~/.codex/sessions rollouts + ~/.claude/projects session files for sessions TensorFold can
// adopt as threads, and create a thread seeded with the matching engine resume cursor so the
// next turn continues the SAME harness session instead of starting fresh.
handle('sessions:importable', async (): Promise<ImportableSession[]> => {
  requireSecurityRole('operator', 'scan external harness sessions')
  const os = await import('node:os')
  const path = await import('node:path')
  const roots: Record<ImportSessionSource, { root: string; accept: (file: string) => boolean }> = {
    codex: {
      root: path.join(os.homedir(), '.codex', 'sessions'),
      accept: (file) => path.basename(file).startsWith('rollout-') && file.endsWith('.jsonl'),
    },
    claude: {
      root: path.join(os.homedir(), '.claude', 'projects'),
      accept: (file) => file.endsWith('.jsonl'),
    },
  }
  const walk = async (root: string, accept: (file: string) => boolean): Promise<string[]> => {
    const out: string[] = []
    const stack = [root]
    while (stack.length) {
      const dir = stack.pop()!
      let entries: Array<{ name: string; isDirectory(): boolean; isFile(): boolean }> = []
      try { entries = await fs.readdir(dir, { withFileTypes: true }) } catch { continue }
      for (const entry of entries) {
        const full = path.join(dir, entry.name)
        if (entry.isDirectory()) stack.push(full)
        else if (entry.isFile() && accept(full)) out.push(full)
      }
    }
    return out
  }
  const sessionFs: SessionFileSystem = {
    async listFiles(source) {
      const spec = roots[source]
      const files = await walk(spec.root, spec.accept)
      const stats = await Promise.all(files.map(async (file) => ({ file, mtime: (await fs.stat(file).catch(() => null))?.mtimeMs ?? 0 })))
      // Newest first so the per-source file cap keeps the sessions a user actually wants.
      return stats.sort((a, b) => b.mtime - a.mtime).map((entry) => entry.file)
    },
    readFile: (file) => fs.readFile(file, 'utf8'),
  }
  return scanImportableSessions(sessionFs)
})

handle('sessions:import', async (_e, input: { sessionId: string; source: ImportSessionSource; title?: string; cwd?: string | null; model?: string | null }): Promise<Thread> => {
  requireSecurityRole('operator', 'import an external harness session')
  const sessionId = String(input?.sessionId ?? '').trim()
  const source: ImportSessionSource = input?.source === 'claude' ? 'claude' : 'codex'
  if (!sessionId) throw new Error('A session id is required to import.')
  // The cursor is rebuilt from source+id below (never trust a renderer-supplied cursor
  // object), and the session's cwd is kept only when it still exists on this machine.
  const cwd = input?.cwd && (await dirExists(input.cwd)) ? input.cwd : ''
  const wantedTransport = source === 'codex' ? 'codex-cli' : 'claude-cli'
  const provider = store.settings.providers.find((p) => p.transport === wantedTransport)
    ?? store.settings.providers.find((p) => p.id === store.settings.activeProviderId)
    ?? null
  const title = (input?.title ?? '').trim() || `Imported ${source === 'codex' ? 'Codex' : 'Claude'} session`
  const sourceName = source === 'codex' ? 'Codex' : 'Claude Code'
  const t: Thread = {
    id: randomUUID(),
    title,
    kind: 'task',
    projectId: null,
    cwd,
    workspace: null,
    providerId: provider?.id ?? null,
    providerAccountId: provider ? selectedProviderAccountId(provider, store.settings.activeProviderAccountId ?? undefined) : null,
    model: input?.model?.trim() || (provider ? resolveProviderModel(provider, null, store.settings.activeModel) : store.settings.activeModel),
    createdAt: Date.now(),
    updatedAt: Date.now(),
    status: 'idle',
    apiMessages: [],
    timeline: [{
      id: randomUUID(),
      kind: 'notice',
      text: `Imported from a ${sourceName} session (${sessionId.slice(0, 8)}…). The next message resumes that session with its full server-side history.`,
      createdAt: Date.now(),
    }],
    ...(source === 'codex' ? { codexResumeCursor: { threadId: sessionId } } : { claudeResumeCursor: { sessionId } }),
  }
  store.threads.unshift(t)
  persistThreads()
  spine()?.executeCommand(
    { commandId: randomUUID(), aggregate: 'thread', streamId: t.id, kind: 'thread.import', actor: 'user' },
    () => ({ aggregate: 'thread', streamId: t.id, type: 'thread.imported', payload: { source, sessionId } }),
  )
  store.win?.webContents.send('thread:update', t)
  remoteServer.publish({ kind: 'thread', data: t })
  await auditEvent('session.import', { source, sessionId: auditPreview(sessionId), cwd: cwd ? auditPreview(cwd) : null })
  return t
})

}
