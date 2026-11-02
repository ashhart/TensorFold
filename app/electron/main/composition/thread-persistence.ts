import type { SshManager } from '../host/ssh'
import path from 'node:path'
import os from 'node:os'
import { randomUUID } from 'node:crypto'
import { loadTurnCheckpoints, saveTurnCheckpoints, saveProviderStatus } from '../../store'
import { store } from '../state'
import { createCheckpointService } from '../host/checkpoint-runtime'
import { createProviderStatusRuntime } from '../host/provider-status-runtime'
import { sendToRenderer } from '../host/renderer-events'
import { resolveWorkspaceCwd } from '../host/workspace'

export function createApplicationThreadPersistence(sshManager: SshManager) {
  const providerStatusRuntime = createProviderStatusRuntime({
    saveProviderStatus,
    sendToRenderer,
  })
  function recordProviderRateLimits(providerId: string, accountId: string | null, rawSnapshot: unknown): void {
    return providerStatusRuntime.record(providerId, accountId, rawSnapshot)
  }

  const checkpoints = createCheckpointService({
    resolveBackend: (ws) => sshManager.backendForWorkspace(ws),
    resolveCwd: (cwd) => resolveWorkspaceCwd(cwd),
    conversationDeps: { clock: { now: () => Date.now() }, newId: () => randomUUID() },
    persistence: { load: loadTurnCheckpoints, save: saveTurnCheckpoints },
    isPersistableThread: (threadId) => {
      const thread = store.threads.find((candidate) => candidate.id === threadId)
      return !!thread && !thread.incognito
    },

    forkWorktreePath: ({ threadId, turn, branch }) =>
      path.join(os.homedir(), '.vontra', 'worktrees', `${threadId.slice(0, 8)}-turn${turn}-${branch.replace(/[^a-zA-Z0-9_-]/g, '')}`),
  })

  return { providerStatusRuntime, checkpoints, recordProviderRateLimits }
}
