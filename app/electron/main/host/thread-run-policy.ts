import type { Provider, Thread, WorkspaceLocation } from '../../agent/types'
import type { ThreadRuntimeDeps } from './thread-runtime'
import { resolveWorkspaceCwd } from './workspace'
import { resolvePairedWorkspace } from '../../agent/local-workspace-policy'

export async function recordPreTurnCheckpoint(input: {
  threadId: string
  incognito?: boolean
  turn: number
  target: { cwd?: string | null; workspace?: WorkspaceLocation | null; kind?: Thread['kind'] }
  conversation: { apiMessages: unknown[]; timeline: unknown[]; goal?: string; goalStatus?: string; plan?: unknown[]; status?: string }
}, deps: Pick<ThreadRuntimeDeps, 'captureCheckpoint' | 'recordTurnCheckpoint'>): Promise<void> {
  if (input.incognito) return
  let fileSha: string | null = null
  if (deps.captureCheckpoint && input.target.kind !== 'chat') {
    try {
      const captured = await deps.captureCheckpoint(input.threadId, input.turn, input.target)
      if (captured.ok && captured.sha) fileSha = captured.sha
    } catch { /* file checkpoints are best-effort; still retain the conversation snapshot */ }
  }
  try { deps.recordTurnCheckpoint?.(input.threadId, input.turn, input.conversation, fileSha) } catch { /* best effort */ }
}

export async function resolveThreadRunCwd(thread: Pick<Thread, 'kind' | 'cwd' | 'workspaceBoundary'>, workspace: WorkspaceLocation | null, boundary = thread.workspaceBoundary): Promise<string> {
  if (thread.kind === 'chat') return thread.cwd
  if (boundary === 'home') return resolvePairedWorkspace(thread.cwd, workspace)
  if (workspace?.kind === 'ssh') {
    const remotePath = String(workspace.path || '').trim()
    if (!remotePath) throw new Error('Choose a remote workspace folder before running workspace tools.')
    return remotePath
  }
  const localPath = workspace?.kind === 'local' && workspace.path ? workspace.path : thread.cwd
  return resolveWorkspaceCwd(localPath)
}

const LOCAL_DELEGATED_TRANSPORTS = new Set<NonNullable<Provider['transport']>>([
  'codex-cli', 'claude-cli', 'opencode', 'kilo', 'pi-cli', 'cursor-cli', 'gemini-cli',
])

export function assertSshThreadTransportSupported(workspace: WorkspaceLocation | null, provider: Provider | null | undefined): void {
  if (workspace?.kind === 'ssh' && provider?.transport && LOCAL_DELEGATED_TRANSPORTS.has(provider.transport)) {
    throw new Error(`${provider.name || provider.transport} runs on this Mac and cannot safely use an SSH workspace. Choose an HTTP/model transport with SSH-backed tools, or run the delegated engine on the remote Orchestrator.`)
  }
}

export function assertSshToolContextConnected(workspace: WorkspaceLocation | null, toolContext: { backend?: unknown } | null | undefined): void {
  if (workspace?.kind === 'ssh' && !toolContext?.backend) {
    throw new Error('The SSH workspace disconnected during startup. Reconnect it before running the agent.')
  }
}

export function assertIncognitoThreadTransportSupported(thread: Pick<Thread, 'incognito' | 'engine'>, provider?: Pick<Provider, 'transport'>): void {
  if (thread.incognito && (thread.engine === 'omp' || provider?.transport && provider.transport !== 'chat-completions')) {
    throw new Error('Incognito requires a TensorFold API chat. This agent runtime keeps its own session history; choose an API model or start an ordinary saved task.')
  }
}
