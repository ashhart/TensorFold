import { isExternalProvider } from '../../../shared/provider-trust'
import { effectiveThreadMemoryPolicy, threadMemoryPolicyRevision } from '../../../shared/thread-memory-policy'
import { botToolBlockReason, botToolScopeKey } from '../../../shared/bot-tool-policy'
import { assertGoogleContextTransport } from '../../../shared/google-workspace-privacy'
import type { Provider, Thread, ThreadMemoryPolicy } from '../../agent/types'
import { isPersistentGoogleWorkspaceTool, runGoogleWorkspaceTool, type GoogleWorkspaceToolService } from '../../agent/google-workspace-tools'

export interface GoogleWorkspaceToolRun {
  thread?: Thread
  provider?: Provider
  model?: string
  signal?: AbortSignal
  subAgentId?: string
  memoryPolicy?: ThreadMemoryPolicy
  memoryOwner?: { id: string; revision: number }
}
export interface GoogleWorkspaceToolRuntimeDeps {
  getThread(id: string): Thread | undefined
  resolveProvider(thread: Thread): Provider | undefined
  service(): GoogleWorkspaceToolService
  disabledCapabilities(): string[]
  disabledTools(): string[]
  assertRecallSourcesAllowed(): void
  hasExternalResearchEnabled(): boolean
  markContext(thread: Thread): Promise<void>
}
/** Concurrent first reads share the same durable marker write; a failed write stays fail-closed. */
export function createGoogleContextMarker(deps: { persist(): Promise<void>; publish(thread: Thread): void }) {
  const pending = new WeakMap<Thread, Promise<void>>()
  const needsPersistence = new WeakSet<Thread>()
  return (thread: Thread): Promise<void> => {
    const existing = pending.get(thread)
    if (existing) return existing
    if (thread.googleLocalOnly && !needsPersistence.has(thread)) return Promise.resolve()
    thread.googleLocalOnly = true
    needsPersistence.add(thread)
    const job = deps.persist().then(() => { needsPersistence.delete(thread); deps.publish(thread) }).finally(() => { if (pending.get(thread) === job) pending.delete(thread) })
    pending.set(thread, job)
    return job
  }
}
function route(provider?: Provider): string { return JSON.stringify([provider?.id, provider?.kind, provider?.source, provider?.transport, provider?.baseUrl]) }
function threadRoute(thread?: Thread): string { return JSON.stringify([thread?.providerId, thread?.providerAccountId, thread?.model, thread?.engine, thread?.orchestratorHostId, thread?.projectId, thread?.workspace, thread?.phoneLocalOnly, thread?.incognito]) }

export function createGoogleWorkspaceToolRunner(run: GoogleWorkspaceToolRun | undefined, deps: GoogleWorkspaceToolRuntimeDeps) {
  const captured = run?.thread
  const capturedRoute = threadRoute(captured)
  const providerRoute = route(run?.provider)
  const revision = captured ? threadMemoryPolicyRevision(captured) : null
  let scope: string
  try { scope = JSON.stringify(captured?.bot?.toolScope) ?? 'null' } catch { scope = '[invalid saved scope]' }
  return async (name: string, args: Record<string, any>) => {
    const assertAllowed = () => {
      if (!captured || !run?.provider || !run.signal || !run.model || run.signal.aborted) throw new Error('Google tools require an active, explicitly bound local model conversation.')
      const live = deps.getThread(captured.id)
      if (live !== captured || live.archivedAt || threadRoute(live) !== capturedRoute || threadMemoryPolicyRevision(live) !== revision) throw new Error('The Google tool conversation or privacy choices changed. Start a new run.')
      effectiveThreadMemoryPolicy(live, run.memoryPolicy)
      if (run.memoryOwner && (run.memoryOwner.id !== live.id || run.memoryOwner.revision !== revision)) throw new Error('Google tools cannot use a different conversation owner.')
      if (run.subAgentId || live.bot?.executionTarget === 'vm' || live.orchestratorHostId || live.workspace?.kind === 'ssh' || live.engine && live.engine !== 'native' || live.phoneLocalOnly) throw new Error('Google account tools are unavailable to remote, child, phone-confined or VM runs.')
      const provider = deps.resolveProvider(live)
      if (!provider || isExternalProvider(run.provider) || isExternalProvider(provider) || route(provider) !== providerRoute || deps.hasExternalResearchEnabled()) throw new Error('Google account data stays on a model running on this Mac. External, LAN and research routes are unavailable.')
      assertGoogleContextTransport(live, provider)
      if (deps.disabledCapabilities().includes('connector:google-workspace') || deps.disabledTools().includes(name)) throw new Error('Google Workspace capability or tool is disabled for this run.')
      botToolScopeKey(live.bot?.toolScope)
      if ((JSON.stringify(live.bot?.toolScope) ?? 'null') !== scope) throw new Error('Bot tool permissions changed during this run.')
      const botProblem = botToolBlockReason(live.bot, name, args)
      if (botProblem) throw new Error(botProblem)
      if (isPersistentGoogleWorkspaceTool(name) && live.incognito) throw new Error('Incognito cannot save Google drafts or pending action reviews. Use an ordinary saved chat.')
      deps.assertRecallSourcesAllowed()
    }
    assertAllowed()
    const result = await runGoogleWorkspaceTool(name, args, deps.service(), assertAllowed)
    assertAllowed()
    await deps.markContext(captured!)
    assertAllowed()
    return result
  }
}
