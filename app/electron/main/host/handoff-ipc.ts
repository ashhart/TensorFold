import { captureThreadMemoryPolicy, assertThreadMemoryTransport } from '../../../shared/thread-memory-policy'
import { randomUUID } from 'node:crypto'
import { classifyProvider } from '../../../shared/provider-trust'
import { latestPlanText, runPlanPrompt } from '../../agent/planning-guard'
import { resolveProviderModel } from '../../agent/provider-accounts'
import { auditPreview } from '../../agent/security'
import { spineEligible } from '../../agent/incognito'
import { buildHandoff } from '../../agent/turn-checkpoints'
import type { ApiMessage, Provider, StartRunPayload, Thread } from '../../agent/types'
import type { EventSpineHost } from '../../agent/event-spine-host'
import { store } from '../state'
import { auditEvent, requireSecurityRole } from './audit'
import { handle } from './ipc'
import { threadTurnNumber } from './checkpoint-runtime'
import { assertSshThreadTransportSupported } from './thread-runtime'
import { runtimeProvider, selectedProviderAccountId } from './provider-runtime'

interface RemoteServerLike {
  publish(event: { kind: string; data: unknown }): void
}

interface HandoffIpcDeps {
  spine: () => EventSpineHost | null
  sendToRenderer(channel: string, payload: unknown): void
  remoteServer: RemoteServerLike
  startThreadRun(payload: StartRunPayload): Promise<{ runId: string }>
  persistThreads(): void
}

/**
 * Cross-engine handoff (VST-036): flatten the source conversation into a seed message and
 * continue it on a DIFFERENT engine in a fresh thread. The transcript never round-trips
 * through the renderer — the host rebuilds it from the live thread (same rule as session
 * import), and the new thread carries NO resume cursors: a different engine cannot resume
 * the source harness session, the seed transcript is its context.
 */
export function registerThreadHandoffIpc(deps: HandoffIpcDeps) {
  // intent 'run-plan': an external model's planning reply goes to a trusted local engine to
  // carry out; the seed turn quotes the plan (already rehydrated) so the engine acts on it.
  handle('thread:handoff', async (_e, input: { threadId: string; toProviderId: string; toModel?: string | null; autoStart?: boolean; intent?: 'run-plan' }): Promise<Thread> => {
    requireSecurityRole('operator', 'hand a thread off to another engine')
    const source = store.threads.find((t) => t.id === String(input?.threadId ?? ''))
    if (!source) throw new Error('The source thread no longer exists.')
    if (source.archivedAt) throw new Error('Restore the archived source thread before handing it off.')
    if (source.bot !== undefined) throw new Error('Bot conversations keep their captured model and permissions. Edit the bot and start a new chat instead of handing this conversation to another engine.')
    const sourceProject = source.projectId ? store.projects.find((project) => project.id === source.projectId) ?? null : null
    if (source.projectId && (!sourceProject || sourceProject.archivedAt)) throw new Error('Restore or choose an active source project before handing this thread off.')
    const toProvider = store.settings.providers.find((p) => p.id === String(input?.toProviderId ?? ''))
    if (!toProvider) throw new Error('Choose a target engine for the handoff.')
    if (toProvider.id === source.providerId) throw new Error('Handoff requires a different target engine.')
    const accountId = selectedProviderAccountId(toProvider as Provider, store.settings.activeProviderAccountId ?? undefined)
    const targetProvider = runtimeProvider(toProvider as Provider, accountId)
    if (source.phoneLocalOnly && (source.orchestratorHostId || classifyProvider(targetProvider) !== 'trusted-local')) throw new Error('Phone conversations can only hand off to an API model running on this Mac; external, remote and CLI engines cannot receive their copied context.')
    assertThreadMemoryTransport({ ...source, engine: 'native' }, toProvider)
    const runPlan = input?.intent === 'run-plan'
    if (runPlan && classifyProvider(toProvider) !== 'trusted-local') throw new Error('A plan can only run on the TensorFold engine or another local provider on this Mac.')
    const plan = runPlan ? latestPlanText(source.apiMessages) : ''
    if (runPlan && !plan) throw new Error('There is no plan in this thread to run yet.')
    const sourceWorkspace = source.workspace ?? sourceProject?.workspace ?? null
    if (source.phoneLocalOnly && sourceWorkspace?.kind === 'ssh') throw new Error('Phone conversations cannot hand off to an SSH workspace.')
    // Delegated engines run as local processes. Reject an SSH/local-lookalike handoff before a
    // new thread is persisted, even when autoStart is false, so the saved configuration is honest.
    assertSshThreadTransportSupported(sourceWorkspace, toProvider as Provider)
    const fromProvider = store.settings.providers.find((p) => p.id === source.providerId)
    const payload = buildHandoff({
      fromThreadId: source.id,
      fromEngine: fromProvider?.name || source.providerId || 'the previous engine',
      toEngine: toProvider.name || toProvider.id,
      atTurn: threadTurnNumber(source.timeline ?? []),
      conversation: { apiMessages: (source.apiMessages ?? []) as never, timeline: (source.timeline ?? []) as never },
    })
    const now = Date.now()
    // Strong-form handoff (VST-049): seed the new thread's apiMessages with the
    // structured transcript (preserving role structure) instead of one flattened
    // user message. The structured transcript keeps system/user/assistant/tool
    // roles intact so the receiving engine sees a real conversation rather than
    // one giant user blob. The seedMessage stays as a fallback (below) for engines
    // whose fresh-session path can't accept a structured history.
    const seededApiMessages = (payload.transcript ?? []).filter((m) => m && typeof m === 'object')
      .map((m): ApiMessage => ({
        role: (m.role === 'system' || m.role === 'user' || m.role === 'assistant' || m.role === 'tool') ? m.role : 'user',
        content: typeof m.content === 'string' ? m.content : (m.content == null ? '' : String(m.content)),
        ...(typeof m.tool_call_id === 'string' ? { tool_call_id: m.tool_call_id } : {}),
        ...(typeof m.name === 'string' ? { name: m.name } : {}),
        ...(Array.isArray(m.tool_calls) ? { tool_calls: m.tool_calls as ApiMessage['tool_calls'] } : {}),
      }))
    const t: Thread = {
      id: randomUUID(),
      title: `${source.title} → ${toProvider.name || toProvider.id}`,
      kind: source.kind,
      memoryPolicy: { ...captureThreadMemoryPolicy(source) },
      memoryPolicyRevision: 0,
      ...(source.phoneLocalOnly ? { phoneLocalOnly: true } : {}),
      projectId: source.projectId ?? null,
      cwd: source.cwd,
      workspace: sourceWorkspace,
      parentId: source.id,
      forkedAt: now,
      orchestratorHostId: source.orchestratorHostId,
      providerId: toProvider.id,
      providerAccountId: accountId,
      model: input?.toModel?.trim() || resolveProviderModel(toProvider as Provider, accountId, store.settings.activeModel),
      createdAt: now,
      updatedAt: now,
      status: 'idle',
      apiMessages: seededApiMessages,
      timeline: [{
        id: randomUUID(),
        kind: 'notice',
        text: `Handed off from “${source.title}” (${payload.fromEngine} → ${payload.toEngine}) at turn ${payload.atTurn}. ${seededApiMessages.length ? `${seededApiMessages.length} messages seed the new thread` : 'The flattened transcript seeds the first turn'}; the original thread is untouched.`,
        createdAt: now,
      }],
      ...(source.incognito ? { incognito: true } : {}),
    }
    store.threads.unshift(t)
    deps.persistThreads()
    if (spineEligible(t)) {
      deps.spine()?.executeCommand(
        { commandId: randomUUID(), aggregate: 'thread', streamId: t.id, kind: 'thread.handoff', actor: 'user' },
        () => ({ aggregate: 'thread', streamId: t.id, type: 'thread.handoff', payload: { sourceThreadId: source.id, fromProviderId: source.providerId ?? null, toProviderId: toProvider.id, atTurn: payload.atTurn } }),
      )
    }
    deps.sendToRenderer('thread:update', t)
    deps.remoteServer.publish({ kind: 'thread', data: t })
    await auditEvent('thread.handoff', { source: auditPreview(source.id), target: auditPreview(t.id), toProvider: auditPreview(toProvider.id), atTurn: payload.atTurn, autoStart: input?.autoStart !== false })
    // Continue seamlessly: with structured apiMessages seeded, the first turn is a
    // short "continue from here" prompt rather than the full flattened transcript
    // (which would duplicate the history). For the rare case where seeding produced
    // nothing (empty source), fall back to the flattened seedMessage as before.
    // autoStart:false leaves the thread idle for the user to kick off manually.
    if (input?.autoStart !== false) {
      const seedText = runPlan
        ? runPlanPrompt(payload.fromEngine, plan)
        : seededApiMessages.length
          ? `Continue from where the previous engine (${payload.fromEngine}) left off. The conversation history above was handed off from ${payload.fromEngine}; pick up the task and proceed.`
          : (typeof payload.seedMessage.content === 'string' ? payload.seedMessage.content : '')
      if (seedText) await deps.startThreadRun({ threadId: t.id, userText: seedText })
    }
    return t
  })
}
