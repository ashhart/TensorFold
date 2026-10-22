import { selectedResponseLocale } from '../../agent/response-language'
import { estimateTokens } from '../../agent/compaction'
import { personalMessageDraftEvent } from '../../agent/tool-labels'
import { emailDraftEvent } from '../../agent/email-connector'
import { assertRoutineSources, personalRoutinePrompt, routineSourceRevision } from './personal-routine-context'
import { randomUUID } from 'node:crypto'
import { automationTick, completionEvaluationRequest, evaluateCompletion, gateAutomationRun, snapshotAutomationPermissions, EMPTY_LEASE, type CompletionPolicy, type CompletionVerdict, type SchedulableAutomation, type SchedulerLease } from '../../agent/automation-scheduler'
import { runAgent } from '../../agent/loop'
import { streamChat } from '../../agent/openai'
import { ensureTensorFoldStudioStylePrompt } from '../../agent/prompt-style'
import { automationRunError } from '../../agent/run-settlement'
import { resolveProviderModel } from '../../agent/provider-accounts'
import { buildHostDigest } from '../../agent/host-digest'
import type { JobPatch } from '../../agent/jobs'
import type { AgentEvent, Automation, AutomationRun, ConnectorEvent, Job, Provider, Thread, TimelineItem } from '../../agent/types'
import { assertProviderEndpoint, auditPreview } from '../../agent/security'
import { store } from '../state'
import { engineService } from './engine-runtime'
import { isEngineProvider } from '../../agent/engine/provider'
import { auditEvent, requireSecurityRole } from './audit'
import { handle } from './ipc'
import { persistAutomations } from './persistence'
import { resolveWorkspaceCwd } from './workspace'
import { sanitizeAutomationCreateInput, sanitizeAutomationUpdatePatch, validateAutomationProviderRoute } from './automation-patch-policy'
import { nextDailyAutomationRun, normalizeAutomationTimeZone } from '../../../shared/automation-time'
import {
  modelRouteAccountLabel,
  modelRouteAutoFailoverEnabled,
  modelRouteResetPolicy,
  modelRouter,
  rememberModelRouteHistory,
  routeHandoffPreview,
  runtimeProviderWithFreshModels,
  selectedProviderAccountId,
} from './provider-runtime'

export interface AutomationRunSource {
  trigger?: 'manual' | 'schedule' | 'webhook' | 'connector'
  connectorEventId?: string | null
  connectorId?: string | null
  connectorAction?: string | null
  threadKey?: string | null
}

interface AutomationRuntimeDeps {
  threadJobHeartbeatMs: number
  claimJob(ref: Parameters<typeof import('../../agent/jobs').enqueueJob>[1]): Job
  patchJob(job: Job | null, patch: JobPatch): Job | null
  heartbeatJob(job: Job | null): Job | null
  finishJob(job: Job | null, outcome: 'completed' | 'failed' | 'cancelled' | 'paused', reason?: string): void
  emitAutomation(automation: Automation): void
  effectiveContextLimit(provider: Provider, model: string | null, accountId?: string | null): number
  fullToolContext(run?: { thread?: Thread; provider?: Provider; model?: string; signal?: AbortSignal }): unknown
  applyEventToThread(thread: Thread, event: AgentEvent): void
  timelineErrorSince(thread: Thread, timelineStart: number): TimelineItem | null
  timelineError(errorItem: TimelineItem): Error
  isHighRiskTool(name: string): boolean
  rememberConnectorEvents(events: ConnectorEvent[]): void | Promise<void>
  // Scheduler lease persistence (VST-038): the single-runner lease is saved so a restart
  // (or a second host reading the same shared state) sees who currently owns scheduling and
  // stands down instead of double-firing. Absent → in-memory only (the pre-persistence path).
  loadSchedulerLease?(): SchedulerLease | null
  saveSchedulerLease?(lease: SchedulerLease): void
}

export interface AutomationRuntime {
  mkAutomation(p: Partial<Automation>): Automation
  computeNextRun(a: Automation, from?: number): number | null
  runAutomation(a: Automation, source?: AutomationRunSource): Promise<void>
  startScheduler(): void
  abort(id: string): void
  abortAll(): void
}

export function createAutomationRuntime(deps: AutomationRuntimeDeps): AutomationRuntime {
  const runningAutomations = new Set<string>()
  const automationAborters = new Map<string, AbortController>()
  let schedulerTimer: ReturnType<typeof setInterval> | null = null
  // Single-runner scheduler lease (VST-038). Per-process for now (in-memory); the owner id is
  // this process. A future cross-host lease persists to the orchestrator so a laptop + mini
  // don't double-fire — the tick already stands down when it doesn't hold the lease.
  const schedulerOwnerId = `sched-${randomUUID().slice(0, 8)}`
  let schedulerLease: SchedulerLease = EMPTY_LEASE

  // Map a stored Automation onto the kernel's minimal schedulable shape.
  function schedulable(a: Automation): SchedulableAutomation {
    return {
      id: a.id,
      enabled: a.enabled,
      manual: a.scheduleKind === 'manual',
      // Daily automations use nextRunAt (below); the kernel's interval math covers 'interval'.
      everyMs: a.scheduleKind === 'interval' ? Math.max(1, a.everyMinutes) * 60_000 : 0,
      lastRunAt: a.lastRunAt,
      nextRunAt: a.nextRunAt,
      misfire: a.misfire ?? 'coalesce',
      requiredRisks: a.requiredRisks ?? [],
      acknowledgedRisks: a.acknowledgedRisks ?? [],
    }
  }

  function mkAutomation(p: Partial<Automation>): Automation {
    return {
      id: randomUUID(),
      name: p.name || 'Untitled automation',
      prompt: p.prompt || '',
      scheduleKind: p.scheduleKind || 'manual',
      everyMinutes: p.everyMinutes ?? 60,
      atHour: p.atHour ?? 9,
      atMinute: p.atMinute ?? 0,
      ...(p.timeZone !== undefined ? { timeZone: normalizeAutomationTimeZone(p.timeZone) } : {}),
      providerId: p.providerId ?? store.settings?.activeProviderId ?? null,
      providerAccountId: p.providerAccountId ?? store.settings?.activeProviderAccountId ?? null,
      model: p.model ?? store.settings?.activeModel ?? null,
      mode: p.mode || 'chat',
      cwd: p.cwd ?? store.settings?.defaultCwd ?? null,
      deliverTo: p.deliverTo || 'inbox',
      ...(p.deliveryTarget ? { deliveryTarget: { ...p.deliveryTarget } } : {}),
      ...(p.previewOnly !== undefined ? { previewOnly: p.previewOnly } : {}),
      ...(p.botId ? { botId: p.botId } : {}),
      ...(p.sourceScope ? { sourceScope: p.sourceScope.map(scope => ({ ...scope })) } : {}),
      ...(p.routineKind ? { routineKind: p.routineKind } : {}),
      enabled: p.enabled ?? false,
      lastRunAt: null,
      nextRunAt: null,
      lastStatus: null,
      runs: [],
      // Industrial fields (VST-038) carried through on create so a template / NL-built draft
      // keeps its misfire policy, risk gating, and completion policy.
      ...(p.misfire ? { misfire: p.misfire } : {}),
      ...(p.requiredRisks ? { requiredRisks: p.requiredRisks } : {}),
      ...(p.acknowledgedRisks ? { acknowledgedRisks: p.acknowledgedRisks } : {}),
      ...(p.completionPolicy ? { completionPolicy: p.completionPolicy } : {}),
      createdAt: Date.now(),
      updatedAt: Date.now(),
    }
  }

  function computeNextRun(a: Automation, from = Date.now()): number | null {
    if (!a.enabled || a.scheduleKind === 'manual') return null
    if (a.scheduleKind === 'interval') {
      const step = Math.max(1, a.everyMinutes) * 60_000
      const base = a.lastRunAt ?? from
      let next = base + step
      if (next <= from) next = from + step
      return next
    }
    return nextDailyAutomationRun(a.atHour, a.atMinute, Math.max(from, a.lastRunAt ?? from), a.timeZone)
  }

  function touchAutomation(a: Automation) {
    try { a.nextRunAt = computeNextRun(a); a.scheduleError = null }
    catch (error) { a.nextRunAt = null; a.scheduleError = error instanceof Error ? error.message : 'Fix this automation schedule before it can run.' }
    a.updatedAt = Date.now()
    persistAutomations()
    deps.emitAutomation(a)
  }

  // The automation's completion policy as the kernel expects it (default: one pass, done).
  function completionPolicyOf(a: Automation): CompletionPolicy {
    const p = a.completionPolicy
    if (!p) return { type: 'none', version: 0 }
    return p
  }

  // Ask a fresh READ-ONLY model call to judge whether the run met its objective, using the
  // run's own output as the transcript excerpt (VST-038). Only 'ai_evaluated' policies call
  // a model; the rest decide deterministically. A model/parse failure degrades to
  // needs_review — never a false "complete".
  async function evaluateRunCompletion(a: Automation, run: AutomationRun, provider: Provider | null, model: string | null, signal: AbortSignal, assertRequestAllowed?: () => void): Promise<{ verdict: CompletionVerdict; reason: string } | undefined> {
    const policy = completionPolicyOf(a)
    if (policy.type === 'none') return undefined
    let aiVerdict: CompletionVerdict | undefined
    const request = completionEvaluationRequest(policy, (run.output || '').slice(-4000))
    if (request && provider && model) {
      try {
        assertRequestAllowed?.()
        let out = ''
        for await (const ch of streamChat({
          provider, baseUrl: provider.baseUrl, apiKey: provider.apiKey, model,
          messages: [
            { role: 'system', content: ensureTensorFoldStudioStylePrompt('You judge whether an automation met its objective. Reply with exactly one word.') },
            { role: 'user', content: request.prompt },
          ],
          temperature: 0, approvalMode: 'readonly', tools: [], signal, assertRequestAllowed,
        })) {
          if (ch.type === 'content') out += ch.text
        }
        const word = out.toLowerCase().match(/\b(complete|continue|needs_review|needs review)\b/)?.[1]
        aiVerdict = word === 'complete' ? 'complete' : word === 'continue' ? 'continue' : word ? 'needs_review' : undefined
      } catch {
        aiVerdict = undefined // fall through to the kernel, which yields needs_review below
      }
    }
    const result = evaluateCompletion(policy, {
      iterations: 1,
      signals: [],
      aiVerdict: policy.type === 'ai_evaluated' ? (aiVerdict ?? 'needs_review') : undefined,
      runPolicyVersion: run.permissions?.completionPolicyVersion,
    })
    return { verdict: result.verdict, reason: result.reason }
  }

  async function runAutomation(a: Automation, source: AutomationRunSource = {}): Promise<void> {
    if (runningAutomations.has(a.id)) return
    const definition = structuredClone(a)
    const trigger = source.trigger ?? (definition.scheduleKind === 'manual' ? 'manual' : 'schedule')
    let job: Job | null = deps.claimJob({
      kind: 'automation',
      owner: a.id,
      automationId: a.id,
      connectorEventId: source.connectorEventId ?? null,
      projectId: null,
      metadata: {
        name: definition.name,
        scheduleKind: definition.scheduleKind,
        trigger,
        connectorId: source.connectorId ?? null,
        connectorAction: source.connectorAction ?? null,
        threadKey: source.threadKey ?? null,
      },
    })
    const baseProvider = store.settings.providers.find((p) => p.id === (definition.providerId ?? store.settings.activeProviderId))
    const requestedProviderAccountId = baseProvider ? selectedProviderAccountId(baseProvider, definition.providerAccountId) : null
    let providerAccountId = requestedProviderAccountId
    let provider: Provider | null = null
    let model: string | null = null
    const run: AutomationRun = { id: randomUUID(), startedAt: Date.now(), finishedAt: null, status: 'running', output: '' }
    // Freeze the safety posture at run start (VST-038): a later edit to the definition
    // can't loosen what this in-flight run was permitted to do. Agent mode runs read-only
    // with no sandbox override here; the acknowledged risks + completion-policy version are
    // the audit-relevant part.
    run.permissions = snapshotAutomationPermissions({
      approvalMode: definition.mode === 'agent' ? 'readonly' : 'readonly',
      acknowledgedRisks: definition.acknowledgedRisks ?? [],
      completionPolicyVersion: definition.completionPolicy?.version ?? 0,
    }, run.startedAt)
    job = deps.patchJob(job, {
      currentRunId: `automation:${a.id}:${run.id}`,
      lastUsefulProgressAt: run.startedAt,
      metadata: {
        runId: run.id,
        trigger,
        connectorEventId: source.connectorEventId ?? null,
        connectorId: source.connectorId ?? null,
        connectorAction: source.connectorAction ?? null,
        threadKey: source.threadKey ?? null,
      },
    })
    a.runs.unshift(run)
    if (a.runs.length > 20) a.runs.length = 20
    a.lastRunAt = run.startedAt
    runningAutomations.add(a.id)
    const controller = new AbortController()
    automationAborters.set(a.id, controller)
    persistAutomations()
    deps.emitAutomation(a)

    try {
      assertRoutineSources(definition, store.settings)
      const context = baseProvider && definition.model ? deps.effectiveContextLimit(baseProvider, definition.model, requestedProviderAccountId) : 0
      const reserve = Math.min(2_048, Math.floor(context / 3))
      const maxCharacters = Math.max(0, (context - reserve - 512) * 4)
      const instruction = personalRoutinePrompt(definition, store.settings, store.connectorEvents, definition.routineKind ? { maxCharacters } : undefined)
      const sourceRevision = routineSourceRevision(definition, store.settings)
      const assertRoutineRequest = (activeProvider?: Provider) => {
        if (!definition.routineKind) return
        assertRoutineSources(definition, store.settings)
        if (routineSourceRevision(definition, store.settings) !== sourceRevision) throw new Error('A selected routine source or contact scope changed; review the routine before running it again.')
        const engine = engineService()
        if (engine?.hasExternalResearchEnabled() || isEngineProvider(activeProvider) && !engine) throw new Error('Personal routines require hidden Cortheon research to be off so selected source and tool boundaries can be enforced.')
        if (activeProvider && (activeProvider.id !== definition.providerId)) throw new Error('Personal routines keep their reviewed provider and model.')
      }
      let lastJobBeat = Date.now()
      const beatAutomationJob = () => {
        if (Date.now() - lastJobBeat < deps.threadJobHeartbeatMs) return
        lastJobBeat = Date.now()
        job = deps.heartbeatJob(job) ?? job
      }
      if (definition.mode === 'digest') {
        // Digest automations run NO model: compose a deterministic briefing from recent host
        // activity and hand it straight to the same deliver path below.
        run.output = buildHostDigest({ now: Date.now(), jobs: store.jobs, missions: store.missions, rooms: store.rooms, connectorEvents: store.connectorEvents })
      } else {
      if (!baseProvider) throw new Error('No provider configured for this automation.')
      const requestedProvider = await runtimeProviderWithFreshModels(baseProvider, requestedProviderAccountId)
      model = resolveProviderModel(requestedProvider, null, definition.model, store.settings.activeModel)
      if (!model) throw new Error('No provider/model configured for this automation.')
      const routeBaseProvider = store.settings.providers.find((p) => p.id === baseProvider.id) ?? baseProvider
      const initialRoute = modelRouter.select({
        role: 'automation',
        provider: routeBaseProvider,
        model,
        requestedAccountId: requestedProviderAccountId,
        allowFailover: !definition.routineKind && modelRouteAutoFailoverEnabled(routeBaseProvider, requestedProviderAccountId),
        contextLimit: deps.effectiveContextLimit(routeBaseProvider, model, requestedProviderAccountId),
        reasoningEffort: store.settings.reasoningEffort,
        budget: definition.scheduleKind,
        source: 'automation_initial_route',
      })
      if (!initialRoute.selected) throw new Error(initialRoute.audit.reason)
      providerAccountId = initialRoute.selectedAccountId
      provider = providerAccountId === requestedProviderAccountId
        ? requestedProvider
        : await runtimeProviderWithFreshModels(routeBaseProvider, providerAccountId)
      assertProviderEndpoint(provider)
      if (initialRoute.audit.action === 'failover') {
        const resetAt = initialRoute.candidates.find((candidate) => candidate.accountId === initialRoute.requestedAccountId)?.exhaustedUntil ?? null
        job = deps.patchJob(job, {
          metadata: {
            providerId: baseProvider.id,
            providerAccountId,
            model,
            modelRoute: {
              action: 'automation_initial_failover',
              preferredProviderAccountId: initialRoute.requestedAccountId,
              failoverProviderAccountId: initialRoute.selectedAccountId,
              resetPolicy: modelRouteResetPolicy(routeBaseProvider, initialRoute.requestedAccountId),
              failedOverAt: Date.now(),
              resetAt,
              reason: initialRoute.audit.reason,
            },
          },
        })
        rememberModelRouteHistory('automation', initialRoute.audit, null, { id: a.id, label: definition.name })
        await auditEvent('model_router.automation_failover', {
          automationId: a.id,
          providerId: routeBaseProvider.id,
          model,
          requestedAccountId: initialRoute.requestedAccountId,
          selectedAccountId: initialRoute.selectedAccountId,
          resetAt,
          reason: auditPreview(initialRoute.audit.reason),
        })
      }
      const runAutomationWithRoute = async (route: { provider: Provider; providerAccountId: string | null; model: string }, handoff?: string): Promise<string> => {
        assertRoutineRequest(route.provider)
        if (definition.mode !== 'agent') {
          let out = ''
          const messages = [
            { role: 'system' as const, content: ensureTensorFoldStudioStylePrompt('You are an automation worker. Carry out the instruction and return only the finished result, concise and ready to deliver.') },
            { role: 'user' as const, content: handoff ? `${handoff}\n\nInstruction:\n${instruction}` : instruction },
          ]
          const assertScopedRequest = () => {
            assertRoutineRequest(route.provider)
            if (definition.routineKind) {
              const currentContext = deps.effectiveContextLimit(route.provider, route.model)
              const outputReserve = Math.min(2_048, Math.floor(currentContext / 3))
              if (estimateTokens(messages) + outputReserve > currentContext) throw new Error('This routine exceeds the selected model context; choose fewer sources or a larger context profile.')
            }
          }
          assertScopedRequest()
          for await (const ch of streamChat({
            provider: route.provider, baseUrl: route.provider.baseUrl, apiKey: route.provider.apiKey, model: route.model,
            messages,
            temperature: store.settings.temperature, reasoningEffort: store.settings.reasoningEffort, approvalMode: 'readonly', tools: [], cwd: definition.routineKind ? undefined : definition.cwd || store.settings.defaultCwd || undefined, signal: controller.signal, assertRequestAllowed: definition.routineKind ? assertScopedRequest : undefined,
          })) {
            if (ch.type === 'content') out += ch.text
            beatAutomationJob()
          }
          return out.trim() || '(empty response)'
        }
        const cwd = await resolveWorkspaceCwd(definition.cwd || store.settings.defaultCwd, 'automation workspace')
        const thread: Thread = {
    documentSourceIds: [],
          id: randomUUID(), title: definition.name, kind: 'task', projectId: null,
          cwd,
          providerId: baseProvider.id, providerAccountId: route.providerAccountId, model: route.model, createdAt: Date.now(), updatedAt: Date.now(),
          status: 'running', apiMessages: [{ role: 'user', content: handoff ? `${handoff}\n\nInstruction:\n${instruction}` : instruction }], timeline: [],
        }
        const timelineStart = thread.timeline.length
        await runAgent({
        responseLocale: selectedResponseLocale(store.settings.uiPreferences),
          thread, provider: route.provider, model: route.model, approvalMode: 'readonly', temperature: store.settings.temperature,
          reasoningEffort: store.settings.reasoningEffort, autoCompact: store.settings.autoCompact, contextLimit: deps.effectiveContextLimit(route.provider, route.model),
          toolContext: deps.fullToolContext({ thread, provider: route.provider, model: route.model, signal: controller.signal }) as any,
          maxSteps: 12,
          maxWallMs: 10 * 60_000,
          emit: (e) => { deps.applyEventToThread(thread, e); beatAutomationJob() }, signal: controller.signal,
          requestApproval: async () => 'deny',
          auditToolCall: async ({ tool, args, resultOk, output }) => {
            if (deps.isHighRiskTool(tool)) await auditEvent('automation.tool', { automationId: a.id, tool, cwd: auditPreview(cwd), ok: resultOk, args: auditPreview(JSON.stringify(args)), output: auditPreview(output) })
          },
          persist: () => {},
        })
        const errorItem = deps.timelineErrorSince(thread, timelineStart)
        if (errorItem && !controller.signal.aborted) throw deps.timelineError(errorItem)
        const last = [...thread.timeline].reverse().find((i) => i.kind === 'assistant' && i.text)
        return last?.text || '(agent run completed with no final message)'
      }
      let activeRoute = { provider, providerAccountId, model }
      try {
        run.output = await runAutomationWithRoute(activeRoute)
      } catch (err) {
        const latestBaseProvider = store.settings.providers.find((p) => p.id === baseProvider.id) ?? baseProvider
        const routed = modelRouter.routeAfterError({
          role: 'automation',
          provider: latestBaseProvider,
          accountId: activeRoute.providerAccountId,
          model: activeRoute.model,
          error: err,
          allowFailover: !definition.routineKind && modelRouteAutoFailoverEnabled(latestBaseProvider, activeRoute.providerAccountId),
          contextLimit: deps.effectiveContextLimit(latestBaseProvider, activeRoute.model, activeRoute.providerAccountId),
          reasoningEffort: store.settings.reasoningEffort,
          budget: definition.scheduleKind,
          source: 'automation_recovery',
        })
        if (!routed.signal || !routed.route.selected || routed.route.selectedAccountId === activeRoute.providerAccountId || controller.signal.aborted) throw err
        providerAccountId = routed.route.selectedAccountId
        provider = await runtimeProviderWithFreshModels(latestBaseProvider, providerAccountId)
        assertProviderEndpoint(provider)
        activeRoute = { provider, providerAccountId, model: activeRoute.model }
        job = deps.patchJob(job, {
          metadata: {
            providerId: latestBaseProvider.id,
            providerAccountId,
            model: activeRoute.model,
            modelRoute: {
              action: 'automation_failover',
              preferredProviderAccountId: routed.route.requestedAccountId,
              failoverProviderAccountId: providerAccountId,
              resetPolicy: modelRouteResetPolicy(latestBaseProvider, routed.route.requestedAccountId),
              failedOverAt: Date.now(),
              resetAt: routed.signal.resetAt,
              reason: routed.route.audit.reason,
            },
          },
        })
        rememberModelRouteHistory('automation', routed.route.audit, routed.signal, { id: a.id, label: definition.name })
        await auditEvent('model_router.automation_failover', {
          automationId: a.id,
          providerId: latestBaseProvider.id,
          model: activeRoute.model,
          requestedAccountId: routed.route.requestedAccountId,
          selectedAccountId: providerAccountId,
          resetAt: routed.signal.resetAt,
          source: routed.signal.source,
          exhaustionKind: routed.signal.kind,
          reason: auditPreview(routed.route.audit.reason),
        })
        run.output = await runAutomationWithRoute(activeRoute, [
          '<model_router_handoff>',
          `Reason: Automation "${routeHandoffPreview(definition.name, 160)}" failed over after ${routed.signal.kind.replace(/_/g, ' ')}.`,
          `Schedule: ${routeHandoffPreview(definition.scheduleKind, 80)}`,
          `Trigger: ${routeHandoffPreview(trigger, 120)}`,
          '</model_router_handoff>',
        ].join('\n'))
        run.output = `Model router failover: ${modelRouteAccountLabel(latestBaseProvider, routed.route.requestedAccountId)} -> ${modelRouteAccountLabel(latestBaseProvider, providerAccountId)}.\n\n${run.output}`
      }
      } // end non-digest model run
      if (controller.signal.aborted) throw new Error('Automation cancelled before completion.')
      // Completion policy verdict (VST-038): for an ai_evaluated policy, feed the finished
      // run's output as the transcript excerpt to a fresh, read-only model call and record
      // its one-word verdict; deterministic policies evaluate without a model. The version
      // guard forces needs_review if the policy was edited after this run started.
      run.completion = await evaluateRunCompletion(definition, run, provider, model, controller.signal, definition.routineKind ? () => assertRoutineRequest(provider ?? undefined) : undefined)
      if ((a.completionPolicy?.version ?? 0) !== (definition.completionPolicy?.version ?? 0)) run.completion = { verdict: 'needs_review', reason: 'Completion policy changed during this run; review the result.' }
      assertRoutineRequest(provider ?? undefined)
      if (controller.signal.aborted) throw new Error('Automation cancelled before completion.')
      run.status = 'ok'
      a.lastStatus = 'ok'
      deps.finishJob(job, 'completed')
      run.deliveryStatus = 'inbox'
      if (definition.deliverTo !== 'inbox' && !definition.previewOnly) {
        const target = definition.deliveryTarget
        if (!target || target.connectorId !== definition.deliverTo || !target.recipient.trim() || (target.connectorId !== 'telegram' && !target.accountId)) {
          // Existing addressless definitions remain safe/local until a user chooses a
          // destination. Successful generation is not a claim of message delivery.
          run.deliveryStatus = 'blocked'
          run.deliveryError = 'Choose an exact account and recipient before reviewing external delivery; this result remains in the inbox.'
        } else {
          const event = target.connectorId === 'email'
            ? emailDraftEvent({ to: target.recipient, subject: target.subject || `${definition.name} result`, body: run.output, mode: 'send' })
            : personalMessageDraftEvent({ connectorId: target.connectorId, target: target.recipient, body: run.output, accountId: target.accountId ?? undefined })
          event.approvalId = `automation-${a.id}-${run.id}`
          event.payload = { ...event.payload, accountId: target.accountId, automationId: a.id, runId: run.id }
          event.payloadKeys = [...event.payloadKeys, 'accountId', 'automationId', 'runId']
          await deps.rememberConnectorEvents([event])
          run.deliveryEventId = event.id
          run.deliveryStatus = 'pending_approval'
        }
      }

    } catch (e: any) {
      const settlement = automationRunError(e, controller.signal.aborted)
      run.status = 'error'
      run.error = settlement.error
      a.lastStatus = 'error'
      deps.finishJob(job, settlement.jobOutcome, run.error)
    } finally {
      run.finishedAt = Date.now()
      runningAutomations.delete(a.id)
      automationAborters.delete(a.id)
      touchAutomation(a)
    }
  }

  function startScheduler() {
    if (schedulerTimer) return
    for (const a of store.automations) {
      if (!a.enabled || a.scheduleKind === 'manual') continue
      try {
        // Validate even an existing committed instant, but preserve overdue instants so
        // restart still coalesces a missed run instead of silently moving it to tomorrow.
        const next = computeNextRun(a)
        if (a.nextRunAt == null) a.nextRunAt = next
        a.scheduleError = null
      } catch (error) {
        a.nextRunAt = null
        a.scheduleError = error instanceof Error ? error.message : 'Fix this automation schedule before it can run.'
      }
    }
    persistAutomations()
    // Restore the persisted lease (VST-038): if another owner still holds it, this process's
    // ticks will stand down until it lapses — the persisted lease survives our restart.
    const restored = deps.loadSchedulerLease?.()
    if (restored) schedulerLease = restored
    schedulerTimer = setInterval(() => {
      const now = Date.now()
      // Interval automations flow through the industrial kernel: single-runner lease +
      // misfire coalescing + risk gate (VST-038). Lease TTL > the 30s tick so it doesn't
      // lapse between ticks.
      const tick = automationTick({
        automations: store.automations.filter((a) => a.scheduleKind === 'interval').map(schedulable),
        lease: schedulerLease,
        ownerId: schedulerOwnerId,
        now,
        leaseTtlMs: 90_000,
        isRunning: (id) => runningAutomations.has(id),
      })
      // Persist on every successful acquire/renew (a fresh heartbeat+expiry) so a
      // cross-host reader sees this owner is alive; skip on a contention stand-down, where
      // acquireLease returns the same object it was given.
      if (tick.lease !== schedulerLease) deps.saveSchedulerLease?.(tick.lease)
      schedulerLease = tick.lease
      // Another scheduler holds the lease → stand down entirely (covers daily too).
      if (!tick.acquired) return
      // run_each dispatches an id once per missed fire on purpose; don't Set-collapse the
      // batch. Run each id's batch SEQUENTIALLY instead: a concurrent batch would be
      // collapsed by runAutomation's isRunning guard, and serialization keeps that guard
      // (plus lastRunAt) as the double-fire protection. coalesce dispatches once per id,
      // so its behavior is unchanged.
      const dispatchCounts = new Map<string, number>()
      for (const id of tick.dispatch) dispatchCounts.set(id, (dispatchCounts.get(id) ?? 0) + 1)
      for (const [id, count] of dispatchCounts) {
        const a = store.automations.find((x) => x.id === id)
        if (!a) continue
        void (async () => { for (let i = 0; i < count; i++) await runAutomation(a) })()
      }
      // Daily automations fire by nextRunAt, under the SAME lease + risk gate.
      for (const a of store.automations) {
        if (a.scheduleKind !== 'daily' || !a.enabled || a.scheduleError) continue
        if (runningAutomations.has(a.id)) continue
        if (a.nextRunAt != null && a.nextRunAt <= now && gateAutomationRun({ requiredRisks: a.requiredRisks ?? [], acknowledgedRisks: a.acknowledgedRisks ?? [] }).allowed) {
          void runAutomation(a)
        }
      }
    }, 30_000)
  }

  return {
    mkAutomation,
    computeNextRun,
    runAutomation,
    startScheduler,
    abort: (id) => automationAborters.get(id)?.abort(),
    abortAll: () => {
      for (const controller of automationAborters.values()) controller.abort()
      automationAborters.clear()
    },
  }
}

export function registerAutomationIpc(runtime: AutomationRuntime) {
  handle('automations:get', async () => store.automations)
  handle('automations:create', async (_e, p: Partial<Automation>) => {
    requireSecurityRole('operator', 'create automations')
    const a = runtime.mkAutomation(sanitizeAutomationCreateInput(p))
    validateAutomationProviderRoute(a, store.settings.providers)
    assertRoutineSources(a, store.settings)
    a.nextRunAt = runtime.computeNextRun(a)
    store.automations.unshift(a)
    persistAutomations()
    return a
  })
  handle('automations:update', async (_e, { id, patch }: { id: string; patch: Partial<Automation> }) => {
    requireSecurityRole('operator', 'update automations')
    const a = store.automations.find((x) => x.id === id)
    if (!a) throw new Error('Automation not found')
    const safePatch = sanitizeAutomationUpdatePatch(a, patch)
    validateAutomationProviderRoute({ ...a, ...safePatch }, store.settings.providers)
    assertRoutineSources({ ...a, ...safePatch }, store.settings)
    const nextRunAt = runtime.computeNextRun({ ...a, ...safePatch })
    Object.assign(a, safePatch, { updatedAt: Date.now(), nextRunAt, scheduleError: null })
    persistAutomations()
    return a
  })
  handle('automations:delete', async (_e, id: string) => {
    requireSecurityRole('operator', 'delete automations')
    runtime.abort(id)
    store.automations = store.automations.filter((a) => a.id !== id)
    persistAutomations()
  })
  handle('automation:run', async (_e, id: string) => {
    requireSecurityRole('operator', 'run automations')
    const a = store.automations.find((x) => x.id === id)
    if (a) void runtime.runAutomation(a)
    return { ok: !!a }
  })
}
