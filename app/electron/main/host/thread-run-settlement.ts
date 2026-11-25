import { canContributeThreadMemory } from '../../../shared/thread-memory-policy'
import { randomUUID } from 'node:crypto'
import { compactThreadJournal } from '../../agent/thread-journal'
import { assertProviderEndpointResolved } from '../../agent/security'
import { buildAgentRunReceipt, buildEventReceipt } from '../../agent/run-contract'
import type { Project, Provider, Thread } from '../../agent/types'
import { isEngineProvider } from '../../agent/engine/provider'
import { store } from '../state'
import { auditModelRetry, auditModelRoute, modelRouteAccountLabel, modelRouteAutoFailoverEnabled, modelRouteFailoverText, modelRouteResetPolicy, modelRouter, rememberProviderModels, runtimeProviderWithFreshModels } from './provider-runtime'

import type { ThreadRoute, ThreadRouteContext } from './thread-run-execution'
import { botRunReasoningEffort } from '../../../shared/bot-reasoning'

interface ThreadSettlementContext extends ThreadRouteContext {
  initialRoute: ThreadRoute
  baseProvider: Provider
  project: Project | null
  threadRouteRole: 'chat' | 'task'
  runTimelineStart: number
  runLease: { release(): void }
  releaseBotVmRun(): void
  runWithRoute(route: ThreadRoute, model?: string): Promise<void>
}

export function settleThreadExecution(pending: Promise<void>, input: ThreadSettlementContext): void {
  const { deps, support, thread, controller, runId, runTimelineStart, workspace, runContract, memoryPolicy, memoryRevision, localHelpersOnly, payload, jobHandle, review, runLease, releaseBotVmRun, baseProvider, project, threadRouteRole, runWithRoute, emit } = input
  let activeRoute = input.initialRoute
  let { provider, providerAccountId } = activeRoute
  const { aborters, sshManager, emitToRenderer, persistThreads, patchJob, settleThreadRunJob, effectiveContextLimit } = deps
  const { addThreadNotice, timelineErrorSince, markRouteErrorRecovered, markUndrainedCues, injectThreadRouteHandoff, availableModelsFromProviderError, compactThreadJournalTail, reviewViewedSkills, proposePostRunMemory } = support
  void pending
    .catch(async (err) => {
      const liveModels = availableModelsFromProviderError(err)
      const fallbackModel = liveModels.find((candidate) => candidate !== activeRoute.model)
      if (fallbackModel && thread.runRecovery?.runId === runId && thread.runRecovery.retry === 'safe' && !thread.bot && !controller.signal.aborted) {
        const rejectedModel = activeRoute.model
        markRouteErrorRecovered(thread, err, runId, `Model "${rejectedModel}" was rejected by ${baseProvider.name}; retrying with "${fallbackModel}".`)
        rememberProviderModels(baseProvider.id, activeRoute.providerAccountId, liveModels)
        activeRoute = { ...activeRoute, model: fallbackModel }
        thread.model = fallbackModel
        thread.updatedAt = Date.now()
        persistThreads()
        emitToRenderer({ runId, threadId: thread.id, type: 'thread_update', patch: { model: fallbackModel } })
        const notice = { id: randomUUID(), kind: 'notice' as const, text: `Model "${rejectedModel}" was rejected by ${baseProvider.name}. Retrying with "${fallbackModel}".`, createdAt: Date.now() }
        thread.timeline.push(notice)
        emitToRenderer({ runId, threadId: thread.id, type: 'item_add', item: notice })
        await auditModelRetry(thread, {
          providerId: baseProvider.id,
          providerAccountId: activeRoute.providerAccountId,
          fromModel: rejectedModel,
          toModel: fallbackModel,
          source: 'provider_available_models',
          reason: `Provider rejected model "${rejectedModel}" and returned live alternatives.`,
          availableModels: liveModels,
        })
        try {
          await runWithRoute(activeRoute, fallbackModel)
        } catch (retryErr) {
          emitToRenderer({ runId, threadId: thread.id, type: 'error', message: String((retryErr as any)?.message ?? retryErr) })
        }
        return
      }
      // The route-recovery setup below performs un-try-caught awaits (audit, provider refresh,
      // and assertProviderEndpointResolved — which throws when a failover endpoint resolves to a
      // private IP). Without this guard a throw here rejects the catch callback itself; the chain
      // has no further .catch, so it becomes an unhandled rejection and the user sees the run
      // silently die. Wrap it so any failure surfaces as the same user-facing error event the
      // non-recovered path emits, then fall through to the shared .finally.
      try {
        const latestBaseProvider = store.settings.providers.find((p) => p.id === baseProvider.id) ?? baseProvider
        const routed = modelRouter.routeAfterError({
          role: threadRouteRole,
          provider: latestBaseProvider,
          accountId: activeRoute.providerAccountId,
          model: activeRoute.model,
          error: err,
          allowFailover: thread.runRecovery?.runId === runId && thread.runRecovery.retry === 'safe' && !thread.bot && (modelRouteAutoFailoverEnabled(latestBaseProvider, activeRoute.providerAccountId) || thread.modelRoute?.resetPolicy === 'rotate_on_limit'),
          localOnly: project?.localOnly === true,
          contextLimit: effectiveContextLimit(latestBaseProvider, activeRoute.model, activeRoute.providerAccountId),
          reasoningEffort: botRunReasoningEffort(thread, store.settings.reasoningEffort),
          budget: thread.kind,
          source: 'thread_recovery',
        })
        if (routed.signal) {
          for (const audit of routed.audit) await auditModelRoute(thread, audit, routed.signal)
        }
        if (thread.runRecovery?.runId === runId && thread.runRecovery.retry === 'safe' && routed.signal && routed.route.selected && routed.route.selectedAccountId !== activeRoute.providerAccountId && !controller.signal.aborted) {
          markRouteErrorRecovered(thread, err, runId, `Recovered provider limit on ${modelRouteAccountLabel(latestBaseProvider, activeRoute.providerAccountId)} for ${activeRoute.model}; retrying on ${modelRouteAccountLabel(latestBaseProvider, routed.route.selectedAccountId)}.`)
          const exhaustedAccountId = activeRoute.providerAccountId
          providerAccountId = routed.route.selectedAccountId
          provider = await runtimeProviderWithFreshModels(latestBaseProvider, providerAccountId)
          await assertProviderEndpointResolved(provider)
          activeRoute = { provider, providerAccountId, model: activeRoute.model }
          thread.providerId = latestBaseProvider.id
          thread.providerAccountId = providerAccountId
          thread.model = activeRoute.model
          thread.modelRoute = {
            preferredProviderAccountId: exhaustedAccountId,
            failoverProviderAccountId: providerAccountId,
            resetPolicy: modelRouteResetPolicy(latestBaseProvider, exhaustedAccountId),
            failedOverAt: Date.now(),
            resetAt: routed.signal.resetAt,
            reason: routed.route.audit.reason,
          }
          thread.updatedAt = Date.now()
          persistThreads()
          emitToRenderer({ runId, threadId: thread.id, type: 'thread_update', patch: { providerId: thread.providerId, providerAccountId, model: activeRoute.model, modelRoute: thread.modelRoute } })
          addThreadNotice(thread, runId, modelRouteFailoverText(latestBaseProvider, routed.route.audit, routed.signal))
          injectThreadRouteHandoff(thread, `Thread "${thread.title}" failed over after ${routed.signal.kind.replace(/_/g, ' ')}.`)
          try {
            await runWithRoute(activeRoute)
          } catch (retryErr) {
            emitToRenderer({ runId, threadId: thread.id, type: 'error', message: String((retryErr as any)?.message ?? retryErr) })
          }
          return
        }
        emitToRenderer({ runId, threadId: thread.id, type: 'error', message: String(err?.message ?? err) })
      } catch (recoveryErr) {
        // Recovery setup itself failed — surface the original error plus the recovery failure.
        emitToRenderer({ runId, threadId: thread.id, type: 'error', message: String((recoveryErr as any)?.message ?? recoveryErr) })
      }
    })
    .finally(() => {
      let job = jobHandle.get()
      markUndrainedCues(thread, controller.signal.aborted ? 'canceled' : 'run ended')
      const failed = !!timelineErrorSince(thread, runTimelineStart)
      const remoteTransport = workspace?.kind === 'ssh' && workspace.hostId && sshManager.status ? sshManager.status(workspace.hostId) : null
      const transportUnverifiable = failed && !!remoteTransport && (remoteTransport.phase === 'disconnected' || remoteTransport.phase === 'error')
      if (transportUnverifiable && workspace?.kind === 'ssh') {
        const reason = remoteTransport?.lastError
          ? `Remote transport lost: ${remoteTransport.lastError}`
          : 'Remote transport lost; the process state is unverifiable.'
        job = patchJob(job, {
          executionVerification: 'unverifiable',
          executionVerificationReason: reason,
          executionVerifiedAt: Date.now(),
        })
      }
      jobHandle.set(job)
      const recoveryPaused = thread.runRecovery?.runId === runId && ['paused', 'needs_review'].includes(thread.runRecovery.state)
      const receipt = buildAgentRunReceipt({ thread, contract: runContract, timelineStart: runTimelineStart, aborted: controller.signal.aborted })
      const turnReceipt = buildEventReceipt({
        kind: 'turn_complete',
        runId,
        threadId: thread.id,
        state: controller.signal.aborted ? 'cancelled' : failed ? 'error' : receipt.needsReview || recoveryPaused ? 'blocked' : 'ok',
        summary: receipt.summary,
        evidenceRefs: [receipt.id],
      })
      const jobReceipt = buildEventReceipt({
        kind: 'job_complete',
        runId,
        threadId: thread.id,
        state: controller.signal.aborted ? 'cancelled' : failed ? 'error' : receipt.needsReview || recoveryPaused ? 'blocked' : 'ok',
        summary: `Thread job finished with ${receipt.verdict}.`,
        evidenceRefs: [receipt.id],
      })
      const savedReceipt = buildEventReceipt({
        kind: 'run_receipt_saved',
        runId,
        threadId: thread.id,
        state: controller.signal.aborted ? 'cancelled' : failed ? 'error' : receipt.needsReview || recoveryPaused ? 'blocked' : 'ok',
        summary: 'Run receipt saved at run end.',
        evidenceRefs: [receipt.id],
      })
      receipt.eventReceipts.push(turnReceipt, jobReceipt, savedReceipt)
      emit({ runId, threadId: thread.id, type: 'thread_update', patch: { activeRunContract: null, lastRunReceipt: receipt } })
      emit({
        runId,
        threadId: thread.id,
        type: 'item_add',
        item: {
          id: randomUUID(),
          kind: 'notice',
          title: 'Run receipt',
          text: `Run receipt: ${receipt.verdict}. ${receipt.summary}`,
          state: receipt.needsReview ? 'awaiting' : 'ok',
          runReceipt: receipt,
          eventReceipt: savedReceipt,
          createdAt: Date.now(),
        },
      })
      thread.status = failed ? 'error' : 'idle'
      thread.updatedAt = Date.now()
      // Only retire our own controller — never clobber a controller a later run installed.
      if (aborters.get(thread.id) === controller) aborters.delete(thread.id)
      runLease.release()
      releaseBotVmRun()
      settleThreadRunJob(
        job,
        thread,
        controller.signal.aborted || transportUnverifiable || recoveryPaused ? 'paused' : failed ? 'failed' : 'completed',
        transportUnverifiable
          ? 'Remote transport disconnected; execution is unverifiable and must be reconciled after reconnecting.'
          : recoveryPaused ? thread.runRecovery?.message : failed ? 'Thread agent run ended with an error.' : undefined,
      )
      persistThreads()
      // Bound the per-thread NDJSON journal: high-frequency item_delta chunks make it grow
      // without limit since compactThreadJournal was never called in production. At run-end the
      // events are already folded into the persisted thread snapshot, so keep only a recent tail
      // for crash replay. Fire-and-forget; failures must not affect run settlement.
      if (!thread.incognito) void compactThreadJournalTail(thread.id)

      // Skill self-improvement (opt-in): on a successful run that consulted a skill, let an
      // aux model review those skills and FILE improvement suggestions (consent-first; never
      // auto-applied). Fully fire-and-forget — it must never disturb the settled run.
      if (!localHelpersOnly && !thread.phoneLocalOnly && !thread.googleLocalOnly && canContributeThreadMemory(thread, memoryPolicy, memoryRevision, controller.signal) && store.settings.skillSelfImprovement && !failed && !recoveryPaused && !controller.signal.aborted && review.viewedSkillNames.size && review.provider) {
        void reviewViewedSkills({ provider: review.provider, model: review.model, thread, userText: payload.userText, viewed: [...review.viewedSkillNames], signal: controller.signal, memoryPolicy, memoryRevision })
      }
      if (!failed && !recoveryPaused && !controller.signal.aborted) {
        // Let the foreground run release the built-in engine before starting its background title.
        const titleRoute = input.initialRoute
        if (isEngineProvider(titleRoute.provider) && !localHelpersOnly && !aborters.has(thread.id) && (thread.title === 'New task' || thread.title === 'New chat')) {
          support.maybeAutoTitle(thread, titleRoute.provider, titleRoute.model, payload.userText)
        }
        try { proposePostRunMemory(thread, memoryPolicy, memoryRevision, controller.signal) } catch { /* best effort — never disturb the settled run */ }
      }
    })
    // Backstop: every awaited path inside .catch/.finally is itself guarded, but attach a
    // terminal handler so no recovery/journal failure can ever escape as an unhandled rejection.
    .catch(() => {
      if (aborters.get(thread.id) === controller) aborters.delete(thread.id)
      runLease.release()
      releaseBotVmRun()
    })

}
