import { saveThreads, saveThreadQuestion } from '../../store'
import { surfaceModelOverride } from '../../agent/provider-accounts'
import { createEmitHelpers } from '../host/emit-helpers'
import { vontraHomeDir } from '../../store'
import { store } from '../state'
import { requireSecurityRole } from '../host/audit'
import { WorkspaceMutationGuard } from '../host/workspace-mutation-guard'
import { type ThreadRuntime } from '../host/thread-runtime'
import { sendToRenderer } from '../host/renderer-events'
import { createEngineRuntime, engineService, engineRuntimes } from '../host/engine-runtime'
import { registerEngineRuntimeIpc } from '../host/ipc/engine-runtime-ipc'
import { registerEngineIpc } from '../host/ipc/engine-ipc'
import { registerModelBenchmarkIpc } from '../host/ipc/model-benchmark-ipc'
import { initializePhoneHost } from '../host/phone-host'
import { registerPhoneIpc } from '../host/ipc/phone-ipc'
import { initializeBotVmHost } from '../host/bot-vm-host'
import { registerBotVmIpc } from '../host/ipc/bot-vm-ipc'
import { assertBotLifecycleIdle } from '../host/bot-run-boundary'
import { captureBotConfig } from '../../../shared/bot-policy'
import { isRestrictedBot } from '../../../shared/bot-tool-policy'
import { initializeUserQuestionHost, getUserQuestionHost } from '../host/user-question-host'
import { createUserQuestionPersistence } from '../host/user-question-persistence'
import { registerUserQuestionIpc } from '../host/ipc/user-question-ipc'
import { isExternalProvider } from '../../../shared/provider-trust'
import { runtimeProvider } from '../host/provider-runtime'
import { HEADLESS_MODE } from '../early-electron-paths'

interface ApplicationDeviceDeps {
  aborters: Map<string, AbortController>
  workspaceMutationGuard: WorkspaceMutationGuard
  cancelThreadRun: ThreadRuntime['cancelThreadRun']
  emitToRenderer: ReturnType<typeof createEmitHelpers>['emitToRenderer']
}

export function initializeApplicationDevices(deps: ApplicationDeviceDeps) {
  const { aborters, workspaceMutationGuard, cancelThreadRun, emitToRenderer } = deps
  createEngineRuntime()
  const assertCanBenchmark = () => {
    requireSecurityRole('operator', 'benchmark a local model')
    if (store.threads.some((thread) => {
      const providerId = thread.providerId ?? surfaceModelOverride(store.settings, thread.kind === 'chat' ? 'chat' : 'code')?.providerId ?? store.settings.activeProviderId
      return providerId === 'tensorfold' && !thread.orchestratorHostId && thread.workspace?.kind !== 'ssh' && (thread.status === 'running' || aborters.has(thread.id))
    })) throw new Error('Finish or stop active local conversations before benchmarking a model.')
  }
  registerEngineRuntimeIpc(engineRuntimes, { assertIdle: assertCanBenchmark, reserve: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['engine:local'] }) })
  registerModelBenchmarkIpc(engineService, { assertCanBenchmark, reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['engine:local'] }) })
  registerEngineIpc(engineService, {
    assertCanCompare: () => {
      requireSecurityRole('operator', 'compare two local models')
      if (store.threads.some((thread) => {
        const providerId = thread.providerId ?? surfaceModelOverride(store.settings, thread.kind === 'chat' ? 'chat' : 'code')?.providerId ?? store.settings.activeProviderId
        return providerId === 'tensorfold' && !thread.orchestratorHostId && thread.workspace?.kind !== 'ssh' && (thread.status === 'running' || aborters.has(thread.id))
      })) throw new Error('Finish or stop active local conversations before comparing models.')
    },
    reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['engine:local'] }),
    assertCanEnableCortheon: () => {
      if (store.threads.some((thread) => aborters.has(thread.id) && (thread.phoneLocalOnly || thread.googleLocalOnly || thread.bot?.executionTarget === 'vm' || isRestrictedBot(thread.bot)))) {
        throw new Error('Stop active phone, isolated or restricted bot conversations before enabling Cortheon; hidden research cannot bypass their execution boundaries.')
      }
    },
  })
  const phones = initializePhoneHost({
    appHome: vontraHomeDir(), supported: !HEADLESS_MODE,
    authorizeThread: (id) => {
      const thread = store.threads.find((candidate) => candidate.id === id)
      if (HEADLESS_MODE || !thread || thread.kind !== 'chat' || thread.orchestratorHostId || thread.workspace?.kind === 'ssh') throw new Error('Phone control requires a chat in this local desktop app.')
      const provider = store.settings.providers.find((candidate) => candidate.id === (thread.providerId ?? store.settings.activeProviderId))
      if (!provider || isExternalProvider(runtimeProvider(provider, thread.providerAccountId))) throw new Error('Choose the built-in TensorFold engine or another model running on this Mac before granting phone access.')
      if (engineService()?.hasExternalResearchEnabled()) throw new Error('Turn Cortheon off in AI settings before granting phone access; phone chats keep research and model processing local.')
    },
    markChatLocal: async (id) => {
      const thread = store.threads.find((candidate) => candidate.id === id)
      if (!thread) throw new Error('This chat is no longer available.')
      if (thread.phoneLocalOnly) return
      thread.phoneLocalOnly = true
      try { await saveThreads(store.threads) } catch (error) { delete thread.phoneLocalOnly; throw error }
      sendToRenderer('thread:update', thread)
    },
    cancelThread: (id) => { cancelThreadRun(id) },
    emit: (snapshot) => { sendToRenderer('phone:event', snapshot) },
  })
  registerPhoneIpc(phones, { reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['phone:local'] }) })
  const botVms = initializeBotVmHost({
    appHome: vontraHomeDir(), supported: !HEADLESS_MODE,
    reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['bot:vm'] }),
  })
  registerBotVmIpc(botVms, {
    authorizeBot: (id) => {
      if (HEADLESS_MODE || !store.settings?.agentPresets.some((bot) => bot.id === id)) throw new Error('Choose a saved bot in this local desktop app.')
    },
    authorizeBotStart: (id) => {
      const preset = store.settings?.agentPresets.find((bot) => bot.id === id)
      if (HEADLESS_MODE || !preset || preset.executionTarget !== 'vm') throw new Error('Choose a saved VM bot in this local desktop app.')
      assertBotLifecycleIdle(id)
      captureBotConfig(preset, store.settings.providers)
    },
    cancelBotRuns: (id) => {
      for (const thread of store.threads) if (thread.bot?.presetId === id) aborters.get(thread.id)?.abort()
    },
    reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['bot:vm'] }),
  })
  const isQuestionRunActive = (threadId: string, runId: string) => {
    const controller = aborters.get(threadId)
    return !!controller && !controller.signal.aborted && store.threads.find((thread) => thread.id === threadId)?.activeRunContract?.runId === runId
  }
  const questionPersistence = createUserQuestionPersistence({
    getThread: (id) => store.threads.find((thread) => thread.id === id) ?? null,
    save: (run, patch) => saveThreadQuestion(() => store.threads, run, patch),
    emit: (run, patch) => {

      if (patch.question?.state !== 'interrupted' && !isQuestionRunActive(run.threadId, run.runId)) return
      if (patch.question?.state === 'pending' && !getUserQuestionHost()?.isWaiting(patch.question.id)) return
      emitToRenderer({ ...run, type: 'item_update', id: run.itemId, patch })
    },
  })
  const userQuestions = HEADLESS_MODE ? null : initializeUserQuestionHost({
    getThread: (id) => store.threads.find((thread) => thread.id === id) ?? null,
    isRunActive: isQuestionRunActive,
    commitQuestion: questionPersistence.commit,
  })
  if (userQuestions) registerUserQuestionIpc(userQuestions, { reserveOperation: () => workspaceMutationGuard.reserveStart({ workspaceKeys: ['user:question'] }) })

  return { phones, botVms, userQuestions, questionPersistence }
}
