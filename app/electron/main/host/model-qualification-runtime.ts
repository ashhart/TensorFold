import { createHash } from 'node:crypto'
import { promises as fs } from 'node:fs'
import type { ModelCheck, ModelQualificationInput, ModelQualificationReport } from '../../../shared/model-fit'
import { ModelQualificationService } from '../../agent/model-qualification'
import { providerWithAccount } from '../../agent/provider-accounts'
import { isEngineProvider } from '../../agent/engine/provider'
import { assertProviderEndpointResolved } from '../../agent/security'
import { isExternalProvider } from '../../../shared/provider-trust'
import { readProtectedJson, writeProtectedJson, vontraHomeDir, vontraPath } from '../../store'
import { store } from '../state'
import { handle } from './ipc'
import { requireSecurityRole } from './audit'
import { engineService } from './engine-runtime'

let instance: { root: string; service: ModelQualificationService } | null = null
let accepting = true
const stamp = (input: unknown) => createHash('sha256').update(JSON.stringify(input)).digest('hex')
export function resolveQualificationRoute(input: ModelQualificationInput) {
  const root = vontraHomeDir(), stored = store.settings.providers.find(provider => provider.id === input.providerId)
  if (!stored || stored.transport && stored.transport !== 'chat-completions') throw new Error('Choose a configured API model; delegated engines cannot run fixed capability checks.')
  if (input.accountId && !stored.accounts?.some(account => account.id === input.accountId && account.enabled !== false)) throw new Error('The chosen provider account is unavailable.')
  const provider = providerWithAccount(stored, input.accountId)
  if (!isEngineProvider(provider) && !provider.models.includes(input.model)) throw new Error('Choose a model in the configured account catalogue.')
  const captured = stamp(provider), engine = isEngineProvider(provider) ? engineService()?.supervisor.status() : null
  if (isEngineProvider(provider) && (engine?.state !== 'ready' || engine.model !== input.model)) throw new Error('Load this model first; capability checks never load or switch models automatically.')
  const thread = input.threadId ? store.threads.find(thread => thread.id === input.threadId && !thread.orchestratorHostId) : null
  if (input.threadId && (!thread || thread.bot || thread.phoneLocalOnly || thread.googleLocalOnly)) throw new Error('Use a separate ordinary conversation for model qualification.')
  if (thread?.incognito && isExternalProvider(provider)) throw new Error('Private model qualification stays on this Mac.')
  const allowedChecks: ModelCheck[] = isExternalProvider(provider) ? ['text', 'cancellation', 'reasoning'] : ['text', 'cancellation', 'tools', 'vision', 'reasoning']
  const identity = stamp({ root, provider, model: input.model, engine: engine ? { processStart: engine.startedAt, port: engine.port, model: engine.model, context: engine.contextTokens } : null })
  return { provider, model: input.model, accountId: input.accountId, identity, allowedChecks, assertCurrent: () => {
    requireSecurityRole('operator', 'use current model capability evidence')
    const liveProvider = store.settings.providers.find(candidate => candidate.id === input.providerId)
    if (!accepting || !liveProvider || input.accountId && !liveProvider.accounts?.some(account => account.id === input.accountId && account.enabled !== false) || vontraHomeDir() !== root || stamp(providerWithAccount(liveProvider!, input.accountId)) !== captured) throw new Error('The route changed; run a new check for its current configuration.')
    if (isEngineProvider(provider)) { const live = engineService()?.supervisor.status(); if (live?.state !== 'ready' || live.model !== input.model || live.startedAt !== engine?.startedAt || live.port !== engine?.port) throw new Error('The loaded engine model changed; the check was stopped.') }
    if (thread && (!store.threads.includes(thread) || thread.bot || thread.phoneLocalOnly || thread.googleLocalOnly || thread.incognito && isExternalProvider(provider))) throw new Error('Conversation privacy changed; the check was stopped.')
  } }
}
export function modelQualificationService(): ModelQualificationService {
  const root = vontraHomeDir()
  if (!instance || instance.root !== root) {
    if (instance) void instance.service.dispose()
    instance = { root, service: new ModelQualificationService({
      resolveRoute: async input => resolveQualificationRoute(input),
      load: () => { if (vontraHomeDir() !== root) throw new Error('Model evidence profile changed.'); return readProtectedJson<ModelQualificationReport[]>(`${root}/state/model-qualifications.json`, [], false, true) },
      save: reports => { if (vontraHomeDir() !== root) throw new Error('Model evidence profile changed.'); return writeProtectedJson(`${root}/state/model-qualifications.json`, reports, { validateCommit: () => { if (vontraHomeDir() !== root) throw new Error('Model evidence profile changed before saving.') } }) },
    }) }
  }
  return instance.service
}
export async function quiesceModelQualifications(): Promise<void> { accepting = false; await instance?.service.dispose() }
export function resumeModelQualifications(): void { accepting = true }
export async function purgeModelQualifications(): Promise<void> { await fs.unlink(vontraPath('state', 'model-qualifications.json')).catch((error: NodeJS.ErrnoException) => { if (error.code !== 'ENOENT') throw error }) }
export function registerModelQualificationRuntime(options: { reserveOperation(): { release(): void }; assertIdle(): void }): void {
  handle('modelQualification:reports', async () => { requireSecurityRole('operator', 'read model capability evidence'); return modelQualificationService().reports() })
  handle('modelQualification:check', async (_event, input: ModelQualificationInput) => {
    requireSecurityRole('operator', 'run fixed model capability checks')
    if (!accepting) throw new Error('Model checks are paused while the workspace stops.')
    options.assertIdle()
    const lease = options.reserveOperation()
    try {
      const route = resolveQualificationRoute(input)
      await assertProviderEndpointResolved(route.provider); requireSecurityRole('operator', 'run fixed model capability checks'); route.assertCurrent()
      return await modelQualificationService().check(input)
    } finally { lease.release() }
  })
  handle('modelQualification:cancel', async (_event, requestId: string) => {
    requireSecurityRole('operator', 'cancel a model check')
    if (typeof requestId !== 'string' || requestId.length > 120) throw new Error('Choose a valid model check.')
    await modelQualificationService().cancel(requestId)
  })
}
