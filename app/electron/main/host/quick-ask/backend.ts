import { createHash } from 'node:crypto'
import { store } from '../../state'
import { engineService } from '../engine-runtime'
import { requireSecurityRole } from '../audit'
import { providerWithAccount } from '../../../agent/provider-accounts'
import { isEngineProvider } from '../../../agent/engine/provider'
import { isExternalProvider } from '../../../../shared/provider-trust'
import type { QuickAskRoute } from '../../../../shared/quick-ask'
import { qualificationStream } from '../../../agent/model-qualification/transport'
import { generateAsk, type AskGeneration } from '../../../agent/quick-ask/generate'
import { executeWebTool } from '../../../agent/tool-dispatch-web'
import { modelQualificationService } from '../model-qualification-runtime'
import { currentModelQualification } from '../../../../shared/model-fit'
import { selectedResponseLocale, withResponseLanguage } from '../../../agent/response-language'
import { assertProviderEndpointResolved } from '../../../agent/security'
import { buildCapabilityRegistry, capabilityAuditForTool } from '../../../agent/capability-registry'
import { trackAskOperation, reserveAskOperation } from './admission'

const fingerprint = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex')
export function askRouteId(providerId: string, accountId: string | null, model: string, hostId: string | null = null): string {
  return JSON.stringify([hostId, providerId, accountId, model])
}
export async function localAskRoutes(): Promise<QuickAskRoute[]> {
  requireSecurityRole('operator', 'choose an Ask TensorFold model')
  const engine = await engineService()?.snapshot()
  const reports = await modelQualificationService().reports()
  const routes: QuickAskRoute[] = []
  for (const stored of store.settings.providers) {
    if (stored.transport && stored.transport !== 'chat-completions') continue
    const accounts = [null, ...(stored.accounts ?? []).filter(account => account.enabled !== false).map(account => account.id)]
    for (const accountId of accounts) {
      const provider = providerWithAccount(stored, accountId)
      if (['openai', 'openrouter', 'ai-gateway'].includes(provider.source ?? '') && !provider.apiKey.trim()) continue
      const models = isEngineProvider(provider) ? engine?.installed.map(item => item.repo) ?? [] : provider.models
      for (const model of Array.from(new Set(models)).slice(0, 200)) {
        const report = currentModelQualification(reports, { providerId: provider.id, accountId, model })
        const observed = report?.checks.vision.state
        const accountName = stored.accounts?.find(account => account.id === accountId)?.name
        routes.push({ id: askRouteId(provider.id, accountId, model), providerId: provider.id, accountId, model,
          label: `${model} · ${provider.name}${accountName ? ` · ${accountName}` : ''}`,
          destination: provider.remoteCompute ? 'remote' : isExternalProvider(provider) ? 'cloud' : 'device',
          location: provider.remoteCompute ? 'Remote inference' : isExternalProvider(provider) ? provider.name : 'This computer',
          state: isEngineProvider(provider) && (engine?.status.state !== 'ready' || engine.status.model !== model) ? 'load-required' : 'ready',
          vision: observed === 'observed' ? 'observed' : observed === 'unsupported' || observed === 'failed' ? 'unsupported' : 'unknown',
          binding: fingerprint([provider.id, accountId, model, provider.baseUrl, provider.kind, provider.remoteCompute]) })
      }
    }
  }
  return routes
}
export async function resolveAskRoute(routeId: string) {
  const route = (await localAskRoutes()).find(route => route.id === routeId)
  if (!route || route.state !== 'ready') throw new Error('The selected model is unavailable or needs loading. No alternative model was substituted.')
  const stored = store.settings.providers.find(provider => provider.id === route.providerId)!
  const provider = providerWithAccount(stored, route.accountId)
  const captured = fingerprint(provider)
  await assertProviderEndpointResolved(provider)
  const assertCurrent = () => {
    requireSecurityRole('operator', 'ask the selected model')
    const current = store.settings.providers.find(item => item.id === route.providerId)
    if (!current || route.accountId && !current.accounts?.some(account => account.id === route.accountId && account.enabled !== false) || fingerprint(providerWithAccount(current, route.accountId)) !== captured) throw new Error('The selected model account changed. Your ask stopped without changing its destination.')
  }
  assertCurrent()
  return { binding: route.binding, params: { provider, baseUrl: provider.baseUrl, apiKey: provider.apiKey, model: route.model }, assertCurrent }
}
export async function* localAskGeneration(input: AskGeneration, signal: AbortSignal) {
  const operation = trackAskOperation(signal)
  const locale = selectedResponseLocale(store.settings.uiPreferences)
  const allowWeb = (name: string) => {
    if (!capabilityAuditForTool(buildCapabilityRegistry({ disabledCapabilityIds: store.settings.capabilities?.disabledCapabilityIds ?? [] }), name)) throw new Error('Web research is disabled in TensorFold capability settings.')
  }
  try {
    if (input.web) { allowWeb('web_search'); allowWeb('web_fetch') }
    yield* generateAsk(input, operation.signal, {
    resolve: resolveAskRoute,
    stream: params => qualificationStream({ ...params, messages: withResponseLanguage(params.messages, locale) }),
    web: async (name, args, abort) => {
      abort.throwIfAborted()
      allowWeb(name)
      const result = await executeWebTool(name, args, { cwd: '', signal: abort })
      abort.throwIfAborted(); return result
    },
  }) } finally { operation.release() }
}
export async function loadLocalAskModel(routeId: string): Promise<void> {
  requireSecurityRole('operator', 'load the explicitly selected Ask TensorFold model')
  const route = (await localAskRoutes()).find(route => route.id === routeId)
  if (!route || route.providerId !== 'tensorfold') throw new Error('Choose an installed TensorFold model.')
  if (store.threads.some(thread => !thread.orchestratorHostId && thread.providerId === 'tensorfold' && thread.status === 'running')) throw new Error('Finish or stop active local conversations before changing models.')
  const service = engineService()
  if (!service) throw new Error('The built-in engine is unavailable.')
  const lease = reserveAskOperation()
  try { await service.load(route.model) } finally { lease.release() }
}
