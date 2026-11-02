import type { AskGeneration, AskStep } from '../../../agent/quick-ask/generate'
import { validateAskStep } from '../../../agent/quick-ask/generate'
import { createHash } from 'node:crypto'
import type { AskSavedState } from '../../../agent/quick-ask/session'
import type { QuickAskRoute } from '../../../../shared/quick-ask'
import type { RemoteOrchestratorProfile, Thread } from '../../../agent/types'
import { createRemoteOrchestratorFetch } from '../../../agent/secure-fetch'
import { assertRemoteOrchestratorUrl } from '../../../agent/security'
import { readBoundedConnectorJson } from '../../../agent/connector-http'
import { discardResponseBody } from '../../../agent/pinned-fetch'
import { store } from '../../state'
import { localAskGeneration, localAskRoutes, loadLocalAskModel } from './backend'
import { handoffAsk } from './handoff'

function remoteIdentity(routeId: string): { host: RemoteOrchestratorProfile; localId: string; stamp: string } | null {
  const parts = JSON.parse(routeId)
  if (!Array.isArray(parts) || parts.length !== 4) throw new Error('Choose a valid model route.')
  if (parts[0] === null) return null
  const host = store.settings.remoteOrchestrators.find(host => host.id === parts[0] && host.enabled)
  if (!host) throw new Error('The selected remote host is unavailable. Your ask stays on its chosen route.')
  return { host: { ...host }, localId: JSON.stringify([null, ...parts.slice(1)]), stamp: JSON.stringify([host.serverUrl, host.authToken]) }
}
function assertHostCurrent(host: RemoteOrchestratorProfile, stamp: string): void {
  const current = store.settings.remoteOrchestrators.find(item => item.id === host.id && item.enabled)
  if (!current || JSON.stringify([current.serverUrl, current.authToken]) !== stamp) throw new Error('The selected remote host changed. The ask stopped.')
}
async function remoteRequest(host: RemoteOrchestratorProfile, pathname: string, init: RequestInit = {}): Promise<Response> {
  const base = await assertRemoteOrchestratorUrl(host.serverUrl)
  const response = await createRemoteOrchestratorFetch(base)(`${base}${pathname}`, { ...init, headers: { authorization: `Bearer ${host.authToken}`, 'content-type': 'application/json' }, signal: init.signal ?? AbortSignal.timeout(8000) })
  if (!response.ok) {
    await discardResponseBody(response)
    throw new Error(response.status === 404 || response.status === 501 ? 'Update TensorFold on this host to use Ask TensorFold.' : response.status === 401 || response.status === 403 ? 'The remote host rejected access. Check its owner connection.' : `The remote host returned HTTP ${response.status}.`)
  }
  return response
}
export async function allAskRoutes(): Promise<{ routes: QuickAskRoute[]; errors: string[] }> {
  const routes = await localAskRoutes(), errors: string[] = []
  const hosts = store.settings.remoteOrchestrators.filter(host => host.enabled).slice(0, 8)
  const results = await Promise.allSettled(hosts.map(async host => {
    const data = await readBoundedConnectorJson(await remoteRequest(host, '/v1/ask/routes'), 'Ask TensorFold routes', 512_000) as QuickAskRoute[]
    if (!Array.isArray(data) || data.length > 1000) throw new Error('The host returned an invalid model catalogue.')
    return data.map(route => {
      const parts = JSON.parse(route.id)
      if (!Array.isArray(parts) || parts.length !== 4 || parts[0] !== null || typeof route.model !== 'string' || route.model.length > 500 || typeof route.label !== 'string' || route.label.length > 1000 || !['ready', 'load-required'].includes(route.state) || !['observed', 'unknown', 'unsupported'].includes(route.vision)) throw new Error('The host returned an invalid model route.')
      if (!/^[a-f0-9]{64}$/.test(route.binding)) throw new Error('Update TensorFold on this host to pin model destinations.')
      return { ...route, id: JSON.stringify([host.id, ...parts.slice(1)]), hostId: host.id, destination: 'remote' as const, location: route.destination === 'cloud' ? `${host.name} → ${route.location}` : host.name, label: `${route.label} · ${host.name}`, upstreamBinding: route.binding, binding: createHash('sha256').update(JSON.stringify([host.serverUrl, host.authToken, route.binding])).digest('hex') }
    })
  }))
  results.forEach((result, index) => { if (result.status === 'fulfilled') routes.push(...result.value); else errors.push(`${hosts[index].name}: ${result.reason instanceof Error ? result.reason.message : 'Connection unavailable.'}`) })
  return { routes, errors }
}
export async function* readAskNdjson(response: Response, signal: AbortSignal): AsyncGenerator<AskStep> {
  if (!response.body || !response.headers.get('content-type')?.includes('application/x-ndjson')) throw new Error('The host did not return an ask stream.')
  const reader = response.body.getReader(), decoder = new TextDecoder()
  let pending = '', total = 0, finished = false
  try {
    while (true) {
      signal.throwIfAborted()
      let idle: ReturnType<typeof setTimeout> | undefined
      const timeout = new Promise<never>((_, reject) => { idle = setTimeout(() => reject(new Error('The host stopped responding. Your partial answer is preserved.')), 65_000) })
      const next = await Promise.race([reader.read(), timeout]).finally(() => clearTimeout(idle))
      if (next.done) break
      total += next.value.byteLength
      if (total > 16_000_000) throw new Error('The remote answer exceeded its limit.')
      pending += decoder.decode(next.value, { stream: true })
      if (pending.length > 4_000_000) throw new Error('The host returned an oversized stream frame.')
      let newline: number
      while ((newline = pending.indexOf('\n')) >= 0) {
        const line = pending.slice(0, newline); pending = pending.slice(newline + 1)
        if (!line.trim()) continue
        const step = JSON.parse(line)
        if (step.type === 'error') throw new Error(typeof step.error === 'string' ? step.error.slice(0, 350) : 'Remote generation failed.')
        if (!['content', 'reasoning', 'tool', 'message', 'finish'].includes(step.type)) throw new Error('The host returned an invalid stream event.')
        if ((step.type === 'content' || step.type === 'reasoning') && (typeof step.text !== 'string' || step.text.length > 64_000)) throw new Error('The host returned invalid answer text.')
        if (step.type === 'finish') finished = true
        yield validateAskStep(step)
      }
    }
    if (pending.trim() || !finished) throw new Error('The remote connection ended before the answer completed.')
  } finally { await reader.cancel().catch(() => {}); reader.releaseLock() }
}
export async function* routedAskGeneration(input: AskGeneration, signal: AbortSignal): AsyncGenerator<AskStep> {
  const remote = remoteIdentity(input.routeId)
  if (!remote) { yield* localAskGeneration(input, signal); return }
  if (!input.upstreamBinding) throw new Error('Refresh this remote model before sending.')
  const response = await remoteRequest(remote.host, '/v1/ask/stream', { method: 'POST', body: JSON.stringify({ ...input, routeId: remote.localId, binding: input.upstreamBinding, upstreamBinding: undefined }), signal })
  try { for await (const step of readAskNdjson(response, signal)) { assertHostCurrent(remote.host, remote.stamp); yield step } }
  finally { await discardResponseBody(response) }
}
export async function loadRoutedAskModel(routeId: string): Promise<void> {
  const remote = remoteIdentity(routeId)
  if (!remote) { await loadLocalAskModel(routeId); return }
  const response = await remoteRequest(remote.host, '/v1/ask/load', { method: 'POST', body: JSON.stringify({ routeId: remote.localId }), signal: AbortSignal.timeout(120_000) })
  await discardResponseBody(response); assertHostCurrent(remote.host, remote.stamp)
}
export async function handoffRoutedAsk(state: AskSavedState): Promise<Thread> {
  const remote = state.session.route && remoteIdentity(state.session.route.id)
  if (!remote) return handoffAsk(state)
  const copy = structuredClone(state)
  copy.session.draft = { revision: 0, text: '', attachments: [], routeId: null, presetId: 'ask', web: false }
  copy.session.route = { ...copy.session.route!, id: remote.localId, hostId: undefined, binding: copy.session.route!.upstreamBinding! }
  const response = await remoteRequest(remote.host, '/v1/ask/handoff', { method: 'POST', body: JSON.stringify(copy), signal: AbortSignal.timeout(20_000) })
  const thread = await readBoundedConnectorJson(response, 'Ask TensorFold conversation', 16_000_000) as Thread
  assertHostCurrent(remote.host, remote.stamp)
  if (thread.id !== `ask-${state.session.id}` || thread.kind !== 'chat') throw new Error('The host returned an invalid conversation handoff.')
  return { ...thread, orchestratorHostId: remote.host.id }
}
