import { MODEL_CHECKS, MODEL_PROBE_REVISION, emptyModelChecks, modelQualificationKey } from '../../../shared/model-fit'
import type { ModelCheck, ModelQualificationInput, ModelQualificationReport } from '../../../shared/model-fit'
import type { Provider } from '../types'
import { runQualificationProbe, type QualificationStream } from './probe'

export interface ResolvedQualificationRoute {
  provider: Provider
  model: string
  accountId: string | null
  /** Host-generated digest of current provider/account/model/configuration identity. */
  identity: string
  allowedChecks: ModelCheck[]
  assertCurrent: () => void
}
export interface ModelQualificationOptions {
  resolveRoute: (input: ModelQualificationInput) => Promise<ResolvedQualificationRoute>
  load?: () => Promise<ModelQualificationReport[]>
  save?: (reports: ModelQualificationReport[]) => Promise<void>
  stream?: QualificationStream
  timeoutMs?: number
  now?: () => number
}
export class ModelQualificationService {
  private data: ModelQualificationReport[] | null = null
  private active: { input: ModelQualificationInput; controller: AbortController; promise: Promise<ModelQualificationReport> } | null = null
  constructor(private options: ModelQualificationOptions) {}
  async reports(): Promise<ModelQualificationReport[]> {
    const reports = await this.load()
    const current: ModelQualificationReport[] = []
    for (const report of reports) {
      try {
        const route = await this.options.resolveRoute({ requestId: report.requestId, providerId: report.providerId, accountId: report.accountId, model: report.model, checks: report.requestedChecks })
        route.assertCurrent()
        if (route.identity === report.identity && report.probeRevision === MODEL_PROBE_REVISION) current.push(structuredClone(report))
      } catch { /* Removed, disabled, private or changed routes cannot lend old capability claims. */ }
    }
    return current
  }
  check(raw: ModelQualificationInput): Promise<ModelQualificationReport> {
    const input = validateQualificationInput(raw)
    if (this.active) {
      if (JSON.stringify(this.active.input) === JSON.stringify(input)) return this.active.promise
      return Promise.reject(new Error('A model qualification check is already running; wait or cancel it first.'))
    }
    const controller = new AbortController()
    const promise = this.run(input, controller).finally(() => { if (this.active?.controller === controller) this.active = null })
    this.active = { input, controller, promise }
    return promise
  }
  async cancel(requestId: string): Promise<void> { if (this.active?.input.requestId === requestId) this.active.controller.abort() }
  async drain(): Promise<void> { await this.active?.promise.catch(() => {}) }
  async dispose(): Promise<void> { this.active?.controller.abort(); await this.drain(); this.data = null }
  private async run(input: ModelQualificationInput, controller: AbortController): Promise<ModelQualificationReport> {
    const data = await this.load()
    const route = await this.options.resolveRoute(input)
    if (route.provider.id !== input.providerId || route.model !== input.model || route.accountId !== input.accountId || !/^[a-f0-9]{64}$/.test(route.identity)) throw new Error('The host-resolved qualification route does not match this request.')
    route.assertCurrent()
    const prior = data.find(report => report.requestId === input.requestId)
    if (prior) {
      if (prior.key !== modelQualificationKey(input) || JSON.stringify(prior.requestedChecks) !== JSON.stringify(input.checks) || prior.identity !== route.identity) throw new Error('This qualification identity belongs to another route revision or check; invoke a new explicit check.')
      return structuredClone(prior)
    }
    const previous = data.filter(report => report.key === modelQualificationKey(input) && report.identity === route.identity && report.probeRevision === MODEL_PROBE_REVISION).sort((a, b) => b.revision - a.revision)[0]
    const checks = previous ? structuredClone(previous.checks) : emptyModelChecks()
    for (const check of input.checks) checks[check] = { state: 'requested', message: 'A user-invoked fixed-fixture check was requested.' }
    const report: ModelQualificationReport = { key: modelQualificationKey(input), identity: route.identity, requestId: input.requestId, requestedChecks: input.checks, probeRevision: MODEL_PROBE_REVISION, revision: (previous?.revision ?? 0) + 1, providerId: input.providerId, accountId: input.accountId, model: input.model, checkedAt: (this.options.now ?? Date.now)(), state: 'requested', checks }
    for (const check of input.checks) {
      if (controller.signal.aborted) { checks[check] = { state: 'cancelled', message: 'The user cancelled before this check was dispatched.' }; continue }
      route.assertCurrent()
      if (!route.allowedChecks.includes(check)) { checks[check] = { state: 'unsupported', message: 'This check is unavailable under the current route permissions or native capability contract.' }; continue }
      checks[check] = await runQualificationProbe(route.provider, route.model, check, { signal: controller.signal, assertCurrent: route.assertCurrent, stream: this.options.stream, timeoutMs: this.options.timeoutMs, now: this.options.now })
      route.assertCurrent()
    }
    report.state = controller.signal.aborted ? 'cancelled' : input.checks.some(check => checks[check].state === 'failed') ? 'failed' : 'complete'
    report.checkedAt = (this.options.now ?? Date.now)()
    const next = [...data, report].slice(-500)
    route.assertCurrent()
    await this.options.save?.(structuredClone(next))
    route.assertCurrent()
    this.data = next
    return structuredClone(report)
  }
  private async load(): Promise<ModelQualificationReport[]> {
    if (this.data) return this.data
    const data = await this.options.load?.() ?? []
    if (!Array.isArray(data) || data.length > 500) throw new Error('Saved model qualifications exceeded their bounded schema.')
    for (const report of data) {
      if (!report || report.key !== modelQualificationKey(report) || !Array.isArray(report.requestedChecks) || !report.checks || typeof report.checkedAt !== 'number' || !Number.isSafeInteger(report.revision)) throw new Error('Saved model qualification metadata is invalid.')
      for (const check of MODEL_CHECKS) if (!report.checks[check] || !['unknown', 'requested', 'observed', 'failed', 'unsupported', 'cancelled'].includes(report.checks[check].state) || typeof report.checks[check].message !== 'string' || report.checks[check].message.length > 600) throw new Error('Saved capability evidence is invalid.')
    }
    this.data = structuredClone(data).map(report => report.state === 'requested' ? { ...report, state: 'failed', checks: Object.fromEntries(MODEL_CHECKS.map(check => [check, report.checks[check].state === 'requested' ? { state: 'unknown', message: 'This check was interrupted before completion; invoke it again explicitly.' } : report.checks[check]])) as ModelQualificationReport['checks'] } : report)
    return this.data
  }
}
export function validateQualificationInput(raw: ModelQualificationInput): ModelQualificationInput {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw) || Object.keys(raw).some(key => !['requestId', 'providerId', 'accountId', 'model', 'checks', 'threadId'].includes(key))) throw new Error('Qualification accepts only a configured route and fixed check names.')
  const valid = (value: unknown, max = 300) => typeof value === 'string' && value.length > 0 && value.length <= max && !/[\u0000-\u001f]/.test(value)
  if (!valid(raw.requestId, 120) || !valid(raw.providerId) || !valid(raw.model) || raw.accountId !== null && !valid(raw.accountId) || raw.threadId !== undefined && !valid(raw.threadId)) throw new Error('Select a valid provider account and model before qualification.')
  if (!Array.isArray(raw.checks) || !raw.checks.length || raw.checks.length > MODEL_CHECKS.length || raw.checks.some(check => !MODEL_CHECKS.includes(check))) throw new Error('Select one or more supported fixed-fixture checks.')
  return { requestId: raw.requestId, providerId: raw.providerId, accountId: raw.accountId, model: raw.model, checks: [...new Set(raw.checks)], ...(raw.threadId ? { threadId: raw.threadId } : {}) }
}
