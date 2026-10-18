export const MODEL_PROBE_REVISION = 1
export const MODEL_CHECKS = ['text', 'cancellation', 'tools', 'vision', 'reasoning'] as const
export type ModelCheck = typeof MODEL_CHECKS[number]
export type ModelCheckState = 'unknown' | 'requested' | 'observed' | 'failed' | 'unsupported' | 'cancelled'
export interface ModelQualificationRoute { providerId: string; accountId: string | null; model: string }
export interface ModelQualificationInput extends ModelQualificationRoute { requestId: string; checks: ModelCheck[]; threadId?: string }
export interface ModelCheckResult { state: ModelCheckState; message: string; elapsedMs?: number; transportDrained?: boolean; providerComputeStopped?: false }
export interface ModelQualificationReport extends ModelQualificationRoute {
  key: string
  identity: string
  requestId: string
  requestedChecks: ModelCheck[]
  probeRevision: number
  revision: number
  checkedAt: number
  state: 'requested' | 'complete' | 'cancelled' | 'failed'
  checks: Record<ModelCheck, ModelCheckResult>
}
export interface ModelQualificationApi {
  modelQualificationReports(): Promise<ModelQualificationReport[]>
  modelQualificationCheck(input: ModelQualificationInput): Promise<ModelQualificationReport>
  modelQualificationCancel(requestId: string): Promise<void>
}
export interface ModelTaskRequirements { images?: boolean; tools?: boolean; reasoning?: boolean; localOnly?: boolean; privateContext?: 'incognito' | 'bot' | 'phone' | 'google'; toolsDisabled?: boolean }
export interface ModelTaskFit { disabledReason: string | null; labels: string[]; requested: ModelCheck[]; unknown: ModelCheck[] }
export function modelQualificationKey(route: ModelQualificationRoute): string { return JSON.stringify([route.providerId, route.accountId ?? null, route.model]) }
export function emptyModelChecks(): Record<ModelCheck, ModelCheckResult> { return Object.fromEntries(MODEL_CHECKS.map(check => [check, { state: 'unknown', message: 'Not checked for this provider account and model.' }])) as Record<ModelCheck, ModelCheckResult> }
export function currentModelQualification(reports: readonly ModelQualificationReport[], route: ModelQualificationRoute): ModelQualificationReport | null {
  return reports.filter(report => report.key === modelQualificationKey(route) && report.probeRevision === MODEL_PROBE_REVISION).sort((a, b) => b.revision - a.revision || b.checkedAt - a.checkedAt)[0] ?? null
}
export function modelTaskFit(input: { requirements?: ModelTaskRequirements; local: boolean; report?: ModelQualificationReport | null; toolPolicyAllowed?: boolean }): ModelTaskFit {
  const requirements = input.requirements ?? {}
  const requested: ModelCheck[] = [...(requirements.images ? ['vision' as const] : []), ...(requirements.tools ? ['tools' as const] : []), ...(requirements.reasoning ? ['reasoning' as const] : [])]
  const checks = input.report?.probeRevision === MODEL_PROBE_REVISION ? input.report.checks : emptyModelChecks()
  const unknown = requested.filter(check => ['unknown', 'requested', 'cancelled'].includes(checks[check].state))
  const disabledReason = (requirements.localOnly || requirements.privateContext === 'phone' || requirements.privateContext === 'google' || requirements.privateContext === 'bot') && !input.local ? 'This task requires a model on this Mac; external fallback is unavailable.'
    : requirements.tools && (requirements.toolsDisabled || input.toolPolicyAllowed === false) ? 'Tools are unavailable for this task under the current permission policy.'
    : requested.some(check => ['failed', 'unsupported'].includes(checks[check].state)) ? `This model has not passed the required ${requested.filter(check => ['failed', 'unsupported'].includes(checks[check].state)).join(', ')} check; choose another model or run a new explicit check.` : null
  return { disabledReason, requested, unknown, labels: (['text', 'vision', 'tools', 'reasoning'] as const).map(check => `${check === 'vision' ? 'Images' : check[0].toUpperCase() + check.slice(1)}: ${checks[check].state === 'observed' ? 'observed' : checks[check].state}${requested.includes(check) ? ' · required' : ''}`) }
}
export type ModelHelperRole = 'naming' | 'document_extraction' | 'search'
export type ModelHelperDefaults = Partial<Record<ModelHelperRole, ModelQualificationRoute>>
export function helperRouterRole(role: ModelHelperRole): 'summarizer' | 'scout' { return role === 'search' ? 'scout' : 'summarizer' }
/** Resolves a named job without broadening the conversation's data-sharing boundary. */
export function eligibleHelperRoute(input: { role: ModelHelperRole; preferred?: ModelQualificationRoute | null; candidates: Array<{ route: ModelQualificationRoute; local: boolean; available: boolean; report?: ModelQualificationReport | null }>; requirements: ModelTaskRequirements }): { route: ModelQualificationRoute | null; reason: string | null } {
  if (!input.preferred) return { route: null, reason: 'No helper model is configured for this job; the local deterministic workflow remains available.' }
  const selected = input.candidates.find(candidate => modelQualificationKey(candidate.route) === modelQualificationKey(input.preferred!))
  if (!selected?.available) return { route: null, reason: 'The chosen helper is unavailable; no other provider was substituted.' }
  const qualified = selected.report ? currentModelQualification([selected.report], selected.route) : null
  const fit = modelTaskFit({ local: selected.local, report: qualified, requirements: input.requirements })
  if (fit.disabledReason) return { route: null, reason: fit.disabledReason }
  const required: ModelCheck[] = ['text', ...(input.requirements.images ? ['vision' as const] : []), ...(input.requirements.tools ? ['tools' as const] : [])]
  if (required.some(check => qualified?.checks[check].state !== 'observed')) return { route: null, reason: 'The chosen helper has not passed the required capability checks for this job.' }
  return { route: { ...selected.route }, reason: null }
}
