import { documentPolicySignature, validateDocumentSourceIds } from '../../../shared/document-collections'
import { selectedResponseLocale } from '../../agent/response-language'
import type { ModelHelperRequest } from '../../../shared/model-helper'
import type { ModelHelperRole } from '../../../shared/model-fit'
import { eligibleHelperRoute, currentModelQualification } from '../../../shared/model-fit'
import { isExternalProvider } from '../../../shared/provider-trust'
import { validateModelHelperRequest, runModelHelper, type ResolvedModelHelper } from '../../agent/model-helper'
import { qualificationStream } from '../../agent/model-qualification/transport'
import { store } from '../state'
import { documentLibrary } from './document-library-runtime'
import { modelQualificationService, resolveQualificationRoute } from './model-qualification-runtime'
import { handle } from './ipc'
import { requireSecurityRole } from './audit'

let active: { id: string; controller: AbortController; promise: Promise<unknown> } | null = null
let accepting = true
export async function resolveConfiguredHelper(role: ModelHelperRole, threadId: string): Promise<ResolvedModelHelper> {
  const thread = store.threads.find(thread => thread.id === threadId && !thread.orchestratorHostId)
  if (!thread || thread.archivedAt || thread.bot || thread.phoneLocalOnly || thread.googleLocalOnly || thread.workspace?.kind === 'ssh') throw new Error('Helper jobs require an ordinary local conversation; guarded account and bot histories keep their existing model boundary.')
  const sourceIds = validateDocumentSourceIds(thread.documentSourceIds)
  const documentPolicy = documentPolicySignature(sourceIds)
  const sourceConfig = () => JSON.stringify((store.settings.documents?.sources ?? []).filter(source => sourceIds === undefined || sourceIds.includes(source.id)).map(source => [source.id, source.path, source.kind, source.enabled]))
  const initialSources = sourceConfig()
  const preferred = store.settings.modelHelperDefaults?.[role]
  if (!preferred) throw new Error('Choose an optional helper model for this job in model settings first.')
  const route = resolveQualificationRoute({ ...preferred, requestId: 'helper-validation', checks: ['text'], threadId })
  if (isExternalProvider(route.provider)) throw new Error('Local helper jobs never send document or conversation data to an external provider.')
  const reports = await modelQualificationService().reports()
  const verdict = eligibleHelperRoute({ role, preferred, candidates: [{ route: preferred, local: true, available: true, report: currentModelQualification(reports, preferred) }], requirements: { localOnly: true } })
  if (!verdict.route) throw new Error(verdict.reason ?? 'The selected helper is unavailable.')
  const initial = JSON.stringify(preferred)
  return { provider: route.provider, route: verdict.route, assertCurrent: () => {
    requireSecurityRole('operator', 'run the selected local helper')
    route.assertCurrent()
    if (!accepting || !store.threads.includes(thread) || thread.archivedAt || thread.orchestratorHostId || thread.workspace?.kind === 'ssh' || thread.bot || thread.phoneLocalOnly || thread.googleLocalOnly || documentPolicySignature(thread.documentSourceIds) !== documentPolicy || sourceConfig() !== initialSources || JSON.stringify(store.settings.modelHelperDefaults?.[role]) !== initial) throw new Error('The helper model or conversation boundary changed; no alternative model was substituted.')
  } }
}
async function complete(resolved: ResolvedModelHelper, messages: Array<{ role: 'system' | 'user'; content: string }>, signal: AbortSignal): Promise<string> {
  let text = '', finished = false
  for await (const chunk of qualificationStream({ provider: resolved.provider, baseUrl: resolved.provider.baseUrl, apiKey: resolved.provider.apiKey, model: resolved.route.model, messages, maxOutputTokens: 768, requireProviderFinish: true, temperature: 0, approvalMode: 'readonly', signal, assertRequestAllowed: resolved.assertCurrent })) {
    resolved.assertCurrent()
    if (chunk.type === 'tool_calls') throw new Error('The read-only helper returned an unsolicited tool response; no action was executed or accepted.')
    if (chunk.type === 'finish') finished = chunk.reason === 'stop'
    if (chunk.type === 'content') { text += chunk.text; if (text.length > 12_000) throw new Error('Helper output exceeded its bound.') }
  }
  if (!finished) throw new Error('The helper response ended without a complete finish; no finished result was accepted.')
  return text
}
export async function quiesceModelHelpers(): Promise<void> { accepting = false; active?.controller.abort(); await active?.promise.catch(() => {}) }
export function resumeModelHelpers(): void { accepting = true }
export function registerModelHelperRuntime(options: { reserveOperation(): { release(): void }; assertIdle(): void }): void {
  handle('modelHelper:run', async (_event, input: ModelHelperRequest) => {
    requireSecurityRole('operator', 'run an explicitly selected local helper')
    if (!accepting || active) throw new Error('A helper job is already running or the workspace is stopping.')
    input = validateModelHelperRequest(input)
    options.assertIdle()
    const lease = options.reserveOperation(), controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 25_000)
    const promise = runModelHelper(input, { responseLocale: selectedResponseLocale(store.settings.uiPreferences), signal: controller.signal, resolve: request => resolveConfiguredHelper(request.role, request.threadId), complete,
      search: async query => {
        const owner = store.threads.find(thread => thread.id === input.threadId)
        if (!owner) throw new Error('The document collection owner is no longer available.')
        const result = await documentLibrary().search({ query: query.slice(0, 256), limit: 8 }, validateDocumentSourceIds(owner.documentSourceIds))
        return { text: result.passages.map((passage, index) => `[${index + 1}] ${passage.relativePath}\n${passage.text}`).join('\n\n'), sources: result.passages.map(({ documentId, passageId, revision }) => ({ documentId, passageId, revision })) }
      } })
    active = { id: input.requestId, controller, promise }
    try { return await promise }
    finally { clearTimeout(timeout); controller.abort(); if (active?.controller === controller) active = null; lease.release() }
  })
  handle('modelHelper:cancel', async (_event, requestId: string) => { requireSecurityRole('operator', 'cancel a local helper'); if (typeof requestId !== 'string' || requestId.length > 120) throw new Error('Choose a valid helper request.'); if (active?.id === requestId) active.controller.abort() })
}
