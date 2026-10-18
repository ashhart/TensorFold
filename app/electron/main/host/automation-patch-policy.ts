import type { Automation, AutomationCompletionPolicy, Provider } from '../../agent/types'
import { pickStrictOwnPatch } from './project-patch-policy'
import { normalizeAutomationTimeZone } from '../../../shared/automation-time'

export const AUTOMATION_USER_FIELDS = [
  'name',
  'prompt',
  'scheduleKind',
  'everyMinutes',
  'atHour',
  'atMinute',
  'timeZone',
  'providerId',
  'providerAccountId',
  'model',
  'mode',
  'cwd',
  'deliverTo',
  'deliveryTarget',
  'previewOnly',
  'botId',
  'sourceScope',
  'routineKind',
  'enabled',
  'misfire',
  'requiredRisks',
  'acknowledgedRisks',
  'completionPolicy',
] as const satisfies readonly (keyof Automation)[]

function shortNullableString(value: unknown, field: string, max: number): asserts value is string | null | undefined {
  if (value !== undefined && value !== null && (typeof value !== 'string' || value.length > max)) {
    throw new Error(`Automation ${field} must be a string no longer than ${max} characters or null.`)
  }
}

function boundedRiskList(value: unknown, field: string): string[] | undefined {
  if (value === undefined) return undefined
  if (!Array.isArray(value) || value.length > 32 || value.some((item) => typeof item !== 'string' || !item.trim() || item.length > 100)) {
    throw new Error(`Automation ${field} must contain at most 32 short risk identifiers.`)
  }
  return [...new Set(value.map((item) => item.trim()))]
}

function completionShape(policy: AutomationCompletionPolicy | undefined): unknown {
  if (!policy) return null
  const { version: _version, ...shape } = policy
  return shape
}

function sanitizeCompletionPolicy(value: unknown, current: AutomationCompletionPolicy | undefined): AutomationCompletionPolicy | undefined {
  if (value === undefined) {
    if (!current) return undefined
    if (current.type === 'none') return { type: 'none', version: Math.max(0, current.version) }
    return { type: 'none', version: Math.max(0, current.version) + 1 }
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Automation completionPolicy must be an object or undefined.')
  const source = value as Record<string, unknown>
  const type = source.type
  if (!['none', 'max_iterations', 'on_signal', 'ai_evaluated'].includes(String(type))) throw new Error('Automation completionPolicy type is invalid.')
  const allowed = type === 'max_iterations'
    ? ['type', 'version', 'max']
    : type === 'on_signal'
      ? ['type', 'version', 'signal']
      : type === 'ai_evaluated'
        ? ['type', 'version', 'prompt']
        : ['type', 'version']
  for (const key of Object.keys(source)) if (!allowed.includes(key)) throw new Error(`Automation completionPolicy ${key} is not supported.`)

  let next: AutomationCompletionPolicy
  if (type === 'max_iterations') {
    if (!Number.isInteger(source.max) || Number(source.max) < 1 || Number(source.max) > 100) throw new Error('Automation max_iterations must be an integer from 1 to 100.')
    next = { type, version: 0, max: Number(source.max) }
  } else if (type === 'on_signal') {
    if (typeof source.signal !== 'string' || !source.signal.trim() || source.signal.length > 500) throw new Error('Automation completion signal must be non-empty text no longer than 500 characters.')
    next = { type, version: 0, signal: source.signal.trim() }
  } else if (type === 'ai_evaluated') {
    if (typeof source.prompt !== 'string' || !source.prompt.trim() || source.prompt.length > 20_000) throw new Error('Automation completion prompt must be non-empty text no longer than 20,000 characters.')
    next = { type, version: 0, prompt: source.prompt.trim() }
  } else {
    next = { type: 'none', version: 0 }
  }

  // The host owns the policy epoch. Otherwise a caller can change the completion contract while
  // preserving its version and make an in-flight run look as though it used the new policy.
  const unchanged = JSON.stringify(completionShape(current)) === JSON.stringify(completionShape(next))
  next.version = unchanged ? Math.max(0, current?.version ?? 0) : Math.max(0, current?.version ?? 0) + 1
  return next
}

function sanitizeAutomationUserFields(input: unknown, current?: Automation): Partial<Automation> {
  const patch = pickStrictOwnPatch<Automation, typeof AUTOMATION_USER_FIELDS[number]>(input, AUTOMATION_USER_FIELDS, 'automation')
  if (patch.name !== undefined && (typeof patch.name !== 'string' || !patch.name.trim() || patch.name.length > 200)) throw new Error('Automation name must be non-empty text no longer than 200 characters.')
  if (patch.prompt !== undefined && (typeof patch.prompt !== 'string' || patch.prompt.length > 1_000_000)) throw new Error('Automation prompt must be text no longer than 1,000,000 characters.')
  if (patch.scheduleKind !== undefined && !['manual', 'interval', 'daily'].includes(String(patch.scheduleKind))) throw new Error('Automation scheduleKind is invalid.')
  if (patch.mode !== undefined && !['chat', 'agent', 'digest'].includes(String(patch.mode))) throw new Error('Automation mode is invalid.')
  if (patch.deliverTo !== undefined && !['inbox', 'telegram', 'email', 'whatsapp'].includes(String(patch.deliverTo))) throw new Error('Automation deliverTo is invalid.')
  if (patch.previewOnly !== undefined && typeof patch.previewOnly !== 'boolean') throw new Error('Automation previewOnly must be a boolean.')
  shortNullableString(patch.botId, 'botId', 100)
  if (patch.botId !== undefined && !/^[a-zA-Z0-9_-]{1,100}$/.test(patch.botId)) throw new Error('Choose a saved bot ID.')
  if (patch.routineKind !== undefined && !['morning-brief', 'meeting-prep', 'follow-up'].includes(patch.routineKind)) throw new Error('Choose a supported personal routine.')
  if (patch.deliveryTarget !== undefined) {
    const target = patch.deliveryTarget
    if (!target || typeof target !== 'object' || Array.isArray(target) || Object.keys(target).some(key => !['connectorId', 'accountId', 'recipient', 'subject'].includes(key))) throw new Error('Choose an exact supported delivery target.')
    if (!['telegram', 'email', 'whatsapp'].includes(target.connectorId) || (target.accountId !== null && (typeof target.accountId !== 'string' || !/^[a-zA-Z0-9_-]{1,100}$/.test(target.accountId))) || typeof target.recipient !== 'string' || !target.recipient.trim() || target.recipient.length > 500 || /[\r\n\0]/.test(target.recipient)) throw new Error('Choose an exact account and recipient for this destination.')
    if (target.connectorId === 'whatsapp' && !target.accountId) throw new Error('Choose a linked WhatsApp account for this destination.')
    if (target.connectorId === 'email' && !target.accountId) throw new Error('Choose an email account for this destination.')
    if (target.subject !== undefined && (typeof target.subject !== 'string' || !target.subject.trim() || target.subject.length > 1_000 || /[\r\n\0]/.test(target.subject))) throw new Error('Choose a valid single-line email subject.')
    patch.deliveryTarget = { ...target, recipient: target.recipient.trim() }
  }
  if (patch.sourceScope !== undefined) {
    if (!Array.isArray(patch.sourceScope) || patch.sourceScope.length > 32 || patch.sourceScope.some(scope => !scope || typeof scope !== 'object' || Array.isArray(scope) || Object.keys(scope).some(key => !['connectorId', 'sourceId'].includes(key)) || !['email','whatsapp','telegram','imessage','calendar','docs','slack','discord','mattermost','matrix','googlechat','browser'].includes(scope.connectorId) || typeof scope.sourceId !== 'string' || !/^[a-zA-Z0-9_-]{1,100}$/.test(scope.sourceId))) throw new Error('Choose at most 32 exact source identities for this routine.')
    patch.sourceScope = [...new Map(patch.sourceScope.map(scope => [`${scope.connectorId}:${scope.sourceId}`, { ...scope }])).values()]
  }
  const resulting = { ...current, ...patch }
  if (resulting.deliverTo && resulting.deliverTo !== 'inbox' && resulting.deliveryTarget?.connectorId !== resulting.deliverTo) throw new Error('Choose a matching explicit account and recipient before enabling external delivery.')
  if ((resulting.routineKind || resulting.botId || resulting.sourceScope) && (!resulting.routineKind || !resulting.botId || !resulting.sourceScope || resulting.mode !== 'chat' || !resulting.providerId || !resulting.model)) throw new Error('Personal routines require a saved bot, pinned model, explicit source scope and read-only chat mode.')
  if (patch.misfire !== undefined && !['coalesce', 'run_each'].includes(String(patch.misfire))) throw new Error('Automation misfire policy is invalid.')
  if (patch.enabled !== undefined && typeof patch.enabled !== 'boolean') throw new Error('Automation enabled must be a boolean.')
  if (patch.everyMinutes !== undefined && (!Number.isInteger(patch.everyMinutes) || patch.everyMinutes < 1 || patch.everyMinutes > 525_600)) throw new Error('Automation everyMinutes must be an integer from 1 to 525600.')
  if (patch.atHour !== undefined && (!Number.isInteger(patch.atHour) || patch.atHour < 0 || patch.atHour > 23)) throw new Error('Automation atHour must be an integer from 0 to 23.')
  if (patch.atMinute !== undefined && (!Number.isInteger(patch.atMinute) || patch.atMinute < 0 || patch.atMinute > 59)) throw new Error('Automation atMinute must be an integer from 0 to 59.')
  if (Object.prototype.hasOwnProperty.call(patch, 'timeZone')) patch.timeZone = normalizeAutomationTimeZone(patch.timeZone)
  if ((patch.scheduleKind ?? current?.scheduleKind) === 'daily' && (!current || current.scheduleKind !== 'daily') && !(patch.timeZone ?? current?.timeZone)) {
    throw new Error('Choose an explicit IANA timezone before creating a daily schedule.')
  }
  shortNullableString(patch.providerId, 'providerId', 500)
  shortNullableString(patch.providerAccountId, 'providerAccountId', 500)
  shortNullableString(patch.model, 'model', 1_000)
  shortNullableString(patch.cwd, 'cwd', 4_096)
  if (Object.prototype.hasOwnProperty.call(patch, 'requiredRisks')) patch.requiredRisks = boundedRiskList(patch.requiredRisks, 'requiredRisks')
  if (Object.prototype.hasOwnProperty.call(patch, 'acknowledgedRisks')) patch.acknowledgedRisks = boundedRiskList(patch.acknowledgedRisks, 'acknowledgedRisks')
  if (Object.prototype.hasOwnProperty.call(patch, 'completionPolicy')) patch.completionPolicy = sanitizeCompletionPolicy(patch.completionPolicy, current?.completionPolicy)
  return patch
}

export function sanitizeAutomationCreateInput(input: unknown): Partial<Automation> {
  return sanitizeAutomationUserFields(input)
}

export function sanitizeAutomationUpdatePatch(current: Automation, input: unknown): Partial<Automation> {
  return sanitizeAutomationUserFields(input, current)
}

export function validateAutomationProviderRoute(automation: Pick<Automation, 'providerId' | 'providerAccountId' | 'model'>, providers: readonly Provider[]): void {
  if (!automation.providerId) {
    if (automation.providerAccountId) throw new Error('Choose an automation provider before choosing its account.')
    return
  }
  const provider = providers.find((candidate) => candidate.id === automation.providerId)
  if (!provider) throw new Error('Choose a configured automation provider.')
  const account = automation.providerAccountId ? provider.accounts?.find((candidate) => candidate.id === automation.providerAccountId) : null
  if (automation.providerAccountId && !account) throw new Error('Choose an account that belongs to the automation provider.')
  const models = account?.models?.length ? account.models : provider.models
  if (automation.model && models.length && !models.includes(automation.model)) throw new Error('Choose a model offered by the automation provider/account.')
}
