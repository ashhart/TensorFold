import type { Command, CreateIncidentInput, Incident, Skill, WebhookEndpoint, WebhookEventKind } from '../../agent/types'

const COMMAND_FIELDS = new Set(['id', 'name', 'description', 'prompt', 'argHint'])
const SKILL_FIELDS = new Set(['id', 'name', 'description', 'body', 'enabled', 'trigger'])
const INCIDENT_FIELDS = new Set(['title', 'severity', 'status', 'summary'])
const INCIDENT_CREATE_FIELDS = new Set(['title', 'severity', 'cwd', 'summary'])
const WEBHOOK_FIELDS = new Set([
  'id', 'name', 'direction', 'enabled', 'scope', 'url', 'path', 'secretRef',
  'replayWindowSeconds', 'capturePolicy', 'triggerPolicy', 'runAutomationId',
  'eventKinds', 'approvalRequired',
])
const INCIDENT_SEVERITIES = new Set(['sev1', 'sev2', 'sev3', 'sev4'])
const INCIDENT_STATUSES = new Set(['open', 'investigating', 'mitigating', 'resolved', 'closed'])
const WEBHOOK_DIRECTIONS = new Set(['inbound', 'outbound'])
const WEBHOOK_CAPTURE_POLICIES = new Set(['metadata', 'redacted_payload', 'ttl_payload'])
const WEBHOOK_TRIGGER_POLICIES = new Set(['notify', 'suggest', 'approval', 'run_automation'])
const WEBHOOK_EVENT_KINDS = new Set<WebhookEventKind>(['agent', 'room', 'ticket', 'automation', 'mission', 'connector', 'test'])
export const RESERVED_COMMAND_NAMES = new Set(['mission', 'compact', 'goal'])
export const RESERVED_WEBHOOK_PATHS = new Set(['/integrations/whatsapp'])

function strictOwnRecord(input: unknown, allowed: ReadonlySet<string>, label: string): Record<string, unknown> {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error(`${label} must be a plain object.`)
  const prototype = Object.getPrototypeOf(input)
  if (prototype !== Object.prototype && prototype !== null) throw new Error(`${label} must not inherit fields.`)
  const out = Object.create(null) as Record<string, unknown>
  for (const key of Reflect.ownKeys(input)) {
    if (typeof key !== 'string' || !allowed.has(key)) throw new Error(`${String(key)} cannot be changed through ${label}.`)
    const descriptor = Object.getOwnPropertyDescriptor(input, key)
    if (!descriptor?.enumerable || !('value' in descriptor)) throw new Error(`${key} must be an ordinary ${label} field.`)
    // Renderer drafts commonly materialize optional fields as own `undefined` values. Treat those
    // as omitted, while still rejecting unknown/host-owned keys even when their value is undefined.
    if (descriptor.value === undefined) continue
    Object.defineProperty(out, key, { value: descriptor.value, enumerable: true, configurable: true, writable: true })
  }
  return out
}

function boundedString(value: unknown, field: string, max: number, options: { empty?: boolean; nullable?: boolean; trim?: boolean } = {}): string | null {
  if (options.nullable && value === null) return null
  if (typeof value !== 'string' || value.length > max || (!options.empty && !value.trim())) {
    throw new Error(`${field} must be ${options.empty ? '' : 'non-empty '}text no longer than ${max} characters.`)
  }
  return options.trim ? value.trim() : value
}

function optionalId(raw: Record<string, unknown>, label: string): string | undefined {
  if (!Object.prototype.hasOwnProperty.call(raw, 'id')) return undefined
  return boundedString(raw.id, `${label} id`, 256, { trim: true }) as string
}

function slug(value: unknown, field: string): string {
  const normalized = boundedString(value, field, 64, { trim: true }) as string
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]*$/.test(normalized)) throw new Error(`${field} must be a 1-64 character letter, number, dash, or underscore slug.`)
  return normalized
}

export function assertCommandNameNotReserved(name: string): void {
  if (RESERVED_COMMAND_NAMES.has(name.trim().toLocaleLowerCase())) throw new Error(`/${name.trim()} is reserved by a built-in command.`)
}

export type CommandConfigInput = Partial<Pick<Command, 'name' | 'description' | 'prompt' | 'argHint'>> & { id?: string }

export function sanitizeCommandSave(input: unknown): CommandConfigInput {
  const raw = strictOwnRecord(input, COMMAND_FIELDS, 'command save')
  const out: CommandConfigInput = {}
  const id = optionalId(raw, 'command'); if (id) out.id = id
  if (Object.prototype.hasOwnProperty.call(raw, 'name')) {
    out.name = slug(raw.name, 'Command name')
    assertCommandNameNotReserved(out.name)
  }
  if (Object.prototype.hasOwnProperty.call(raw, 'description')) out.description = boundedString(raw.description, 'Command description', 2_000, { empty: true }) as string
  if (Object.prototype.hasOwnProperty.call(raw, 'prompt')) out.prompt = boundedString(raw.prompt, 'Command prompt', 200_000, { empty: true }) as string
  if (Object.prototype.hasOwnProperty.call(raw, 'argHint')) {
    const value = boundedString(raw.argHint, 'Command argument hint', 500, { empty: true }) as string
    out.argHint = value || undefined
  }
  return out
}

export type SkillConfigInput = Partial<Pick<Skill, 'name' | 'description' | 'body' | 'enabled' | 'trigger'>> & { id?: string }

export function sanitizeSkillSave(input: unknown): SkillConfigInput {
  const raw = strictOwnRecord(input, SKILL_FIELDS, 'skill save')
  const out: SkillConfigInput = {}
  const id = optionalId(raw, 'skill'); if (id) out.id = id
  if (Object.prototype.hasOwnProperty.call(raw, 'name')) out.name = boundedString(raw.name, 'Skill name', 200, { trim: true }) as string
  if (Object.prototype.hasOwnProperty.call(raw, 'description')) out.description = boundedString(raw.description, 'Skill description', 2_000, { empty: true }) as string
  if (Object.prototype.hasOwnProperty.call(raw, 'body')) out.body = boundedString(raw.body, 'Skill body', 500_000, { empty: true }) as string
  if (Object.prototype.hasOwnProperty.call(raw, 'enabled')) {
    if (typeof raw.enabled !== 'boolean') throw new Error('Skill enabled must be a boolean.')
    out.enabled = raw.enabled
  }
  if (Object.prototype.hasOwnProperty.call(raw, 'trigger')) {
    const value = boundedString(raw.trigger, 'Skill trigger', 4_000, { empty: true }) as string
    out.trigger = value || undefined
  }
  return out
}

export function assertUniqueConfigName(records: readonly { id: string; name: string }[], name: string, excludeId: string | undefined, label: string): void {
  const normalized = name.trim().toLocaleLowerCase()
  if (records.some((record) => record.id !== excludeId && record.name.trim().toLocaleLowerCase() === normalized)) {
    throw new Error(`${label} name already exists.`)
  }
}

export function canonicalizeSuggestedCommand(candidate: Command, corpus: readonly Command[], now: number): Command {
  const config = sanitizeCommandSave({ id: candidate.id, name: candidate.name, description: candidate.description, prompt: candidate.prompt, argHint: candidate.argHint })
  assertCommandNameNotReserved(config.name!)
  assertUniqueConfigName(corpus, config.name!, undefined, 'Command')
  return { id: config.id!, name: config.name!, description: config.description ?? '', prompt: config.prompt ?? '', argHint: config.argHint, createdAt: now, updatedAt: now }
}

export function canonicalizeSuggestedSkill(candidate: Skill, current: Skill | null, corpus: readonly Skill[], now: number): Skill {
  if (current && (current.scope === 'project' || current.projectId)) throw new Error('Project skills are repo-owned and cannot be replaced by suggestions.')
  const config = sanitizeSkillSave({ id: candidate.id, name: candidate.name, description: candidate.description, body: candidate.body, enabled: candidate.enabled, trigger: candidate.trigger })
  assertUniqueConfigName(corpus, config.name!, current?.id, 'Skill')
  return {
    id: current?.id ?? config.id!,
    name: config.name!,
    description: config.description ?? '',
    body: config.body ?? '',
    enabled: current?.enabled ?? true,
    ...(current?.trigger ? { trigger: current.trigger } : {}),
    scope: current ? current.scope : 'global',
    projectId: current ? current.projectId : null,
    createdAt: current?.createdAt ?? now,
    updatedAt: now,
  }
}

export type IncidentConfigPatch = Partial<Pick<Incident, 'title' | 'severity' | 'status' | 'summary'>>

export function sanitizeIncidentPatch(input: unknown): IncidentConfigPatch {
  const raw = strictOwnRecord(input, INCIDENT_FIELDS, 'incident update')
  const out: IncidentConfigPatch = {}
  if (Object.prototype.hasOwnProperty.call(raw, 'title')) out.title = boundedString(raw.title, 'Incident title', 200, { trim: true }) as string
  if (Object.prototype.hasOwnProperty.call(raw, 'severity')) {
    if (typeof raw.severity !== 'string' || !INCIDENT_SEVERITIES.has(raw.severity)) throw new Error('Incident severity has an invalid value.')
    out.severity = raw.severity as Incident['severity']
  }
  if (Object.prototype.hasOwnProperty.call(raw, 'status')) {
    if (typeof raw.status !== 'string' || !INCIDENT_STATUSES.has(raw.status)) throw new Error('Incident status has an invalid value.')
    out.status = raw.status as Incident['status']
  }
  if (Object.prototype.hasOwnProperty.call(raw, 'summary')) {
    const value = boundedString(raw.summary, 'Incident summary', 4_000, { empty: true }) as string
    out.summary = value || undefined
  }
  return out
}

export function sanitizeIncidentCreate(input: unknown): Partial<CreateIncidentInput> {
  const raw = strictOwnRecord(input, INCIDENT_CREATE_FIELDS, 'incident creation')
  const out: Partial<CreateIncidentInput> = {}
  if (Object.prototype.hasOwnProperty.call(raw, 'title')) out.title = boundedString(raw.title, 'Incident title', 200, { trim: true }) as string
  if (Object.prototype.hasOwnProperty.call(raw, 'severity')) {
    if (typeof raw.severity !== 'string' || !INCIDENT_SEVERITIES.has(raw.severity)) throw new Error('Incident severity has an invalid value.')
    out.severity = raw.severity as Incident['severity']
  }
  if (Object.prototype.hasOwnProperty.call(raw, 'cwd')) out.cwd = boundedString(raw.cwd, 'Incident cwd', 4_096, { trim: true }) as string
  if (Object.prototype.hasOwnProperty.call(raw, 'summary')) out.summary = boundedString(raw.summary, 'Incident summary', 4_000, { empty: true }) as string
  return out
}

export type WebhookConfigInput = Partial<Pick<WebhookEndpoint,
  'name' | 'direction' | 'enabled' | 'scope' | 'url' | 'path' | 'secretRef' |
  'replayWindowSeconds' | 'capturePolicy' | 'triggerPolicy' | 'runAutomationId' |
  'eventKinds' | 'approvalRequired'>> & { id?: string }

export function canonicalWebhookPath(value: unknown): string {
  const raw = boundedString(value, 'Webhook path', 512, { trim: true }) as string
  if (!raw.startsWith('/') || raw.startsWith('//') || /[?#\\\u0000-\u001f\u007f]/.test(raw)) throw new Error('Webhook path must be a canonical pathname without a query or hash.')
  let parsed: URL
  try { parsed = new URL(raw, 'http://127.0.0.1') } catch { throw new Error('Webhook path must be a valid pathname.') }
  if (parsed.pathname !== raw || parsed.search || parsed.hash) throw new Error('Webhook path must be canonical.')
  if (raw === '/' || RESERVED_WEBHOOK_PATHS.has(raw)) throw new Error('Webhook path is reserved.')
  return raw
}

export function assertUniqueWebhookPath(endpoints: readonly Pick<WebhookEndpoint, 'id' | 'direction' | 'path'>[], path: string, excludeId?: string): void {
  if (endpoints.some((endpoint) => endpoint.id !== excludeId && endpoint.direction === 'inbound' && endpoint.path === path)) throw new Error('Webhook path already exists.')
}

export function resolveWebhookAutomationId(
  endpoint: Pick<WebhookEndpoint, 'direction' | 'triggerPolicy' | 'runAutomationId'>,
  automations: readonly { id: string; archivedAt?: number | null }[],
): string | null {
  if (endpoint.direction !== 'inbound' || endpoint.triggerPolicy !== 'run_automation') return null
  const id = endpoint.runAutomationId
  if (!id || !automations.some((automation) => automation.id === id && !automation.archivedAt)) throw new Error('Choose an active automation for this webhook.')
  return id
}

export function nextWebhookApproval(existing: Pick<WebhookEndpoint, 'direction' | 'url' | 'approvalRequired' | 'approvedAt'>, next: Pick<WebhookEndpoint, 'direction' | 'url' | 'approvalRequired'>): number | null {
  if (next.direction !== 'outbound' || next.approvalRequired === false) return null
  const destinationChanged = existing.direction !== 'outbound' || existing.url !== next.url
  const approvalRearmed = existing.approvalRequired === false
  return destinationChanged || approvalRearmed ? null : existing.approvedAt ?? null
}

export function nextWebhookApprovalRequired(
  existing: Pick<WebhookEndpoint, 'direction' | 'approvalRequired'>,
  nextDirection: WebhookEndpoint['direction'],
  requested: boolean | undefined,
): boolean {
  if (nextDirection !== 'outbound') return false
  if (requested !== undefined) return requested
  if (existing.direction !== 'outbound') return true
  return existing.approvalRequired ?? true
}

export function stampWebhookApproval(endpoint: WebhookEndpoint, now: number): WebhookEndpoint {
  if (endpoint.direction !== 'outbound' || endpoint.approvalRequired === false) throw new Error('This webhook does not require outbound destination approval.')
  if (!Number.isFinite(now) || now < 0) throw new Error('Webhook approval timestamp is invalid.')
  return { ...endpoint, approvedAt: now, updatedAt: now }
}

export type WebhookApprovalSnapshot = Pick<WebhookEndpoint, 'id' | 'direction' | 'url' | 'approvalRequired' | 'updatedAt'>

export function captureWebhookApproval(endpoint: WebhookEndpoint): WebhookApprovalSnapshot {
  return { id: endpoint.id, direction: endpoint.direction, url: endpoint.url, approvalRequired: endpoint.approvalRequired, updatedAt: endpoint.updatedAt }
}

export function assertWebhookApprovalCurrent(endpoint: WebhookEndpoint, expected: WebhookApprovalSnapshot): void {
  if (endpoint.id !== expected.id || endpoint.direction !== expected.direction || endpoint.url !== expected.url || endpoint.approvalRequired !== expected.approvalRequired || endpoint.updatedAt !== expected.updatedAt) {
    throw new Error('Webhook destination changed while approval was being checked. Review and approve it again.')
  }
}

export function sanitizeWebhookSave(input: unknown): WebhookConfigInput {
  const raw = strictOwnRecord(input, WEBHOOK_FIELDS, 'webhook save')
  const out: WebhookConfigInput = {}
  const id = optionalId(raw, 'webhook'); if (id) out.id = id
  if (Object.prototype.hasOwnProperty.call(raw, 'name')) out.name = boundedString(raw.name, 'Webhook name', 200, { trim: true }) as string
  if (Object.prototype.hasOwnProperty.call(raw, 'direction')) {
    if (typeof raw.direction !== 'string' || !WEBHOOK_DIRECTIONS.has(raw.direction)) throw new Error('Webhook direction has an invalid value.')
    out.direction = raw.direction as WebhookEndpoint['direction']
  }
  if (Object.prototype.hasOwnProperty.call(raw, 'enabled')) {
    if (typeof raw.enabled !== 'boolean') throw new Error('Webhook enabled must be a boolean.')
    out.enabled = raw.enabled
  }
  if (Object.prototype.hasOwnProperty.call(raw, 'scope')) {
    const scope = strictOwnRecord(raw.scope, new Set(['projectId', 'workspaceId']), 'webhook scope')
    out.scope = {}
    for (const field of ['projectId', 'workspaceId'] as const) {
      if (!Object.prototype.hasOwnProperty.call(scope, field)) continue
      const max = field === 'workspaceId' ? 4_096 : 256
      out.scope[field] = boundedString(scope[field], `Webhook scope ${field}`, max, { nullable: true, trim: true })
    }
  }
  if (Object.prototype.hasOwnProperty.call(raw, 'url')) out.url = boundedString(raw.url, 'Webhook URL', 4_096, { empty: true, trim: true }) as string
  if (Object.prototype.hasOwnProperty.call(raw, 'path')) out.path = raw.path === '' ? '' : canonicalWebhookPath(raw.path)
  if (Object.prototype.hasOwnProperty.call(raw, 'secretRef')) out.secretRef = boundedString(raw.secretRef, 'Webhook secret', 16_384, { empty: true }) as string
  if (Object.prototype.hasOwnProperty.call(raw, 'replayWindowSeconds')) {
    if (!Number.isInteger(raw.replayWindowSeconds) || Number(raw.replayWindowSeconds) < 1 || Number(raw.replayWindowSeconds) > 86_400) throw new Error('Webhook replay window must be an integer from 1 to 86400 seconds.')
    out.replayWindowSeconds = raw.replayWindowSeconds as number
  }
  if (Object.prototype.hasOwnProperty.call(raw, 'capturePolicy')) {
    if (typeof raw.capturePolicy !== 'string' || !WEBHOOK_CAPTURE_POLICIES.has(raw.capturePolicy)) throw new Error('Webhook capture policy has an invalid value.')
    out.capturePolicy = raw.capturePolicy as WebhookEndpoint['capturePolicy']
  }
  if (Object.prototype.hasOwnProperty.call(raw, 'triggerPolicy')) {
    if (typeof raw.triggerPolicy !== 'string' || !WEBHOOK_TRIGGER_POLICIES.has(raw.triggerPolicy)) throw new Error('Webhook trigger policy has an invalid value.')
    out.triggerPolicy = raw.triggerPolicy as WebhookEndpoint['triggerPolicy']
  }
  if (Object.prototype.hasOwnProperty.call(raw, 'runAutomationId')) out.runAutomationId = boundedString(raw.runAutomationId, 'Webhook automation id', 256, { nullable: true, trim: true }) as string | null
  if (Object.prototype.hasOwnProperty.call(raw, 'eventKinds')) {
    if (!Array.isArray(raw.eventKinds) || raw.eventKinds.length > WEBHOOK_EVENT_KINDS.size) throw new Error('Webhook event kinds must be a bounded array.')
    const kinds = raw.eventKinds as unknown[]
    if (kinds.some((kind) => typeof kind !== 'string' || !WEBHOOK_EVENT_KINDS.has(kind as WebhookEventKind))) throw new Error('Webhook event kind has an invalid value.')
    if (new Set(kinds).size !== kinds.length) throw new Error('Webhook event kinds must be unique.')
    out.eventKinds = [...kinds] as WebhookEventKind[]
  }
  if (Object.prototype.hasOwnProperty.call(raw, 'approvalRequired')) {
    if (typeof raw.approvalRequired !== 'boolean') throw new Error('Webhook approvalRequired must be a boolean.')
    out.approvalRequired = raw.approvalRequired
  }
  return out
}
