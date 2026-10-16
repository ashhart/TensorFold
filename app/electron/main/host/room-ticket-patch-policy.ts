import type { Project, Provider, Room, RoomEventItem, RoomManualMemoryCommand, RoomMemoryEntry, Seat, Ticket, TicketExecutionRecord, TicketRemoteLifecycleCommand, TimelineItem } from '../../agent/types'

const ROOM_CONFIG_KEYS = new Set<keyof Room>([
  'seats',
  'mode',
  'collaborationMode',
  'deliberationEffort',
  'workerBudget',
])

const TICKET_USER_KEYS = new Set<keyof Ticket>([
  'roomId',
  'executionMode',
  'title',
  'body',
  'type',
  'status',
  'priority',
  'effort',
  'tags',
  'risk',
  'allowedCapabilities',
  'requiredEvidence',
  'acceptanceCriteria',
  'dependsOn',
  'dependencyFailurePolicy',
  'parentId',
  'order',
])
const TICKET_CREATE_KEYS = new Set<keyof Ticket>(['projectId', ...TICKET_USER_KEYS])

const ACTIVE_ROOM_STATES = new Set<Room['state']>(['queued', 'running'])
const ACTIVE_TICKET_RUN_STATES = new Set<NonNullable<Ticket['runState']>>(['queued', 'running'])
const ROOM_MODES = new Set(['plan', 'implement', 'review', 'debug', 'incident', 'research'])
const ROOM_COLLABORATION_MODES = new Set(['balanced', 'frontier_plans_local_builds', 'local_private', 'frontier_council'])
const ROOM_EFFORTS = new Set(['quick', 'normal', 'deep', 'max'])
const SEAT_LANES = new Set(['planning', 'development', 'review', 'research', 'operations'])
const ROLE_PROFILES = new Set(['planner', 'builder', 'reviewer', 'researcher', 'operator'])
const FAN_OUT_BUDGETS = new Set(['single', 'paired', 'council', 'swarm'])
const MEMORY_KINDS = new Set(['summary', 'decision', 'preference', 'finding'])
const TICKET_EXECUTION_MODES = new Set(['solo', 'frontier', 'room', 'remote', 'cli'])
const TICKET_TYPES = new Set(['epic', 'task', 'subticket', 'bug'])
const TICKET_STATUSES = new Set(['todo', 'ready', 'in_progress', 'needs_clarification', 'review', 'completed'])
const TICKET_PRIORITIES = new Set(['low', 'medium', 'high', 'urgent'])
const TICKET_EFFORTS = new Set(['small', 'medium', 'large'])
const TICKET_RISKS = new Set(['low', 'medium', 'high'])
const TICKET_CAPABILITIES = new Set(['read', 'write', 'shell', 'network', 'browser', 'connector', 'mcp'])
const TICKET_DEPENDENCY_FAILURE_POLICIES = new Set(['block', 'continue'])

function strictOwnPatch<T extends object>(input: unknown, allowed: ReadonlySet<keyof T>, label: string): Partial<T> {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error(`${label} patch must be a plain object.`)
  const prototype = Object.getPrototypeOf(input)
  if (prototype !== Object.prototype && prototype !== null) throw new Error(`${label} patch must not inherit fields.`)

  const output = Object.create(null) as Partial<T>
  for (const key of Reflect.ownKeys(input)) {
    if (typeof key !== 'string' || !allowed.has(key as keyof T)) {
      throw new Error(`${String(key)} cannot be changed through ${label} updates.`)
    }
    const descriptor = Object.getOwnPropertyDescriptor(input, key)
    if (!descriptor || !descriptor.enumerable || !('value' in descriptor)) {
      throw new Error(`${key} must be an ordinary ${label} patch field.`)
    }
    Object.defineProperty(output, key, {
      value: descriptor.value,
      enumerable: true,
      configurable: true,
      writable: true,
    })
  }
  return output
}

function assertEnum(value: unknown, allowed: ReadonlySet<string>, field: string, optional = false): void {
  if (optional && (value === undefined || value === null)) return
  if (typeof value !== 'string' || !allowed.has(value)) throw new Error(`${field} has an invalid value.`)
}

function assertNullableId(value: unknown, field: string): void {
  if (value !== null && (typeof value !== 'string' || !value.trim() || value.length > 256)) throw new Error(`${field} must be a valid id or null.`)
}

function assertPlainRecord(value: unknown, field: string): asserts value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${field} must be an object.`)
  const prototype = Object.getPrototypeOf(value)
  if (prototype !== Object.prototype && prototype !== null) throw new Error(`${field} must not inherit fields.`)
}

function assertOnlyKeys(value: Record<string, unknown>, keys: ReadonlySet<string>, field: string): void {
  for (const key of Reflect.ownKeys(value)) {
    if (typeof key !== 'string' || !keys.has(key)) throw new Error(`${field}.${String(key)} is not supported.`)
    const descriptor = Object.getOwnPropertyDescriptor(value, key)
    if (!descriptor?.enumerable || !('value' in descriptor)) throw new Error(`${field}.${key} must be an ordinary field.`)
  }
}

function assertBoundedString(value: unknown, field: string, max: number, allowEmpty = false): void {
  if (typeof value !== 'string' || (!allowEmpty && !value.trim()) || value.length > max) throw new Error(`${field} must be ${allowEmpty ? '' : 'non-empty and '}at most ${max} characters.`)
}

function assertStringList(value: unknown, field: string, maxItems = 64, maxLength = 2_000): void {
  if (!Array.isArray(value) || value.length > maxItems) throw new Error(`${field} must be an array with at most ${maxItems} items.`)
  for (const item of value) assertBoundedString(item, field, maxLength)
}

function validateSeats(value: unknown): void {
  if (!Array.isArray(value) || value.length > 16) throw new Error('seats must be an array with at most 16 entries.')
  const ids = new Set<string>()
  const allowed = new Set(['id', 'name', 'role', 'providerId', 'providerAccountId', 'model', 'lane', 'roleProfileId', 'fanOutBudget'])
  for (const [index, candidate] of value.entries()) {
    const field = `seats[${index}]`
    assertPlainRecord(candidate, field)
    assertOnlyKeys(candidate, allowed, field)
    assertBoundedString(candidate.id, `${field}.id`, 256)
    assertBoundedString(candidate.name, `${field}.name`, 128)
    assertBoundedString(candidate.role, `${field}.role`, 2_000)
    assertBoundedString(candidate.providerId, `${field}.providerId`, 256)
    assertBoundedString(candidate.model, `${field}.model`, 512)
    if (candidate.providerAccountId !== undefined && candidate.providerAccountId !== null) assertBoundedString(candidate.providerAccountId, `${field}.providerAccountId`, 256)
    if (candidate.lane !== undefined) assertEnum(candidate.lane, SEAT_LANES, `${field}.lane`)
    if (candidate.roleProfileId !== undefined) assertEnum(candidate.roleProfileId, ROLE_PROFILES, `${field}.roleProfileId`)
    if (candidate.fanOutBudget !== undefined) assertEnum(candidate.fanOutBudget, FAN_OUT_BUDGETS, `${field}.fanOutBudget`)
    if (ids.has(candidate.id as string)) throw new Error('Seat ids must be unique.')
    ids.add(candidate.id as string)
  }
}

function validateWorkerBudget(value: unknown): void {
  assertPlainRecord(value, 'workerBudget')
  assertOnlyKeys(value, new Set(['maxActive', 'maxStarts']), 'workerBudget')
  if (!Number.isInteger(value.maxActive) || Number(value.maxActive) < 1 || Number(value.maxActive) > 8) throw new Error('workerBudget.maxActive must be an integer from 1 to 8.')
  if (!Number.isInteger(value.maxStarts) || Number(value.maxStarts) < Number(value.maxActive) || Number(value.maxStarts) > 32) throw new Error('workerBudget.maxStarts must be an integer from maxActive to 32.')
}

export function pickRoomConfigPatch(input: unknown): Partial<Room> {
  const patch = strictOwnPatch<Room>(input, ROOM_CONFIG_KEYS, 'room configuration')
  if (Object.prototype.hasOwnProperty.call(patch, 'seats')) validateSeats(patch.seats)
  if (Object.prototype.hasOwnProperty.call(patch, 'mode')) assertEnum(patch.mode, ROOM_MODES, 'mode', true)
  if (Object.prototype.hasOwnProperty.call(patch, 'collaborationMode')) assertEnum(patch.collaborationMode, ROOM_COLLABORATION_MODES, 'collaborationMode')
  if (Object.prototype.hasOwnProperty.call(patch, 'deliberationEffort')) assertEnum(patch.deliberationEffort, ROOM_EFFORTS, 'deliberationEffort')
  if (Object.prototype.hasOwnProperty.call(patch, 'workerBudget')) validateWorkerBudget(patch.workerBudget)
  return patch
}

export function assertRoomSeatRoutesConfigured(
  seats: readonly Seat[],
  providers: readonly Provider[],
  project: Pick<Project, 'localOnly'> | null | undefined,
): void {
  for (const seat of seats) {
    const provider = providers.find((candidate) => candidate.id === seat.providerId)
    if (!provider) throw new Error(`Seat ${seat.name} uses an unconfigured provider.`)
    if (project?.localOnly && provider.kind !== 'local') throw new Error(`Seat ${seat.name} must use a local provider for this local-only project.`)
    const account = seat.providerAccountId ? provider.accounts?.find((candidate) => candidate.id === seat.providerAccountId && candidate.enabled) : null
    if (seat.providerAccountId && !account) throw new Error(`Seat ${seat.name} uses an unavailable provider account.`)
    const models = account?.models?.length ? account.models : provider.models
    if (!models.includes(seat.model)) throw new Error(`Seat ${seat.name} uses a model that is not offered by its provider account.`)
  }
}

export function pickRoomManualMemoryCommand(input: unknown): RoomManualMemoryCommand {
  assertPlainRecord(input, 'room memory command')
  assertEnum(input.action, new Set(['add', 'remove']), 'room memory action')
  if (input.action === 'add') {
    assertOnlyKeys(input, new Set(['action', 'kind', 'text']), 'room memory command')
    assertEnum(input.kind, MEMORY_KINDS, 'room memory kind')
    assertBoundedString(input.text, 'room memory text', 8_000)
    return { action: 'add', kind: input.kind as RoomMemoryEntry['kind'], text: (input.text as string).trim() }
  }
  assertOnlyKeys(input, new Set(['action', 'entryId']), 'room memory command')
  assertBoundedString(input.entryId, 'room memory entryId', 256)
  return { action: 'remove', entryId: input.entryId as string }
}

export function applyRoomManualMemoryCommand(
  room: Pick<Room, 'memory'>,
  command: RoomManualMemoryCommand,
  createId: () => string,
  now = Date.now(),
): RoomMemoryEntry[] {
  const current = [...(room.memory ?? [])]
  if (command.action === 'remove') {
    const entry = current.find((candidate) => candidate.id === command.entryId)
    if (!entry) return current
    if (entry.source !== 'manual') throw new Error('Only manually added room memory can be removed.')
    return current.filter((candidate) => candidate.id !== command.entryId)
  }
  const entry: RoomMemoryEntry = { id: createId(), kind: command.kind, text: command.text, source: 'manual', createdAt: now, updatedAt: now }
  if (current.length < 24) return [entry, ...current]
  let oldestManualIndex = -1
  for (let index = current.length - 1; index >= 0; index--) {
    if (current[index].source === 'manual') { oldestManualIndex = index; break }
  }
  if (oldestManualIndex < 0) throw new Error('Room memory is full of runtime-generated entries; remove or consolidate them through a room run before adding another note.')
  current.splice(oldestManualIndex, 1)
  return [entry, ...current]
}

export function roomHasActiveRun(room: Pick<Room, 'state'>, runtimeActive = false): boolean {
  return runtimeActive || ACTIVE_ROOM_STATES.has(room.state)
}

export function assertRoomConfigUpdateAllowed(room: Pick<Room, 'state'>, patch: Partial<Room>, runtimeActive = false): void {
  if (Object.keys(patch).length && roomHasActiveRun(room, runtimeActive)) {
    throw new Error('Stop the active room run before changing its configuration.')
  }
}

export function appendRoomSteeringEvent(
  room: Pick<Room, 'state' | 'seats' | 'events'>,
  text: unknown,
  createId: () => string,
  now = Date.now(),
): RoomEventItem {
  if (!roomHasActiveRun(room)) throw new Error('Start the room before sending steering guidance.')
  const body = typeof text === 'string' ? text.trim() : ''
  if (!body) throw new Error('Steering guidance cannot be empty.')
  if (body.length > 16_000) throw new Error('Steering guidance is too long (maximum 16000 characters).')
  const mention = body.match(/^@([\w.-]+)/)?.[1]?.toLowerCase()
  const target = mention
    ? room.seats.find((seat) =>
      seat.name.toLowerCase() === mention ||
      seat.roleProfileId?.toLowerCase() === mention ||
      seat.lane?.toLowerCase() === mention
    )?.name ?? 'all'
    : 'all'
  const tick = Math.max(0, ...room.events.map((event) => Number.isFinite(event.tick) ? event.tick : 0)) + 1
  const item: RoomEventItem = { id: createId(), tick, kind: 'user', seat: 'you', target, text: body, createdAt: now }
  room.events.push(item)
  return item
}

export function pickTicketUserPatch(input: unknown): Partial<Ticket> {
  const patch = strictOwnPatch<Ticket>(input, TICKET_USER_KEYS, 'ticket')
  if (Object.prototype.hasOwnProperty.call(patch, 'roomId')) assertNullableId(patch.roomId, 'roomId')
  if (Object.prototype.hasOwnProperty.call(patch, 'parentId')) assertNullableId(patch.parentId, 'parentId')
  if (Object.prototype.hasOwnProperty.call(patch, 'executionMode')) assertEnum(patch.executionMode, TICKET_EXECUTION_MODES, 'executionMode')
  if (Object.prototype.hasOwnProperty.call(patch, 'title')) assertBoundedString(patch.title, 'title', 500)
  if (Object.prototype.hasOwnProperty.call(patch, 'body')) assertBoundedString(patch.body, 'body', 64_000, true)
  if (Object.prototype.hasOwnProperty.call(patch, 'type')) assertEnum(patch.type, TICKET_TYPES, 'type')
  if (Object.prototype.hasOwnProperty.call(patch, 'status')) assertEnum(patch.status, TICKET_STATUSES, 'status')
  if (Object.prototype.hasOwnProperty.call(patch, 'priority')) assertEnum(patch.priority, TICKET_PRIORITIES, 'priority')
  if (Object.prototype.hasOwnProperty.call(patch, 'effort')) assertEnum(patch.effort, TICKET_EFFORTS, 'effort')
  if (Object.prototype.hasOwnProperty.call(patch, 'risk')) assertEnum(patch.risk, TICKET_RISKS, 'risk', true)
  if (Object.prototype.hasOwnProperty.call(patch, 'tags')) assertStringList(patch.tags, 'tags', 64, 128)
  if (Object.prototype.hasOwnProperty.call(patch, 'allowedCapabilities')) {
    if (!Array.isArray(patch.allowedCapabilities) || patch.allowedCapabilities.length > TICKET_CAPABILITIES.size || new Set(patch.allowedCapabilities).size !== patch.allowedCapabilities.length || patch.allowedCapabilities.some((item) => !TICKET_CAPABILITIES.has(item))) throw new Error('allowedCapabilities must contain unique supported values.')
  }
  if (Object.prototype.hasOwnProperty.call(patch, 'requiredEvidence')) assertStringList(patch.requiredEvidence, 'requiredEvidence')
  if (Object.prototype.hasOwnProperty.call(patch, 'acceptanceCriteria')) assertStringList(patch.acceptanceCriteria, 'acceptanceCriteria')
  if (Object.prototype.hasOwnProperty.call(patch, 'dependsOn')) assertStringList(patch.dependsOn, 'dependsOn', 64, 256)
  if (Object.prototype.hasOwnProperty.call(patch, 'dependencyFailurePolicy')) assertEnum(patch.dependencyFailurePolicy, TICKET_DEPENDENCY_FAILURE_POLICIES, 'dependencyFailurePolicy')
  if (Object.prototype.hasOwnProperty.call(patch, 'order') && (typeof patch.order !== 'number' || !Number.isFinite(patch.order))) throw new Error('order must be a finite number.')
  return patch
}

export function pickTicketCreateInput(input: unknown): Partial<Ticket> & { projectId: string } {
  const candidate = strictOwnPatch<Ticket>(input, TICKET_CREATE_KEYS, 'ticket creation')
  assertBoundedString(candidate.projectId, 'projectId', 256)
  const { projectId, ...requestedPatch } = candidate
  return { projectId: projectId!, ...pickTicketUserPatch(requestedPatch) }
}

export function ticketHasActiveRun(ticket: Pick<Ticket, 'runState'>, runtimeActive = false): boolean {
  return runtimeActive || (ticket.runState != null && ACTIVE_TICKET_RUN_STATES.has(ticket.runState))
}

export function assertTicketUserUpdateAllowed(ticket: Pick<Ticket, 'runState'>, patch: Partial<Ticket>, runtimeActive = false): void {
  if (Object.keys(patch).length && ticketHasActiveRun(ticket, runtimeActive)) {
    throw new Error('Stop the active ticket run before changing its configuration or board state.')
  }
}

export function pickTicketRemoteLifecycleCommand(input: unknown): TicketRemoteLifecycleCommand {
  assertPlainRecord(input, 'ticket lifecycle command')
  assertEnum(input.action, new Set(['begin', 'sync', 'start_failed', 'cancel']), 'ticket lifecycle action')
  const action = input.action as TicketRemoteLifecycleCommand['action']
  const allowed = action === 'begin'
    ? new Set(['action', 'threadId', 'hostId'])
    : action === 'start_failed'
      ? new Set(['action', 'message'])
      : new Set(['action'])
  assertOnlyKeys(input, allowed, 'ticket lifecycle command')
  if (action === 'begin') {
    assertBoundedString(input.threadId, 'threadId', 256)
    assertBoundedString(input.hostId, 'hostId', 256)
    return { action, threadId: input.threadId as string, hostId: input.hostId as string }
  }
  if (action === 'start_failed') {
    if (input.message !== undefined) assertBoundedString(input.message, 'message', 500, true)
    return { action, ...(typeof input.message === 'string' ? { message: input.message } : {}) }
  }
  return { action }
}

export function ticketRunEvidence(
  timeline: readonly TimelineItem[],
  execution: Pick<TicketExecutionRecord, 'timelineBaseline' | 'runId'>,
): { correlated: boolean; ran: boolean; failed: boolean } {
  const baseline = execution.timelineBaseline
  const runId = execution.runId
  if (!Number.isInteger(baseline) || (baseline as number) < 0 || !runId) return { correlated: false, ran: false, failed: false }
  const items = timeline.slice(baseline).filter((item) => item.trace?.runId === runId)
  return {
    correlated: true,
    ran: items.some((item) => item.kind !== 'user'),
    failed: items.some((item) => item.kind === 'error'),
  }
}

export function normalizeRemoteTicketHost(ticket: Ticket, hostId: string): Ticket {
  return {
    ...ticket,
    orchestratorHostId: hostId,
    execution: ticket.execution?.target === 'remote' ? { ...ticket.execution, hostId } : ticket.execution,
  }
}
