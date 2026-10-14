import type { Room, RoomEventItem, Seat } from './types'

export type RoomAgentNodeStatus = 'idle' | 'active' | 'done' | 'failed'

export interface RoomAgentTreeNode {
  id: string
  kind: 'seat' | 'subagent'
  name: string
  seatName: string
  role: string
  status: RoomAgentNodeStatus
  detail: string
  fanOutBudget?: Seat['fanOutBudget']
  eventId?: string
  lastEventAt: number | null
  children: RoomAgentTreeNode[]
}

type RichRoomEvent = RoomEventItem & {
  args?: RichRoomEventArgs
  agent?: unknown
  subAgent?: unknown
  agentId?: unknown
  subAgentId?: unknown
  subagentId?: unknown
  parentAgentId?: unknown
  parentId?: unknown
  parentSeat?: unknown
  status?: unknown
}

type RichRoomEventArgs = Record<string, unknown> & {
  agent?: unknown
  subAgent?: unknown
  agentId?: unknown
  subAgentId?: unknown
  subagentId?: unknown
  id?: unknown
  runId?: unknown
  parent?: unknown
  parentAgentId?: unknown
  parentId?: unknown
  parentSeat?: unknown
  name?: unknown
  task?: unknown
  status?: unknown
  result?: unknown
  error?: unknown
}

interface PendingRoomAgentNode {
  event: RichRoomEvent
  node: RoomAgentTreeNode
  fallbackSeat: RoomAgentTreeNode
  hasExplicitName: boolean
}

const FINISHED_ROOM_STATES = new Set<Room['state']>(['complete', 'stopped', 'failed', 'archived', 'error'])
const AGENT_PREFIX = 'subagent:'
const UUID_RE = /^(?:[a-z]+[-_:])?[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
const HEX_ID_RE = /^(?:[a-z]+[-_:])?[0-9a-f]{20,}$/i

export function buildRoomAgentTree(room: Pick<Room, 'id' | 'seats' | 'events' | 'state'>): RoomAgentTreeNode[] {
  const seatNodes = new Map<string, RoomAgentTreeNode>()
  const seatAliases = new Map<string, RoomAgentTreeNode>()
  const nodesByAlias = new Map<string, RoomAgentTreeNode>()
  const pending: PendingRoomAgentNode[] = []
  const explicitNames = new WeakSet<RoomAgentTreeNode>()

  for (const seat of room.seats) {
    const last = latestEventForSeat(room.events, seat.name)
    const node: RoomAgentTreeNode = {
      id: `seat:${seat.id}`,
      kind: 'seat',
      name: seat.name,
      seatName: seat.name,
      role: seat.role,
      status: seatStatus(room, seat.name, last),
      detail: last ? summarizeRoomEvent(last) : 'Waiting for room activity',
      fanOutBudget: seat.fanOutBudget,
      lastEventAt: last?.createdAt ?? null,
      children: [],
    }
    seatNodes.set(seat.name, node)
    seatAliases.set(node.id, node)
    seatAliases.set(seat.id, node)
    seatAliases.set(seat.name, node)
    nodesByAlias.set(node.id, node)
    nodesByAlias.set(seat.id, node)
    nodesByAlias.set(seat.name, node)
  }

  for (const event of room.events) {
    if (!isDelegateEvent(event)) continue
    const rich = event as RichRoomEvent
    const fallbackSeat = seatAliases.get(parentSeatForEvent(rich))
    if (!fallbackSeat) continue

    const { node, hasExplicitName } = makeSubAgentNode(rich, fallbackSeat)
    pending.push({ event: rich, node, fallbackSeat, hasExplicitName })
    if (hasExplicitName) explicitNames.add(node)
    for (const alias of aliasesForNode(node, rich)) nodesByAlias.set(alias, node)
  }

  for (const item of pending) {
    const parent = findParentNode(item.event, nodesByAlias) ?? item.fallbackSeat
    const safeParent = wouldCreateCycle(parent, item.node) ? item.fallbackSeat : parent
    item.node.seatName = safeParent.seatName
    safeParent.children.push(item.node)
    if (safeParent.status === 'idle') safeParent.status = 'active'
  }

  for (const seat of seatNodes.values()) assignFallbackNames(seat, explicitNames)
  return [...seatNodes.values()]
}

function wouldCreateCycle(parent: RoomAgentTreeNode, node: RoomAgentTreeNode): boolean {
  if (parent === node) return true
  return node.children.some((child) => wouldCreateCycle(parent, child))
}

function assignFallbackNames(parent: RoomAgentTreeNode, explicitNames: WeakSet<RoomAgentTreeNode>) {
  parent.children.forEach((child, index) => {
    if (!explicitNames.has(child)) child.name = `${parent.name} sub-agent ${index + 1}`
    assignFallbackNames(child, explicitNames)
  })
}

function latestEventForSeat(events: RoomEventItem[], seatName: string): RoomEventItem | undefined {
  for (let i = events.length - 1; i >= 0; i--) {
    if (events[i].seat === seatName) return events[i]
  }
  return undefined
}

function seatStatus(room: Pick<Room, 'state'>, seatName: string, last?: RoomEventItem): RoomAgentNodeStatus {
  if (!last) return 'idle'
  if (last.kind === 'error') return 'failed'
  if (last.kind === 'final' || FINISHED_ROOM_STATES.has(room.state)) return 'done'
  return last.seat === seatName ? 'active' : 'idle'
}

function isDelegateEvent(event: RoomEventItem): boolean {
  return event.kind === 'tool' && (event.tool === 'delegate' || event.tool === 'launch_subagent')
}

function parentSeatForEvent(event: RichRoomEvent): string {
  const parent = firstCleanString(
    event.parentSeat,
    event.args?.parentSeat,
    parentSeatValue(event.subAgent, 'id'),
    parentSeatValue(event.subAgent, 'label'),
    parentSeatValue(event.args?.subAgent, 'id'),
    parentSeatValue(event.args?.subAgent, 'label'),
    parentRefValue(event, 'label'),
    parentRefValue(event.args, 'label'),
  )
  return parent || event.seat
}

function makeSubAgentNode(event: RichRoomEvent, fallbackSeat: RoomAgentTreeNode): { node: RoomAgentTreeNode; hasExplicitName: boolean } {
  const id = firstCleanString(
    event.subAgentId,
    event.subagentId,
    event.agentId,
    event.args?.subAgentId,
    event.args?.subagentId,
    event.args?.agentId,
    event.args?.id,
    nestedValue(event.subAgent, 'id'),
    nestedValue(event.agent, 'id'),
    nestedValue(event.args?.subAgent, 'id'),
    nestedValue(event.args?.agent, 'id'),
  ) || `${event.id}:subagent`
  const name = firstUsefulName(
    event.args?.name,
    nestedValue(event.subAgent, 'name'),
    nestedValue(event.agent, 'name'),
    nestedValue(event.args?.subAgent, 'name'),
    nestedValue(event.args?.agent, 'name'),
  )
  return {
    node: {
      id: agentNodeId(id),
      kind: 'subagent',
      name,
      seatName: fallbackSeat.seatName,
      role: 'Delegate',
      status: subAgentStatus(event),
      detail: subAgentDetail(event),
      eventId: event.id,
      lastEventAt: event.createdAt ?? null,
      children: [],
    },
    hasExplicitName: !!name,
  }
}

function subAgentStatus(event: RichRoomEvent): RoomAgentNodeStatus {
  const raw = firstCleanString(event.status, event.args?.status, nestedValue(event.subAgent, 'status'), nestedValue(event.args?.subAgent, 'status'))
  if (/fail|error/i.test(raw) || /sub-agent failed/i.test(event.text)) return 'failed'
  if (/run|active|queued|pending/i.test(raw)) return 'active'
  return 'done'
}

function subAgentDetail(event: RichRoomEvent): string {
  const task = firstCleanString(event.args?.task, nestedValue(event.subAgent, 'task'), nestedValue(event.agent, 'task'), nestedValue(event.args?.subAgent, 'task'), nestedValue(event.args?.agent, 'task'))
  if (task) return task
  const result = firstCleanString(event.args?.result, nestedValue(event.subAgent, 'result'), nestedValue(event.args?.subAgent, 'result'), event.args?.error, nestedValue(event.subAgent, 'error'))
  if (result) return result
  const text = event.text
    .replace(/^delegate:\s*/i, '')
    .replace(/^sub-agent\s+(result|failed):\s*/i, '')
    .trim()
  return text || 'Delegate run'
}

function findParentNode(event: RichRoomEvent, nodesByAlias: Map<string, RoomAgentTreeNode>): RoomAgentTreeNode | null {
  for (const alias of parentAliasesForEvent(event)) {
    const parent = nodesByAlias.get(alias) ?? nodesByAlias.get(agentNodeId(alias))
    if (parent) return parent
  }
  return null
}

function parentAliasesForEvent(event: RichRoomEvent): string[] {
  const aliases = [
    parentAgentId(event.subAgent),
    parentAgentId(event.agent),
    parentAgentId(event.args?.subAgent),
    parentAgentId(event.args?.agent),
    firstCleanString(event.parentAgentId, event.parentId, event.args?.parentAgentId, event.args?.parentId, nestedValue(event.args?.parent, 'agentId'), nestedValue(event.args?.parent, 'id')),
  ]
  return [...new Set(aliases.filter(Boolean))]
}

function aliasesForNode(node: RoomAgentTreeNode, event: RichRoomEvent): string[] {
  const aliases = [
    node.id,
    node.id.replace(AGENT_PREFIX, ''),
    event.id,
    firstCleanString(event.agentId, event.subAgentId, event.subagentId, nestedValue(event.agent, 'id'), nestedValue(event.subAgent, 'id')),
    firstCleanString(event.args?.agentId, event.args?.subAgentId, event.args?.subagentId, event.args?.id, event.args?.runId, nestedValue(event.args?.agent, 'id'), nestedValue(event.args?.subAgent, 'id')),
  ]
  return [...new Set(aliases.filter(Boolean))]
}

function parentAgentId(value: unknown): string {
  const parent = nestedValue(value, 'parent')
  const kind = firstCleanString(nestedValue(parent, 'kind'))
  return kind === 'agent' ? firstCleanString(nestedValue(parent, 'id')) : ''
}

function parentRefValue(value: unknown, key: string): unknown {
  return nestedValue(nestedValue(value, 'parent'), key)
}

function parentSeatValue(value: unknown, key: string): unknown {
  const parent = nestedValue(value, 'parent')
  return firstCleanString(nestedValue(parent, 'kind')) === 'seat' ? nestedValue(parent, key) : undefined
}

function firstCleanString(...values: unknown[]): string {
  for (const value of values) {
    const text = cleanText(value)
    if (text) return text
  }
  return ''
}

function firstUsefulName(...values: unknown[]): string {
  for (const value of values) {
    const text = cleanText(value)
    if (!text) continue
    if (/^(launch[_\s-]*)?sub[_\s-]*agent$/i.test(text) || /^delegate$/i.test(text)) continue
    if (isLikelyGeneratedId(text)) continue
    return text
  }
  return ''
}

function isLikelyGeneratedId(text: string): boolean {
  if (UUID_RE.test(text) || HEX_ID_RE.test(text)) return true
  const generatedParent = text.match(/^(.*?)\s+sub-agent\s+\d+$/i)
  if (!generatedParent) return false
  return UUID_RE.test(generatedParent[1]) || HEX_ID_RE.test(generatedParent[1])
}

function cleanText(value: unknown): string {
  return typeof value === 'string' ? value.trim().replace(/\s+/g, ' ') : ''
}

function nestedValue(value: unknown, key: string): unknown {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>)[key] : undefined
}

function agentNodeId(value: string): string {
  return value.startsWith(AGENT_PREFIX) ? value : `${AGENT_PREFIX}${value}`
}

function summarizeRoomEvent(event: RoomEventItem): string {
  if (event.kind === 'tool') return event.tool ? `Used ${event.tool}` : 'Used a room tool'
  if (event.kind === 'ask' && event.target && event.target !== 'all') return `Asked ${event.target}`
  if (event.kind === 'final') return 'Posted final room answer'
  return event.text.trim() || event.kind
}
