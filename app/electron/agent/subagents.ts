import { randomUUID } from 'node:crypto'
import type { Room, Seat, Thread, TimelineItem } from './types'

export type SubAgentParentKind = 'seat' | 'thread' | 'agent'
export type SubAgentStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'
export type SubAgentEventKind = 'created' | 'started' | 'progress' | 'result' | 'error' | 'cancelled'
export type SubAgentMessageSender = 'parent' | 'child' | 'system'
export type SubAgentMessageAudience = 'parent' | 'child' | 'both'
export type SubAgentMessageKind = 'message' | 'question' | 'reply' | 'handoff' | 'heartbeat' | 'result' | 'error'

export interface SubAgentParentRef {
  kind: SubAgentParentKind
  id: string
  label: string
  projectId?: string | null
  roomId?: string | null
  threadId?: string | null
  seatId?: string | null
}

export interface SubAgentEvent {
  id: string
  kind: SubAgentEventKind
  text: string
  createdAt: number
}

export interface SubAgentMessage {
  id: string
  sender: SubAgentMessageSender
  audience: SubAgentMessageAudience
  kind: SubAgentMessageKind
  text: string
  replyToId: string | null
  createdAt: number
  readByParentAt: number | null
  readByChildAt: number | null
}

export interface AppendSubAgentMessageInput {
  sender: SubAgentMessageSender
  audience?: SubAgentMessageAudience
  kind?: SubAgentMessageKind
  text: string
  replyToId?: string | null
}

export interface ReadSubAgentMessagesInput {
  audience: Exclude<SubAgentMessageAudience, 'both'>
  unreadOnly?: boolean
  markRead?: boolean
}

export interface SubAgent {
  id: string
  parent: SubAgentParentRef
  name: string
  task: string
  status: SubAgentStatus
  projectId: string | null
  roomId: string | null
  threadId: string | null
  seatId: string | null
  providerId: string | null
  providerAccountId?: string | null
  model: string | null
  runId: string | null
  result: string | null
  error: string | null
  events: SubAgentEvent[]
  messages: SubAgentMessage[]
  createdAt: number
  updatedAt: number
  startedAt: number | null
  finishedAt: number | null
}

export interface CreateSubAgentInput {
  parent: SubAgentParentRef
  task: string
  name?: string
  providerId?: string | null
  providerAccountId?: string | null
  model?: string | null
  runId?: string | null
}

export type SubAgentLaunchSource = 'thread_tool' | 'room_delegate' | 'ui'

export interface SubAgentLaunchInput extends CreateSubAgentInput {
  source?: SubAgentLaunchSource
  cwd?: string | null
}

export interface SubAgentPatch {
  name?: string
  task?: string
  status?: SubAgentStatus
  providerId?: string | null
  providerAccountId?: string | null
  model?: string | null
  runId?: string | null
  result?: string | null
  error?: string | null
  startedAt?: number | null
  finishedAt?: number | null
}

export interface SubAgentParentGroup {
  parent: SubAgentParentRef
  parentKey: string
  label: string
  agents: SubAgent[]
}

export interface SubAgentTimelineSummary {
  kind: SubAgentEventKind
  text: string
}

type IdFactory = () => string

const TERMINAL: ReadonlySet<SubAgentStatus> = new Set(['succeeded', 'failed', 'cancelled'])

export function parentForRoomSeat(room: Pick<Room, 'id' | 'projectId'>, seat: Pick<Seat, 'id' | 'name'>): SubAgentParentRef {
  return {
    kind: 'seat',
    id: seat.id,
    label: seat.name,
    projectId: room.projectId,
    roomId: room.id,
    seatId: seat.id,
  }
}

export function parentForThread(thread: Pick<Thread, 'id' | 'title' | 'projectId'>): SubAgentParentRef {
  return {
    kind: 'thread',
    id: thread.id,
    label: thread.title || 'Thread',
    projectId: thread.projectId ?? null,
    threadId: thread.id,
  }
}

export function parentForSubAgent(agent: Pick<SubAgent, 'id' | 'name' | 'parent' | 'projectId' | 'roomId' | 'threadId' | 'seatId'>): SubAgentParentRef {
  return {
    kind: 'agent',
    id: agent.id,
    label: agent.name,
    projectId: agent.projectId ?? agent.parent.projectId ?? null,
    roomId: agent.roomId ?? agent.parent.roomId ?? null,
    threadId: agent.threadId ?? agent.parent.threadId ?? null,
    seatId: agent.seatId ?? agent.parent.seatId ?? null,
  }
}

export function parentKey(parent: Pick<SubAgentParentRef, 'kind' | 'id' | 'roomId' | 'threadId'>): string {
  const scope = parent.roomId ? `room:${parent.roomId}` : parent.threadId ? `thread:${parent.threadId}` : 'global'
  return `${parent.kind}:${scope}:${parent.id}`
}

export function subAgentCategoryLabel(parent: Pick<SubAgentParentRef, 'label'>): string {
  const label = cleanText(parent.label) || 'Agent'
  return `${label}${label.endsWith('s') ? "'" : "'s"} spawned agents`
}

export function createSubAgent(
  agents: SubAgent[],
  input: CreateSubAgentInput,
  now = Date.now(),
  idFactory: IdFactory = randomUUID,
): { agents: SubAgent[]; agent: SubAgent } {
  const parent = normalizeParent(input.parent)
  const siblingNumber = agents.filter((agent) => parentKey(agent.parent) === parentKey(parent)).length + 1
  const agent: SubAgent = {
    id: idFactory(),
    parent,
    name: cleanText(input.name) || `${parent.label} sub-agent ${siblingNumber}`,
    task: cleanText(input.task),
    status: 'queued',
    projectId: parent.projectId ?? null,
    roomId: parent.roomId ?? null,
    threadId: parent.threadId ?? null,
    seatId: parent.seatId ?? null,
    providerId: input.providerId ?? null,
    providerAccountId: input.providerAccountId ?? null,
    model: input.model ?? null,
    runId: input.runId ?? null,
    result: null,
    error: null,
    events: [{ id: idFactory(), kind: 'created', text: cleanText(input.task), createdAt: now }],
    messages: [],
    createdAt: now,
    updatedAt: now,
    startedAt: null,
    finishedAt: null,
  }
  return { agents: [agent, ...agents], agent }
}

export function prepareSubAgentLaunch(input: SubAgentLaunchInput): CreateSubAgentInput {
  return {
    parent: normalizeParent(input.parent),
    task: cleanText(input.task),
    name: cleanText(input.name),
    providerId: input.providerId ?? null,
    providerAccountId: input.providerAccountId ?? null,
    model: input.model ?? null,
    runId: input.runId ?? null,
  }
}

export function createSubAgentFromLaunch(
  agents: SubAgent[],
  input: SubAgentLaunchInput,
  now = Date.now(),
  idFactory: IdFactory = randomUUID,
): { agents: SubAgent[]; agent: SubAgent } {
  return createSubAgent(agents, prepareSubAgentLaunch(input), now, idFactory)
}

export function updateSubAgent(agents: SubAgent[], id: string, patch: SubAgentPatch, now = Date.now()): { agents: SubAgent[]; agent: SubAgent | null } {
  let updated: SubAgent | null = null
  const next = agents.map((agent) => {
    if (agent.id !== id) return agent
    const status = patch.status ?? agent.status
    const startedAt = patch.startedAt !== undefined
      ? patch.startedAt
      : status === 'running'
        ? agent.startedAt ?? now
        : agent.startedAt
    const finishedAt = patch.finishedAt !== undefined
      ? patch.finishedAt
      : TERMINAL.has(status)
        ? agent.finishedAt ?? now
        : agent.finishedAt
    updated = {
      ...agent,
      ...patch,
      name: patch.name !== undefined ? cleanText(patch.name) || agent.name : agent.name,
      task: patch.task !== undefined ? cleanText(patch.task) : agent.task,
      status,
      startedAt,
      finishedAt,
      updatedAt: now,
    }
    return updated
  })
  return { agents: next, agent: updated }
}

export function startSubAgent(agents: SubAgent[], id: string, runId: string | null = null, now = Date.now()): { agents: SubAgent[]; agent: SubAgent | null } {
  const started = updateSubAgent(agents, id, { status: 'running', runId }, now)
  if (!started.agent) return started
  return appendSubAgentEvent(started.agents, id, 'started', 'Sub-agent started.', now)
}

export function completeSubAgent(agents: SubAgent[], id: string, result: string, now = Date.now()): { agents: SubAgent[]; agent: SubAgent | null } {
  const completed = updateSubAgent(agents, id, { status: 'succeeded', result, error: null }, now)
  if (!completed.agent) return completed
  const event = appendSubAgentEvent(completed.agents, id, 'result', result, now)
  return appendSubAgentMessage(event.agents, id, { sender: 'system', audience: 'parent', kind: 'result', text: result }, now)
}

export function failSubAgent(agents: SubAgent[], id: string, error: string, now = Date.now()): { agents: SubAgent[]; agent: SubAgent | null } {
  const failed = updateSubAgent(agents, id, { status: 'failed', error, result: null }, now)
  if (!failed.agent) return failed
  const event = appendSubAgentEvent(failed.agents, id, 'error', error, now)
  return appendSubAgentMessage(event.agents, id, { sender: 'system', audience: 'parent', kind: 'error', text: error }, now)
}

export function cancelSubAgent(agents: SubAgent[], id: string, reason = 'Sub-agent cancelled.', now = Date.now()): { agents: SubAgent[]; agent: SubAgent | null } {
  const cancelled = updateSubAgent(agents, id, { status: 'cancelled', error: reason, result: null }, now)
  if (!cancelled.agent) return cancelled
  const event = appendSubAgentEvent(cancelled.agents, id, 'cancelled', reason, now)
  return appendSubAgentMessage(event.agents, id, { sender: 'system', audience: 'parent', kind: 'error', text: reason }, now)
}

export function appendSubAgentEvent(
  agents: SubAgent[],
  id: string,
  kind: SubAgentEventKind,
  text: string,
  now = Date.now(),
  idFactory: IdFactory = randomUUID,
): { agents: SubAgent[]; agent: SubAgent | null } {
  let updated: SubAgent | null = null
  const next = agents.map((agent) => {
    if (agent.id !== id) return agent
    updated = {
      ...agent,
      events: [...agent.events, { id: idFactory(), kind, text: cleanText(text), createdAt: now }],
      updatedAt: now,
    }
    return updated
  })
  return { agents: next, agent: updated }
}

export function appendSubAgentMessage(
  agents: SubAgent[],
  id: string,
  input: AppendSubAgentMessageInput,
  now = Date.now(),
  idFactory: IdFactory = randomUUID,
): { agents: SubAgent[]; agent: SubAgent | null; message: SubAgentMessage | null } {
  const text = cleanText(input.text).slice(0, 16_000)
  if (!text) return { agents, agent: agents.find((agent) => agent.id === id) ?? null, message: null }
  const message: SubAgentMessage = {
    id: idFactory(),
    sender: input.sender,
    audience: input.audience ?? defaultMessageAudience(input.sender),
    kind: input.kind && isMessageKind(input.kind) ? input.kind : 'message',
    text,
    replyToId: cleanText(input.replyToId) || null,
    createdAt: now,
    readByParentAt: input.sender === 'parent' ? now : null,
    readByChildAt: input.sender === 'child' ? now : null,
  }
  let updated: SubAgent | null = null
  const next = agents.map((agent) => {
    if (agent.id !== id) return agent
    updated = {
      ...agent,
      messages: [...agent.messages, message].slice(-200),
      updatedAt: now,
    }
    return updated
  })
  return { agents: next, agent: updated, message: updated ? message : null }
}

export function readSubAgentMessages(
  agents: SubAgent[],
  id: string,
  input: ReadSubAgentMessagesInput,
  now = Date.now(),
): { agents: SubAgent[]; agent: SubAgent | null; messages: SubAgentMessage[] } {
  const agent = agents.find((candidate) => candidate.id === id) ?? null
  if (!agent) return { agents, agent: null, messages: [] }
  const visible = agent.messages.filter((message) => {
    if (message.audience !== input.audience && message.audience !== 'both') return false
    return !input.unreadOnly || messageReadAt(message, input.audience) === null
  })
  if (input.markRead === false || visible.length === 0) return { agents, agent, messages: visible }
  const visibleIds = new Set(visible.map((message) => message.id))
  let updated: SubAgent | null = null
  const next = agents.map((candidate) => {
    if (candidate.id !== id) return candidate
    updated = {
      ...candidate,
      messages: candidate.messages.map((message) => visibleIds.has(message.id)
        ? input.audience === 'parent'
          ? { ...message, readByParentAt: message.readByParentAt ?? now }
          : { ...message, readByChildAt: message.readByChildAt ?? now }
        : message),
      updatedAt: now,
    }
    return updated
  })
  return { agents: next, agent: updated, messages: visible }
}

export function subAgentUnreadCount(agent: SubAgent, audience: Exclude<SubAgentMessageAudience, 'both'>): number {
  return agent.messages.filter((message) =>
    (message.audience === audience || message.audience === 'both') && messageReadAt(message, audience) === null,
  ).length
}

export function subAgentsForParent(agents: SubAgent[], parent: SubAgentParentRef): SubAgent[] {
  const key = parentKey(parent)
  return agents.filter((agent) => parentKey(agent.parent) === key)
}

export function groupSubAgentsByParent(agents: SubAgent[]): SubAgentParentGroup[] {
  const groups = new Map<string, SubAgentParentGroup>()
  for (const agent of agents) {
    const key = parentKey(agent.parent)
    const group = groups.get(key)
    if (group) {
      group.agents.push(agent)
    } else {
      groups.set(key, {
        parent: agent.parent,
        parentKey: key,
        label: subAgentCategoryLabel(agent.parent),
        agents: [agent],
      })
    }
  }
  return [...groups.values()].sort((a, b) => newest(b.agents) - newest(a.agents))
}

export function normalizeSubAgents(input: unknown, now = Date.now()): SubAgent[] {
  if (!Array.isArray(input)) return []
  return input
    .filter((item): item is Partial<SubAgent> => !!item && typeof item === 'object')
    .filter((item) => typeof item.id === 'string' && !!item.parent && typeof item.parent === 'object')
    .map((item) => {
      const parent = normalizeParent(item.parent as SubAgentParentRef)
      const status = isStatus(item.status) ? item.status : 'queued'
      return {
        id: item.id!,
        parent,
        name: cleanText(item.name) || `${parent.label} sub-agent`,
        task: cleanText(item.task),
        status,
        projectId: item.projectId ?? parent.projectId ?? null,
        roomId: item.roomId ?? parent.roomId ?? null,
        threadId: item.threadId ?? parent.threadId ?? null,
        seatId: item.seatId ?? parent.seatId ?? null,
        providerId: item.providerId ?? null,
        providerAccountId: item.providerAccountId ?? null,
        model: item.model ?? null,
        runId: item.runId ?? null,
        result: item.result ?? null,
        error: item.error ?? null,
        events: normalizeEvents(item.events, now),
        messages: normalizeMessages(item.messages, now),
        createdAt: finiteNumber(item.createdAt, now),
        updatedAt: finiteNumber(item.updatedAt, now),
        startedAt: nullableFiniteNumber(item.startedAt),
        finishedAt: nullableFiniteNumber(item.finishedAt),
      }
    })
    .sort((a, b) => b.updatedAt - a.updatedAt)
}

export function summarizeSubAgentTimeline(timeline: TimelineItem[], maxEvents = 8): SubAgentTimelineSummary[] {
  const summaries: SubAgentTimelineSummary[] = []
  for (const item of timeline) {
    if (item.kind === 'tool') {
      const label = [item.tool || 'tool', item.path || item.title || ''].filter(Boolean).join(' ')
      const state = item.state ? `${item.state}: ` : ''
      const output = cleanText(item.output).slice(0, 900)
      summaries.push({
        kind: item.state === 'error' || item.state === 'denied' ? 'error' : 'progress',
        text: cleanText(`${state}${label}${output ? ` — ${output}` : ''}`),
      })
    } else if (item.kind === 'error') {
      summaries.push({ kind: 'error', text: cleanText(item.text).slice(0, 900) })
    }
  }
  return summaries.filter((summary) => summary.text).slice(-Math.max(1, maxEvents))
}

export function finalSubAgentText(timeline: TimelineItem[]): string {
  for (let i = timeline.length - 1; i >= 0; i--) {
    const item = timeline[i]
    if (item.kind === 'assistant' && cleanText(item.text)) return cleanText(item.text)
    if (item.kind === 'error' && cleanText(item.text)) return `Error: ${cleanText(item.text)}`
  }
  const lastTool = [...timeline].reverse().find((item) => item.kind === 'tool' && cleanText(item.output))
  return lastTool ? cleanText(lastTool.output) : ''
}

function normalizeParent(input: SubAgentParentRef): SubAgentParentRef {
  const kind: SubAgentParentKind = input.kind === 'thread' || input.kind === 'agent' ? input.kind : 'seat'
  return {
    kind,
    id: cleanText(input.id) || 'unknown',
    label: cleanText(input.label) || 'Agent',
    projectId: input.projectId ?? null,
    roomId: input.roomId ?? null,
    threadId: input.threadId ?? null,
    seatId: input.seatId ?? null,
  }
}

function normalizeEvents(input: unknown, now: number): SubAgentEvent[] {
  if (!Array.isArray(input)) return []
  return input
    .filter((item): item is Partial<SubAgentEvent> => !!item && typeof item === 'object')
    .filter((item) => typeof item.id === 'string')
    .map((item) => ({
      id: item.id!,
      kind: isEventKind(item.kind) ? item.kind : 'progress',
      text: cleanText(item.text),
      createdAt: finiteNumber(item.createdAt, now),
    }))
}

function normalizeMessages(input: unknown, now: number): SubAgentMessage[] {
  if (!Array.isArray(input)) return []
  return input
    .filter((item): item is Partial<SubAgentMessage> => !!item && typeof item === 'object')
    .filter((item) => typeof item.id === 'string')
    .map((item) => ({
      id: item.id!,
      sender: isMessageSender(item.sender) ? item.sender : 'system',
      audience: isMessageAudience(item.audience) ? item.audience : defaultMessageAudience(isMessageSender(item.sender) ? item.sender : 'system'),
      kind: isMessageKind(item.kind) ? item.kind : 'message',
      text: cleanText(item.text).slice(0, 16_000),
      replyToId: cleanText(item.replyToId) || null,
      createdAt: finiteNumber(item.createdAt, now),
      readByParentAt: nullableFiniteNumber(item.readByParentAt),
      readByChildAt: nullableFiniteNumber(item.readByChildAt),
    }))
    .slice(-200)
}

function isStatus(value: unknown): value is SubAgentStatus {
  return value === 'queued' || value === 'running' || value === 'succeeded' || value === 'failed' || value === 'cancelled'
}

function isEventKind(value: unknown): value is SubAgentEventKind {
  return value === 'created' || value === 'started' || value === 'progress' || value === 'result' || value === 'error' || value === 'cancelled'
}

function isMessageSender(value: unknown): value is SubAgentMessageSender {
  return value === 'parent' || value === 'child' || value === 'system'
}

function isMessageAudience(value: unknown): value is SubAgentMessageAudience {
  return value === 'parent' || value === 'child' || value === 'both'
}

function isMessageKind(value: unknown): value is SubAgentMessageKind {
  return value === 'message' || value === 'question' || value === 'reply' || value === 'handoff' || value === 'heartbeat' || value === 'result' || value === 'error'
}

function defaultMessageAudience(sender: SubAgentMessageSender): SubAgentMessageAudience {
  return sender === 'parent' ? 'child' : 'parent'
}

function messageReadAt(message: SubAgentMessage, audience: Exclude<SubAgentMessageAudience, 'both'>): number | null {
  return audience === 'parent' ? message.readByParentAt : message.readByChildAt
}

function cleanText(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function finiteNumber(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function nullableFiniteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function newest(agents: SubAgent[]): number {
  return agents.reduce((max, agent) => Math.max(max, agent.updatedAt), 0)
}
