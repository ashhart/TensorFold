import { randomUUID } from 'node:crypto'
import type { AgentEvent, ConnectorEvent, Thread, TimelineItem, TraceContext, TraceSurface } from './types'

export function createTraceContext(input: {
  traceId?: string | null
  parentTraceId?: string | null
  runId?: string | null
  threadId?: string | null
  itemId?: string | null
  callId?: string | null
  surface: TraceSurface
  origin?: string | null
  at?: number | null
}): TraceContext {
  return {
    traceId: input.traceId || `tr_${randomUUID()}`,
    parentTraceId: input.parentTraceId ?? null,
    runId: input.runId ?? null,
    threadId: input.threadId ?? null,
    itemId: input.itemId ?? null,
    callId: input.callId ?? null,
    surface: input.surface,
    origin: input.origin ?? null,
    at: input.at ?? Date.now(),
  }
}

export function traceSurfaceForTimelineItem(item: TimelineItem): TraceSurface {
  if (item.tool?.startsWith('mcp__')) return 'mcp'
  if (item.tool?.startsWith('email_') || item.tool?.startsWith('message_') || item.tool?.startsWith('calendar_') || item.tool?.startsWith('document_') || item.tool?.startsWith('table_') || item.tool?.startsWith('browser_')) return 'connector'
  if (item.tool || item.command || item.diff?.length) return 'tool'
  if (item.kind === 'user' || item.kind === 'cue') return 'renderer'
  if (item.kind === 'assistant' || item.kind === 'reasoning') return 'model'
  return 'main'
}

export function ensureTimelineItemTrace(item: TimelineItem, input: { runId?: string | null; threadId?: string | null; parentTraceId?: string | null; origin?: string | null } = {}): TimelineItem {
  if (item.trace) {
    item.trace = {
      ...item.trace,
      runId: item.trace.runId ?? input.runId ?? null,
      threadId: item.trace.threadId ?? input.threadId ?? null,
      itemId: item.trace.itemId ?? item.id,
      parentTraceId: item.trace.parentTraceId ?? input.parentTraceId ?? null,
      origin: item.trace.origin ?? input.origin ?? null,
      at: item.trace.at ?? item.createdAt ?? Date.now(),
    }
    return item
  }
  item.trace = createTraceContext({
    runId: input.runId,
    threadId: input.threadId,
    itemId: item.id,
    parentTraceId: input.parentTraceId,
    surface: traceSurfaceForTimelineItem(item),
    origin: input.origin ?? item.tool ?? item.kind,
    at: item.createdAt,
  })
  return item
}

export function ensureAgentEventTrace(event: AgentEvent, origin = 'main'): AgentEvent {
  const itemId = event.type === 'item_add'
    ? event.item.id
    : event.type === 'item_delta' || event.type === 'item_update'
      ? event.id
      : event.type === 'approval_request'
        ? event.itemId
        : null
  const surface: TraceSurface = event.trace?.surface ?? (event.type === 'item_add' ? traceSurfaceForTimelineItem(event.item) : 'main')
  const trace = event.trace ?? createTraceContext({
    runId: event.runId || null,
    threadId: event.threadId,
    itemId,
    callId: event.type === 'approval_request' ? event.callId : null,
    surface,
    origin,
  })
  const next = { ...event, trace }
  if (next.type === 'item_add') {
    next.item.trace = {
      ...trace,
      runId: trace.runId ?? next.runId,
      threadId: trace.threadId ?? next.threadId,
      itemId: trace.itemId ?? next.item.id,
      surface: trace.surface ?? traceSurfaceForTimelineItem(next.item),
      origin: trace.origin ?? origin,
      at: trace.at ?? next.item.createdAt ?? Date.now(),
    }
  }
  return next
}

export function fallbackTimelineTrace(thread: Pick<Thread, 'id'>, item: TimelineItem): TraceContext {
  return item.trace ?? {
    traceId: `trace:${thread.id}:${item.id}`,
    parentTraceId: null,
    runId: null,
    threadId: thread.id,
    itemId: item.id,
    callId: null,
    surface: traceSurfaceForTimelineItem(item),
    origin: item.tool ?? item.kind,
    at: item.createdAt,
  }
}

export function traceForConnectorEvent(event: Pick<ConnectorEvent, 'id' | 'connectorId' | 'pluginName' | 'action' | 'createdAt'>): TraceContext {
  return createTraceContext({
    surface: 'connector',
    itemId: event.id,
    origin: `${event.pluginName}:${event.action}`,
    at: event.createdAt,
  })
}
