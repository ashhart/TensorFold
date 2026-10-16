import type { Thread, TimelineItem } from './types'

export type ThreadAgentNodeStatus = 'idle' | 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'

export interface ThreadAgentTreeNode {
  id: string
  kind: 'thread' | 'subagent'
  name: string
  status: ThreadAgentNodeStatus
  detail: string
  result: string
  eventId?: string
  lastEventAt: number | null
  children: ThreadAgentTreeNode[]
}

type RichTimelineItem = TimelineItem & {
  agent?: unknown
  subAgent?: unknown
  agentId?: unknown
  subAgentId?: unknown
  subagentId?: unknown
  parentAgentId?: unknown
  parentId?: unknown
  status?: unknown
}

type RichArgs = Record<string, unknown> & {
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
  name?: unknown
  task?: unknown
  status?: unknown
  result?: unknown
}

interface PendingThreadAgentNode {
  event: RichTimelineItem
  args: RichArgs
  node: ThreadAgentTreeNode
  hasExplicitName: boolean
}

const ROOT_PREFIX = 'thread:'
const AGENT_PREFIX = 'subagent:'
const UUID_RE = /^(?:[a-z]+[-_:])?[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
const HEX_ID_RE = /^(?:[a-z]+[-_:])?[0-9a-f]{20,}$/i

export function buildThreadAgentTree(thread: Pick<Thread, 'id' | 'title' | 'status' | 'timeline'>): ThreadAgentTreeNode {
  const threadTitle = cleanText(thread.title)
  const root: ThreadAgentTreeNode = {
    id: `${ROOT_PREFIX}${thread.id}`,
    kind: 'thread',
    name: 'Main agent',
    status: thread.status === 'error' ? 'failed' : thread.status === 'running' ? 'running' : 'idle',
    detail: threadTitle || 'Parent thread',
    result: '',
    lastEventAt: null,
    children: [],
  }
  const nodesByAlias = new Map<string, ThreadAgentTreeNode>([[thread.id, root], [root.id, root]])
  const pending: PendingThreadAgentNode[] = []
  const explicitNames = new WeakSet<ThreadAgentTreeNode>()

  for (const event of thread.timeline) {
    if (!isSubAgentLaunch(event)) continue
    const rich = event as RichTimelineItem
    const args = argsForEvent(rich)
    const { node, hasExplicitName } = makeSubAgentNode(rich, args)
    pending.push({ event: rich, args, node, hasExplicitName })
    if (hasExplicitName) explicitNames.add(node)
    for (const alias of aliasesForNode(node, rich, args)) {
      nodesByAlias.set(alias, node)
    }
  }

  for (const item of pending) {
    const parent = findParentNode(item.args, item.event, nodesByAlias) ?? root
    const safeParent = wouldCreateCycle(parent, item.node) ? root : parent
    safeParent.children.push(item.node)
    root.lastEventAt = newestDate(root.lastEventAt, item.node.lastEventAt)
  }

  assignFallbackNames(root, explicitNames)
  root.status = summarizeRootStatus(thread.status, root.children)
  return root
}

function wouldCreateCycle(parent: ThreadAgentTreeNode, node: ThreadAgentTreeNode): boolean {
  if (parent === node) return true
  return node.children.some((child) => wouldCreateCycle(parent, child))
}

function assignFallbackNames(parent: ThreadAgentTreeNode, explicitNames: WeakSet<ThreadAgentTreeNode>) {
  parent.children.forEach((child, index) => {
    if (!explicitNames.has(child)) child.name = `${parent.name} sub-agent ${index + 1}`
    assignFallbackNames(child, explicitNames)
  })
}

export function countThreadAgentNodes(node: ThreadAgentTreeNode): number {
  return node.children.reduce((sum, child) => sum + 1 + countThreadAgentNodes(child), 0)
}

function isSubAgentLaunch(event: TimelineItem): boolean {
  return event.kind === 'tool' && event.tool === 'launch_subagent'
}

function argsForEvent(event: RichTimelineItem): RichArgs {
  return event.args && typeof event.args === 'object' && !Array.isArray(event.args) ? event.args as RichArgs : {}
}

function makeSubAgentNode(event: RichTimelineItem, args: RichArgs): { node: ThreadAgentTreeNode; hasExplicitName: boolean } {
  const id = agentNodeId(firstCleanString(
    args.agentId,
    args.subAgentId,
    args.subagentId,
    args.id,
    nestedValue(args.agent, 'id'),
    nestedValue(args.subAgent, 'id'),
    event.agentId,
    event.subAgentId,
    event.subagentId,
    nestedValue(event.agent, 'id'),
    nestedValue(event.subAgent, 'id'),
    event.id,
  ))
  const name = firstUsefulName(args.name, nestedValue(args.agent, 'name'), nestedValue(args.subAgent, 'name'), nestedValue(event.agent, 'name'), nestedValue(event.subAgent, 'name'), event.title)
  const detail = firstCleanString(args.task, nestedValue(args.agent, 'task'), nestedValue(args.subAgent, 'task'), nestedValue(event.agent, 'task'), nestedValue(event.subAgent, 'task')) || subAgentOutputSummary(event.output) || 'Sub-agent launch'
  const result = firstCleanString(args.result, nestedValue(args.agent, 'result'), nestedValue(args.subAgent, 'result'), nestedValue(event.agent, 'result'), nestedValue(event.subAgent, 'result'), event.output)
  return {
    node: {
      id,
      kind: 'subagent',
      name,
      status: subAgentStatus(event, args),
      detail,
      result,
      eventId: event.id,
      lastEventAt: finiteDate(event.createdAt),
      children: [],
    },
    hasExplicitName: !!name,
  }
}

function findParentNode(args: RichArgs, event: RichTimelineItem, nodesByAlias: Map<string, ThreadAgentTreeNode>): ThreadAgentTreeNode | null {
  const aliases = [
    parentAgentId(args.subAgent),
    parentAgentId(args.agent),
    parentAgentId(event.subAgent),
    parentAgentId(event.agent),
    firstCleanString(args.parentAgentId, args.parentId, nestedValue(args.parent, 'agentId'), nestedValue(args.parent, 'id')),
    firstCleanString(event.parentAgentId, event.parentId),
  ].filter(Boolean)

  for (const alias of aliases) {
    const parent = nodesByAlias.get(alias) ?? nodesByAlias.get(agentNodeId(alias))
    if (parent) return parent
  }
  return null
}

function parentAgentId(value: unknown): string {
  const parent = nestedValue(value, 'parent')
  const kind = firstCleanString(nestedValue(parent, 'kind'))
  return kind === 'agent' ? firstCleanString(nestedValue(parent, 'id')) : ''
}

function aliasesForNode(node: ThreadAgentTreeNode, event: RichTimelineItem, args: RichArgs): string[] {
  const aliases = [
    node.id,
    node.id.replace(AGENT_PREFIX, ''),
    event.id,
    firstCleanString(args.agentId, args.subAgentId, args.subagentId, args.id, args.runId, nestedValue(args.agent, 'id'), nestedValue(args.subAgent, 'id')),
    firstCleanString(event.agentId, event.subAgentId, event.subagentId, nestedValue(event.agent, 'id'), nestedValue(event.subAgent, 'id')),
  ]
  return [...new Set(aliases.filter(Boolean))]
}

function summarizeRootStatus(threadStatus: Thread['status'], children: ThreadAgentTreeNode[]): ThreadAgentNodeStatus {
  if (threadStatus === 'error') return 'failed'
  if (children.some(hasActiveChild)) return 'running'
  if (children.some((child) => child.status === 'failed')) return 'failed'
  if (children.length === 0) return threadStatus === 'running' ? 'running' : 'idle'
  return threadStatus === 'running' ? 'running' : 'succeeded'
}

function hasActiveChild(node: ThreadAgentTreeNode): boolean {
  return node.status === 'queued' || node.status === 'running' || node.children.some(hasActiveChild)
}

function subAgentStatus(event: RichTimelineItem, args: RichArgs): ThreadAgentNodeStatus {
  const raw = firstCleanString(args.status, nestedValue(args.agent, 'status'), nestedValue(args.subAgent, 'status'), event.status, event.meta, event.output)
  if (/cancel|denied/i.test(raw) || event.state === 'denied') return 'cancelled'
  if (/fail|error/i.test(raw) || event.state === 'error') return 'failed'
  if (/queued|pending|await/i.test(raw) || event.state === 'pending' || event.state === 'awaiting') return 'queued'
  if (/run|active|working/i.test(raw) || event.state === 'running') return 'running'
  if (/success|succeed|done|complete|result/i.test(raw) || event.state === 'ok') return 'succeeded'
  return 'queued'
}

function firstUsefulName(...values: unknown[]): string {
  for (const value of values) {
    const text = cleanText(value)
    if (!text) continue
    if (isGenericSubAgentLabel(text) || isLikelyGeneratedId(text)) continue
    return text
  }
  return ''
}

function isGenericSubAgentLabel(text: string): boolean {
  return /^(launch[_\s-]*)?sub[_\s-]*agent$/i.test(text) || /^delegate$/i.test(text)
}

function isLikelyGeneratedId(text: string): boolean {
  if (UUID_RE.test(text) || HEX_ID_RE.test(text)) return true
  const generatedParent = text.match(/^(.*?)\s+sub-agent\s+\d+$/i)
  if (!generatedParent) return false
  return UUID_RE.test(generatedParent[1]) || HEX_ID_RE.test(generatedParent[1])
}

function firstCleanString(...values: unknown[]): string {
  for (const value of values) {
    const text = cleanText(value)
    if (text) return text
  }
  return ''
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

function subAgentOutputSummary(output: unknown): string {
  const text = cleanText(output)
  return text
    .replace(/^sub-agent\s+(result|failed):\s*/i, '')
    .trim()
}

function finiteDate(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function newestDate(left: number | null, right: number | null): number | null {
  if (left === null) return right
  if (right === null) return left
  return Math.max(left, right)
}
