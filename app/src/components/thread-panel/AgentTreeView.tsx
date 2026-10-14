import clsx from 'clsx'
import type { ThreadAgentNodeStatus, ThreadAgentTreeNode } from '../../lib/thread-agent-tree'

const AGENT_STATUS_DOT: Record<ThreadAgentNodeStatus, string> = {
  idle: 'bg-ink-4',
  queued: 'bg-ink-4',
  running: 'bg-accent',
  succeeded: 'bg-positive',
  failed: 'bg-danger',
  cancelled: 'bg-ink-4',
}
const AGENT_STATUS_TEXT: Record<ThreadAgentNodeStatus, string> = {
  idle: 'text-ink-3',
  queued: 'text-ink-3',
  running: 'text-accent',
  succeeded: 'text-positive',
  failed: 'text-danger',
  cancelled: 'text-ink-3',
}
const AGENT_STATUS_LABEL: Record<ThreadAgentNodeStatus, string> = {
  idle: 'idle',
  queued: 'queued',
  running: 'running',
  succeeded: 'done',
  failed: 'failed',
  cancelled: 'stopped',
}

export function ThreadAgentTreeView({ tree }: { tree: ThreadAgentTreeNode }) {
  return (
    <div className="space-y-1">
      <ThreadAgentNodeRow node={tree} />
    </div>
  )
}

export function ThreadAgentNodeRow({ node }: { node: ThreadAgentTreeNode }) {
  return (
    <div className="min-w-0">
      <div className={clsx('flex min-w-0 items-start gap-2 rounded-lg px-2 py-1.5', node.kind === 'thread' ? 'bg-card-2/70' : 'hover:bg-ink-5')}>
        <span className={clsx('mt-[7px] h-2 w-2 shrink-0 rounded-full', AGENT_STATUS_DOT[node.status], node.status === 'running' && 'animate-pulse')} />
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-center gap-1.5">
            <span title={node.name} className={clsx('min-w-0 truncate text-[13px] font-medium', node.kind === 'thread' ? 'text-ink' : 'text-ink-2')}>{node.name}</span>
            {node.children.length > 0 && <span className="shrink-0 rounded bg-ink-5 px-1.5 text-[12px] tabular-nums text-ink-3">{node.children.length}</span>}
          </div>
          <div title={node.detail} className="mt-0.5 truncate text-[12px] text-ink-4">{node.detail}</div>
        </div>
        <span className={clsx('shrink-0 pt-0.5 text-[12px] font-medium', AGENT_STATUS_TEXT[node.status])}>{AGENT_STATUS_LABEL[node.status]}</span>
      </div>
      {node.children.length > 0 && (
        <div className="ml-3 mt-1 space-y-1 border-l border-line pl-2">
          {node.children.map((child) => <ThreadAgentNodeRow key={child.id} node={child} />)}
        </div>
      )}
    </div>
  )
}
