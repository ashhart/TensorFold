import { useEffect, useMemo, useRef, useState } from 'react'
import { threadPendingApproval } from '../lib/thread-approval'
import { useStore } from '../lib/store'
import { api } from '../lib/ipc'
import clsx from 'clsx'
import { Composer } from './Composer'
import { AssistantMessage, CueBubble, ErrorBlock, ToolCard, UserBubble, VisualizationCard } from './timeline'
import { ThreadPanel } from './ThreadPanel'
import { RunPlanCard } from './thread-panel/RunPlanCard'
import { ApprovalCard } from './ApprovalCard'
import { Icon } from '../lib/icons'
import { buildTurnRows, type TurnSummary } from '../lib/turn-blocks'
import type { GitStatus, TimelineItem } from '../lib/types'
import { isPinned, parseAnnotations, pinMessage, serializeAnnotations, unpinMessage } from '../lib/thread-annotations'
import { effectiveThreadCwd, effectiveThreadWorkspace } from '../lib/thread-workspace'
import { TurnWorkRow } from './turn/TurnWorkRow'
import { EditedFilesCard } from './turn/EditedFilesCard'
import { TurnStatsLine } from './turn/TurnStatsLine'

const BUSY_STATES = ['thinking', 'streaming', 'tool', 'awaiting']

export function ThreadView({ threadId }: { threadId: string }) {
  const thread = useStore((s) => s.threads.find((t) => t.id === threadId))
  const st = useStore((s) => s.status[threadId])
  const approvals = useStore((s) => s.approvals[threadId])
  const durableJobs = useStore((s) => s.jobs)
  const approve = useStore((s) => s.approve)
  const collapseToolRunsDefault = useStore((s) => s.collapseToolRunsDefault)
  const dockRequest = useStore((s) => s.threadDockRequest)
  const project = useStore((s) => thread?.projectId ? s.projects.find((candidate) => candidate.id === thread.projectId) ?? null : null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const endRef = useRef<HTMLDivElement>(null)
  const followRef = useRef(true)

  const workspace = thread ? effectiveThreadWorkspace(thread, project) : null
  const workspaceCwd = thread ? effectiveThreadCwd(thread, project) : ''
  const isTask = thread?.kind === 'task' && !!workspaceCwd
  const [git, setGit] = useState<GitStatus | null>(null)
  const [gitTick, setGitTick] = useState(0)
  const refreshGit = () => setGitTick((n) => n + 1)
  // ⌘J and dock requests (command palette, run links) open the work pane's tabs.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'j') { e.preventDefault(); useStore.getState().openWorkPane('terminal') }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
  useEffect(() => {
    if (!dockRequest || dockRequest.threadId !== threadId || !isTask) return
    useStore.getState().openWorkPane(dockRequest.dock)
  }, [dockRequest?.nonce, dockRequest?.threadId, dockRequest?.dock, isTask, threadId])

  const timeline = thread?.timeline ?? []
  const busy = !!st && BUSY_STATES.includes(st)
  const lastAssistantId = useMemo(() => [...timeline].reverse().find((i) => i.kind === 'assistant')?.id, [timeline])
  // Pinned-message annotations (VST-040): parsed from the thread blob, toggled per message.
  const patchThread = useStore((s) => s.patchThread)
  const annotations = useMemo(() => parseAnnotations(thread?.annotations), [thread?.annotations])
  // One turn, one block: the user message, one "Worked for" row, then the answer and its edits.
  const turnRows = useMemo(() => buildTurnRows(timeline, { running: busy }), [timeline, busy])
  const userTurns = useMemo(() => timeline.filter((item) => item.kind === 'user').length, [timeline])
  // Turns that have a checkpoint can be undone from their edits card.
  const [checkpointTurns, setCheckpointTurns] = useState<Set<number>>(() => new Set())
  useEffect(() => {
    if (!isTask) { setCheckpointTurns(new Set()); return }
    let alive = true
    api.checkpointTurns({ threadId })
      .then((list) => { if (alive) setCheckpointTurns(new Set((Array.isArray(list) ? list : []).map((cp) => cp.turn))) })
      .catch(() => { if (alive) setCheckpointTurns(new Set()) })
    return () => { alive = false }
  }, [threadId, isTask, busy, userTurns])
  const pendingApproval = useMemo(() => {
    return threadPendingApproval(threadId, timeline, approvals ?? [], durableJobs)
  }, [threadId, approvals, timeline, durableJobs])

  // Stick to bottom while streaming unless the user scrolled up.
  const signature = timeline.map((i) => i.id + (i.text?.length ?? 0) + (i.output?.length ?? 0) + (i.state ?? '')).join('|')
  useEffect(() => {
    if (followRef.current) endRef.current?.scrollIntoView({ block: 'end' })
  }, [signature])
  useEffect(() => {
    if (pendingApproval) endRef.current?.scrollIntoView({ block: 'end' })
  }, [pendingApproval?.callId])

  // Pull working-tree git state for the panel + the files-changed card.
  useEffect(() => {
    if (!isTask || !thread) { setGit(null); return }
    let alive = true
    api.gitStatus(workspaceCwd, workspace).then((g) => { if (alive) setGit(g) }).catch(() => { if (alive) setGit(null) })
    return () => { alive = false }
  }, [isTask, workspaceCwd, workspace, thread?.updatedAt, gitTick])

  // Refresh git the moment a run finishes so the running pill / files-changed
  // tally reflects the final edits rather than a stale mid-run snapshot (#14).
  const prevBusyRef = useRef(false)
  useEffect(() => {
    if (prevBusyRef.current && !busy && isTask) refreshGit()
    prevBusyRef.current = busy
  }, [busy, isTask])

  const onScroll = () => {
    const el = scrollRef.current
    if (!el) return
    followRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120
  }

  if (!thread) return null
  const renderTimelineItem = (item: TimelineItem, turn: TurnSummary) => {
    if (item.kind === 'user') return <UserBubble text={item.text || ''} attachments={item.attachments} />
    if (item.kind === 'cue') return <CueBubble text={item.text || ''} state={item.state} />
    if (item.kind === 'assistant') return (
      <AssistantMessage
        text={item.text || ''}
        meta={item.meta}
        active={st === 'streaming' && item.id === lastAssistantId}
        footer={item.stats && item.id === turn.answerId && !turn.running ? <TurnStatsLine stats={item.stats} /> : undefined}
      />
    )
    if (item.kind === 'tool') return <ToolCard item={item} />
    if (item.kind === 'visualization') return <VisualizationCard item={item} />
    if (item.kind === 'error') return <ErrorBlock text={item.text || ''} />
    if (item.kind === 'notice') return (
      <div className="flex items-center justify-center py-1 fade-in">
        <span className="flex items-center gap-1.5 rounded-full border border-line bg-card-2/60 px-2.5 py-1 text-[13px] text-ink-3"><Icon name="layers" size={12} className="text-ink-3" />{item.text}</span>
      </div>
    )
    return null
  }

  // Pin affordance (VST-040): user/assistant messages get a hover pin toggle so key turns
  // surface in the right panel's Pinned section. Persisted in thread.annotations.
  const renderPinnable = (item: TimelineItem, turn: TurnSummary) => {
    const rendered = renderTimelineItem(item, turn)
    if (!rendered || (item.kind !== 'user' && item.kind !== 'assistant')) return rendered
    const pinned = isPinned(annotations, item.id)
    const toggle = () => {
      const next = pinned ? unpinMessage(annotations, item.id) : pinMessage(annotations, item.id, null, Date.now(), () => `pin-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`)
      void patchThread(thread.id, { annotations: serializeAnnotations(next) })
    }
    return (
      <div className="group/pin relative">
        {rendered}
        <button
          onClick={toggle}
          title={pinned ? 'Unpin this message' : 'Pin this message'}
          className={clsx('absolute -left-6 top-1 flex h-5 w-5 items-center justify-center rounded-md transition-opacity', pinned ? 'text-accent opacity-100' : 'text-ink-4 opacity-0 hover:text-ink group-hover/pin:opacity-100')}
        >
          <Icon name={pinned ? 'starFilled' : 'star'} size={13} />
        </button>
      </div>
    )
  }

  return (
    <div className="relative flex h-full">
      <div className="flex h-full min-w-0 flex-1 flex-col">
        <div ref={scrollRef} onScroll={onScroll} className="scroll-thin min-h-0 flex-1 overflow-y-auto select-text">
          <div className="mx-auto w-full max-w-3xl space-y-4 px-6 pb-8 pt-12">
            {turnRows.map((row) => {
              if (row.type === 'work') return (
                <div key={`work-${row.turn.key}`}>
                  <TurnWorkRow turn={row.turn} status={row.turn.running ? st : undefined} defaultOpen={!collapseToolRunsDefault} />
                </div>
              )
              if (row.type === 'edits') return (
                <div key={`edits-${row.turn.key}`}>
                  <EditedFilesCard thread={thread} turn={row.turn} canUndo={isTask && !busy && checkpointTurns.has(row.turn.index)} onChanged={refreshGit} />
                </div>
              )
              return <div key={row.item.id} id={`ti-${row.item.id}`}>{renderPinnable(row.item, row.turn)}</div>
            })}

            <RunPlanCard thread={thread} busy={!!busy} />

            <div ref={endRef} className="h-px" />
          </div>
        </div>

        {/* Docked composer — no scroll scrim: a gradient fading to the opaque
            --canvas paints a visible band over the glass theme's translucent
            canvas. The composer is a distinct floating card, so no fade needed. */}
        <div className="relative shrink-0 px-6 pb-5">
          <div className="mx-auto w-full max-w-3xl">
            {/* Live plan (Goal Mode): a compact progress strip in the transcript so
                the thinking→acting→done cadence is visible without opening the right
                panel. Mirrors Codex's inline plan tracking. */}
            {isTask && thread.goal && thread.goalStatus === 'active' && (thread.plan?.length ?? 0) > 0 && (
              <div className="mb-2 fade-in">
                <div className="inline-flex max-w-full items-center gap-2 rounded-full border border-line bg-card-2/70 px-3 py-1 text-[13px] text-ink-2">
                  <Icon name="grid" size={12} className="shrink-0 text-ink-3" />
                  <span className="shrink-0 font-medium text-ink-3">{(thread.plan ?? []).filter((p) => p.done).length}/{thread.plan!.length} steps</span>
                  <span className="min-w-0 truncate text-ink-3">{(thread.plan ?? []).find((p) => !p.done)?.text ?? 'wrapping up'}</span>
                </div>
              </div>
            )}
            {pendingApproval && (
              <ApprovalCard
                key={pendingApproval.callId}
                title={pendingApproval.item.title}
                subject={pendingApproval.item.command || pendingApproval.item.path || pendingApproval.item.tool || ''}
                mono={!!pendingApproval.item.command}
                reason={pendingApproval.item.output}
                canAllowScoped={Boolean(thread.projectId || thread.cwd)}
                onDeny={() => approve(threadId, pendingApproval.callId, 'deny')}
                onAllowScoped={() => approve(threadId, pendingApproval.callId, 'allow_always')}
                onAllow={() => approve(threadId, pendingApproval.callId, 'allow')}
              />
            )}
            <Composer mode="thread" thread={thread} />
          </div>
        </div>
      </div>

      {/* The work pane decides for itself whether it is open (see lib/work-pane.ts). */}
      <ThreadPanel thread={thread} git={git} onChanged={refreshGit} />
    </div>
  )
}
