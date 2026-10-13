import { useMemo } from 'react'
import clsx from 'clsx'
import { useStore } from '../../lib/store'
import { Icon } from '../../lib/icons'
import { formatDuration } from '../../lib/format'
import { buildAgenticRunContract } from '../../lib/agentic-run-contract'
import { workbenchPrReviewState } from '../../lib/workbench-verdict'
import { buildThreadAgentTree, countThreadAgentNodes } from '../../lib/thread-agent-tree'
import { ContextPreviewSection } from '../ContextPreview'
import type { GitStatus, Thread } from '../../lib/types'
import { effectiveThreadCwd, effectiveThreadWorkspace } from '../../lib/thread-workspace'
import { workingLine, deriveActivity, FileRow, estTokens, kfmt, labelize, PanelSection } from './primitives'
import { RunContractSummary, SESSION_TINT } from './RunContractSummary'
import { WorkbenchDoneGate } from './WorkbenchDoneGate'
import { ThreadAgentTreeView } from './AgentTreeView'
import { EnvironmentSection } from './EnvironmentSection'
import { EngineSection } from './EngineSection'
import { TurnsSection } from './TurnsSection'
import { ArtifactsSection } from './ArtifactsSection'
import { HandoffSection } from './HandoffSection'
import { RecapSection } from './RecapSection'
import { PinsSection } from './PinsSection'
import { NotepadSection } from './NotepadSection'
import { GitToolsSection } from './GitToolsSection'
import { DoomPlan } from './DoomPlan'

// The work pane's Details tab: everything the old right panel showed, moved here unchanged.
export function ThreadDetails({ thread, git, onChanged }: { thread: Thread; git: GitStatus | null; onChanged: () => void }) {
  const settings = useStore((s) => s.settings)
  const status = useStore((s) => s.status[thread.id])
  const approvals = useStore((s) => s.approvals[thread.id])
  const jobs = useStore((s) => s.jobs)
  const project = useStore((s) => thread.projectId ? s.projects.find((candidate) => candidate.id === thread.projectId) ?? null : null)
  const workspace = effectiveThreadWorkspace(thread, project)
  const cwd = effectiveThreadCwd(thread, project)
  const busy = !!status && ['thinking', 'streaming', 'tool', 'awaiting'].includes(status)
  const { outputs, sources, commands } = deriveActivity(thread)
  // Chats get the slim panel (VST-036): Environment, Activity, contract, Turns (revert is
  // conversation-only for them), Hand off, and Notepad — no git/file sections.
  const isTask = thread.kind === 'task' && !!cwd
  const plan = thread.plan ?? []
  const done = plan.filter((p) => p.done).length
  const tokens = estTokens(thread)
  const duration = formatDuration(Math.max(0, thread.updatedAt - thread.createdAt))
  const goalStatus = thread.goalStatus ?? 'active'
  const agentTree = buildThreadAgentTree(thread)
  const subAgentCount = countThreadAgentNodes(agentTree)
  const runContract = buildAgenticRunContract({ thread, git, status, settings, pendingApprovals: approvals?.length ?? 0 })
  const prReviewState = useMemo(() => workbenchPrReviewState(thread.id, jobs), [thread.id, jobs])
  // Commit + hygiene run against the local repo; gate them for remote (SSH) workspaces
  // (the git status/diff above are remote-aware). Remote commits go through the agent.
  const remote = workspace?.kind === 'ssh'

  return (
    <div className="pb-6">
      <EnvironmentSection thread={thread} />

      <div className="mx-3.5 border-t border-line" />
      <EngineSection thread={thread} />

      <div className="mx-3.5 border-t border-line" />
      <PanelSection title="Activity">
        <div className="flex items-center gap-2.5 rounded-lg border border-line bg-card-2 px-2.5 py-2 shadow-card">
          {busy
            ? <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-accent" />
            : <Icon name="check" size={14} className="shrink-0 text-positive" />}
          <span className="min-w-0 flex-1 truncate text-[13px] text-ink">{workingLine(status, thread)}</span>
        </div>
        {commands > 0 && (
          <div className="mt-1.5 flex items-center gap-1.5 px-1 text-[13px] text-ink-3"><Icon name="terminal" size={12} />Ran {commands} command{commands === 1 ? '' : 's'}</div>
        )}
      </PanelSection>

      <div className="mx-4 border-t border-line" />
      <PanelSection title="Run contract" right={<span className={clsx('text-[12px] font-medium capitalize', SESSION_TINT[runContract.session.state])}>{labelize(runContract.session.state)}</span>}>
        <RunContractSummary contract={runContract} />
        <WorkbenchDoneGate thread={thread} contract={runContract} reviewState={prReviewState} />
      </PanelSection>

      {subAgentCount > 0 && (
        <>
          <div className="mx-4 border-t border-line" />
          <PanelSection title="Sub-agents" right={<span className="rounded bg-ink-5 px-1.5 text-[12px] tabular-nums text-ink-3">{subAgentCount}</span>}>
            <ThreadAgentTreeView tree={agentTree} />
          </PanelSection>
        </>
      )}

      {isTask && (
        <>
          <div className="mx-4 border-t border-line" />
          <PanelSection title="Outputs" right={outputs.length > 0 ? <span className="rounded bg-ink-5 px-1.5 text-[12px] tabular-nums text-ink-3">{outputs.length}</span> : undefined}>
            {outputs.length === 0
              ? <div className="px-1 text-[13px] text-ink-4">Nothing created or edited yet.</div>
              : <div className="space-y-0.5">{outputs.map((o) => <FileRow key={o.path} path={o.path} tag={o.verb} tagClass={o.verb === 'Created' ? 'text-positive' : 'text-warn'} />)}</div>}
          </PanelSection>

          <div className="mx-4 border-t border-line" />
          <PanelSection title="Sources" right={sources.length > 0 ? <span className="rounded bg-ink-5 px-1.5 text-[12px] tabular-nums text-ink-3">{sources.length}</span> : undefined}>
            {sources.length === 0
              ? <div className="px-1 text-[13px] text-ink-4">No files read yet.</div>
              : <div className="space-y-0.5">{sources.map((s) => <FileRow key={s} path={s} tag="Read" tagClass="text-ink-4" />)}</div>}
          </PanelSection>
        </>
      )}

      <div className="mx-4 border-t border-line" />
      <ContextPreviewSection thread={thread} />

      {isTask && (
        <GitToolsSection thread={thread} cwd={cwd} workspace={workspace} remote={remote} git={git} onChanged={onChanged} />
      )}

      <ArtifactsSection thread={thread} />

      <TurnsSection thread={thread} busy={busy} onChanged={onChanged} />

      <HandoffSection thread={thread} busy={busy} />

      {/* Loop Mode is a rooms / multi-agent concept (rescope #56) — a solo task
          just runs, so no loop launcher is shown here. */}

      {thread.goal && (
        <>
          <div className="mx-4 border-t border-line" />
          <PanelSection title="Goal" right={<span className="text-[12px] font-medium capitalize text-ink-3">{goalStatus}</span>}>
            <div className="flex gap-2.5 px-1">
              <Icon name="grid" size={15} className="mt-0.5 shrink-0 text-ink-3" />
              <div className="min-w-0">
                <div className="text-[13px] font-semibold leading-snug text-ink">{thread.goal}</div>
                    <div className="mt-1 flex items-center gap-2 font-mono text-[12px] text-ink-3">
                      {plan.length > 0 && <span>{done}/{plan.length}</span>}
                      {plan.length > 0 && <span>·</span>}
                      <span>{duration}</span>
                      <span>·</span>
                      <span>{kfmt(tokens)} tokens</span>
                    </div>
                    {plan.length > 0 && settings?.doomTodos !== false && (
                      <div className="mt-1.5 font-mono text-[11px] leading-none tracking-tighter">
                        <span className="text-ink">{'█'.repeat(Math.round((done / plan.length) * 14))}</span>
                        <span className="text-ink-4">{'░'.repeat(14 - Math.round((done / plan.length) * 14))}</span>
                      </div>
                    )}
              </div>
            </div>
          </PanelSection>
        </>
      )}

      {plan.length > 0 && (
        <>
          <div className="mx-4 border-t border-line" />
          <PanelSection title="Progress" right={<span className="font-mono text-[12px] font-medium text-ink-3">{done}/{plan.length}</span>}>
            {settings?.doomTodos === false ? (
              <div className="space-y-3">
                {plan.map((step) => (
                  <div key={step.id} className="flex items-start gap-2.5">
                    <span className={clsx('mt-px flex h-4 w-4 shrink-0 items-center justify-center rounded-full', step.done ? 'text-positive' : 'text-ink-4')}>
                      {step.done ? <Icon name="check" size={13} /> : <span className="h-3 w-3 rounded-full border border-current" />}
                    </span>
                    <span className={clsx('text-[13px] leading-snug', step.done ? 'text-ink-4 line-through' : 'text-ink-2')}>{step.text}</span>
                  </div>
                ))}
              </div>
            ) : (
              <DoomPlan plan={plan} />
            )}
          </PanelSection>
        </>
      )}

      <div className="mx-3.5 border-t border-line" />
      <RecapSection thread={thread} />
      <PinsSection thread={thread} />
      <NotepadSection thread={thread} />
    </div>
  )
}
