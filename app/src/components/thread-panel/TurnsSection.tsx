import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useRef, useState } from 'react'
import { useStore } from '../../lib/store'
import { BUSY_RUN } from '../../lib/store/types'
import { api } from '../../lib/ipc'
import { Icon } from '../../lib/icons'
import { effectiveThreadCwd, effectiveThreadWorkspace } from '../../lib/thread-workspace'
import type { Thread, TurnCheckpoint } from '../../lib/types'
import { PanelSection } from './primitives'

const actionClass = 'min-h-8 rounded-md px-2 py-1 text-[12px] font-medium text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-45'

// Checkpoints are captured before a turn starts. Branch navigation opens the
// existing thread without altering either transcript or any approval decision.
export function TurnsSection({ thread, busy, onChanged }: { thread: Thread; busy: boolean; onChanged: () => void }) {
  const tx = useAuthoredText()
  const forkThreadFromTurn = useStore((s) => s.forkThreadFromTurn)
  const threads = useStore((s) => s.threads)
  const openThread = useStore((s) => s.openThread)
  const [saved, setSaved] = useState<{ threadId: string; turns: TurnCheckpoint[] }>({ threadId: thread.id, turns: [] })
  const [loading, setLoading] = useState(true)
  const [busyTurn, setBusyTurn] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [note, setNote] = useState<string | null>(null)
  const pending = useRef(false)
  const current = useRef({ thread, busy, generation: 0 })
  const generation = current.current.generation + (current.current.thread.id === thread.id ? 0 : 1)
  current.current = { thread, busy, generation }
  const project = useStore((s) => thread.projectId ? s.projects.find((candidate) => candidate.id === thread.projectId) ?? null : null)
  const workspace = effectiveThreadWorkspace(thread, project)
  const cwd = effectiveThreadCwd(thread, project)
  const isChat = thread.kind === 'chat'
  const canForkWorktree = !isChat && !thread.bot && !!cwd && workspace?.kind !== 'ssh'
  const turns = saved.threadId === thread.id ? saved.turns : []
  const related = threads.filter((candidate) => candidate.id === thread.id || candidate.id === thread.parentId || candidate.parentId === thread.id || !!thread.parentId && candidate.parentId === thread.parentId)
  const disabled = busy || loading || busyTurn != null

  useEffect(() => { pending.current = false; setBusyTurn(null) }, [thread.id])

  useEffect(() => {
    let alive = true
    setError(null); setNote(null); setLoading(true)
    api.checkpointTurns({ threadId: thread.id })
      .then((list) => { if (alive) setSaved({ threadId: thread.id, turns: Array.isArray(list) ? list : [] }) })
      .catch((reason) => { if (alive) { setSaved({ threadId: thread.id, turns: [] }); setError(reason instanceof Error ? reason.message : tx("Saved turns could not be loaded.")) } })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [thread.id, thread.updatedAt])

  const sameThread = () => current.current.thread.id === thread.id && current.current.generation === generation
  const mayChange = (cp: TurnCheckpoint) => {
    const state = useStore.getState()
    const live = state.threads.find((candidate) => candidate.id === thread.id)
    return !loading && saved.threadId === thread.id && sameThread() && current.current.thread.updatedAt === thread.updatedAt && !current.current.busy && !pending.current
      && !BUSY_RUN.has(state.status[thread.id]) && !state.approvals[thread.id]?.length
      && !!live && live.updatedAt === thread.updatedAt && live.status !== 'running' && turns.some((savedTurn) => savedTurn.turn === cp.turn && savedTurn.fingerprint === cp.fingerprint)
  }
  const refuseStale = () => { if (sameThread()) setError(tx("This conversation changed or is waiting for the agent. Choose the saved turn again when it is idle.")) }
  const explanation = (cp: TurnCheckpoint) => isChat
    ? 'Restores the conversation to this snapshot. No workspace files are changed.'
    : cp.fileRef
      ? 'Restores the conversation and attempts to restore agent-owned file changes. Newer user edits are preserved. If the file snapshot is unavailable, only the conversation is restored and a warning is shown.'
      : 'Restores the conversation only. This turn has no file snapshot, so workspace files are unchanged.'

  const revert = async (cp: TurnCheckpoint) => {
    if (!mayChange(cp)) { refuseStale(); return }
    if (!confirm(tx("Revert to before turn {value0}? {value1} Later saved turns stay available to fork from.", { value0: cp.turn, value1: explanation(cp) }))) return
    if (!mayChange(cp)) { refuseStale(); return }
    pending.current = true; setBusyTurn(cp.turn); setError(null); setNote(null)
    try {
      const result = await api.checkpointRevertTurn({ threadId: thread.id, turn: cp.turn })
      if (!sameThread()) return
      if (!result.ok) { setError(result.error || tx("Could not revert to this turn.")); return }
      setNote(result.warning || `Conversation restored to before turn ${cp.turn}.`)
      onChanged()
    } catch (reason) { if (sameThread()) setError(reason instanceof Error ? reason.message : tx("Could not revert to this turn.")) }
    finally { if (sameThread()) { pending.current = false; setBusyTurn(null) } }
  }

  const fork = async (cp: TurnCheckpoint, worktree = false) => {
    if (!mayChange(cp)) { refuseStale(); return }
    pending.current = true; setBusyTurn(cp.turn); setError(null); setNote(null)
    try {
      const result = await forkThreadFromTurn(thread.id, cp.turn, worktree ? { worktree: true } : undefined)
      if (!sameThread()) return
      if (!result.ok) setError(result.error || tx("Could not fork from this turn."))
      else if (worktree && result.worktreeError) setNote(`Fork created in the shared folder. Worktree unavailable: ${result.worktreeError}`)
    } catch (reason) { if (sameThread()) setError(reason instanceof Error ? reason.message : tx("Could not fork from this turn.")) }
    finally { if (sameThread()) { pending.current = false; setBusyTurn(null) } }
  }

  return (
    <>
      <div className="mx-4 border-t border-line" />
      <PanelSection title={tx("Branches and turns")} right={<span className="rounded bg-ink-5 px-1.5 text-[12px] tabular-nums text-ink-3">{turns.length}</span>}>
        {related.length > 1 && <nav aria-label={tx("Conversation branches")} className="mb-2 space-y-1">
          {related.map((branch) => <button key={branch.id} type="button" aria-current={branch.id === thread.id ? 'page' : undefined} disabled={branch.id === thread.id} onClick={() => openThread(branch.id)} className="flex min-h-8 w-full items-center gap-2 rounded-md px-2 py-1 text-left text-[12px] text-ink-2 hover:bg-ink-5 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent disabled:text-ink">
            <span className="min-w-0 flex-1 break-words">{branch.title || tx("Untitled conversation")}</span>
            <span className="shrink-0 text-ink-3">{branch.id === thread.id ? tx("Current") : branch.id === thread.parentId ? tx("Source") : tx("Branch")}</span>
          </button>)}
        </nav>}
        <p className="mb-2 px-1.5 text-[12px] text-ink-3">{tx("Saved snapshots before each turn. Fork keeps the original conversation.")}</p>
        {loading && <p role="status" className="px-1.5 text-[12px] text-ink-3">{tx("Loading saved turns…")}</p>}
        {!loading && !turns.length && !error && <p className="px-1.5 text-[12px] text-ink-3">{tx("No saved turns yet.")}</p>}
        <div className="max-h-80 space-y-1 overflow-y-auto scroll-thin" aria-label={tx("Saved turns")} aria-busy={loading}>
          {turns.map((cp) => (
            <div key={cp.turn} className="rounded-lg border border-line px-1.5 py-1.5 text-[13px]">
              <div className="flex items-start gap-2 px-0.5">
                <Icon name="clock" size={13} className="mt-0.5 shrink-0 text-ink-3" />
                <div className="min-w-0 flex-1">
                  <p className="text-ink/85">{tx("Turn ")}{cp.turn}{cp.label ? ` · ${cp.label}` : ''}</p>
                  <p className="text-[12px] text-ink-3">{cp.messageCount} {cp.messageCount === 1 ? tx("message") : tx("messages")} · {isChat || !cp.fileRef ? tx("Conversation snapshot") : tx("Conversation and file snapshot")}</p>
                </div>
                {busyTurn === cp.turn && <span role="status" className="text-[12px] text-ink-3">{tx("Working…")}</span>}
              </div>
              <div className="mt-1 flex flex-wrap items-center gap-1">
                <button type="button" onClick={() => revert(cp)} disabled={disabled} aria-label={tx("Revert to before turn {value0}", { value0: cp.turn })} title={explanation(cp)} className={actionClass}>{tx("Revert")}</button>
                <button type="button" onClick={() => fork(cp)} disabled={disabled} aria-label={tx("Fork conversation before turn {value0}", { value0: cp.turn })} title={tx("Creates an independent conversation from this snapshot. Task forks share the folder unless you choose a worktree.")} className={actionClass}>{tx("Fork")}</button>
                {canForkWorktree && cp.fileRef && <button type="button" onClick={() => fork(cp, true)} disabled={disabled} aria-label={tx("Fork before turn {value0} in a separate worktree", { value0: cp.turn })} title={tx("Attempts to create a separate folder from this file snapshot. A failure creates the fork in the shared folder with a warning.")} className={actionClass}>{tx("Fork worktree")}</button>}
              </div>
            </div>
          ))}
        </div>
        {busy && <p className="mt-1 px-1.5 text-[12px] text-ink-4">{tx("Turn actions are paused while the agent is working or waiting for approval.")}</p>}
        {error && <p role="alert" className="mt-1 px-1.5 text-[12px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
        {note && <p role="status" className="mt-1 px-1.5 text-[12px] text-warn">{note}</p>}
      </PanelSection>
    </>
  )
}
