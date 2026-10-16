import { useEffect, useState } from 'react'
import { useStore } from '../../lib/store'
import { api } from '../../lib/ipc'
import { Icon } from '../../lib/icons'
import { effectiveThreadCwd, effectiveThreadWorkspace } from '../../lib/thread-workspace'
import type { Thread, TurnCheckpoint } from '../../lib/types'
import { PanelSection } from './primitives'

// VST-036 turn checkpoints: revert the thread (files AND conversation) to before a past
// turn, or fork a new thread that diverges from it. Turns are recorded at run start, so
// "turn N" is the state just before that turn's work happened.
export function TurnsSection({ thread, busy, onChanged }: { thread: Thread; busy: boolean; onChanged: () => void }) {
  const forkThreadFromTurn = useStore((s) => s.forkThreadFromTurn)
  const [turns, setTurns] = useState<TurnCheckpoint[]>([])
  const [busyTurn, setBusyTurn] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [note, setNote] = useState<string | null>(null)
  const project = useStore((s) => thread.projectId ? s.projects.find((candidate) => candidate.id === thread.projectId) ?? null : null)
  const workspace = effectiveThreadWorkspace(thread, project)
  const cwd = effectiveThreadCwd(thread, project)
  const isChat = thread.kind === 'chat'
  // Fork worktrees need a local git workspace to diverge from.
  const canForkWorktree = !isChat && !!cwd && workspace?.kind !== 'ssh'

  useEffect(() => {
    let alive = true
    api.checkpointTurns({ threadId: thread.id })
      .then((list) => { if (alive) setTurns(Array.isArray(list) ? list : []) })
      .catch(() => { if (alive) setTurns([]) })
    return () => { alive = false }
  }, [thread.id, thread.updatedAt])

  if (turns.length === 0) return null
  const disabled = busy || busyTurn != null

  const revert = async (turn: number) => {
    if (disabled) return
    const what = isChat ? 'The conversation returns' : 'The workspace files and the conversation both return'
    if (!confirm(`Revert to before turn ${turn}? ${what} to that point. Later turns stay available to fork from.`)) return
    setBusyTurn(turn); setError(null); setNote(null)
    try {
      const r = await api.checkpointRevertTurn({ threadId: thread.id, turn })
      if (!r.ok) { setError(r.error || 'Could not revert to this turn.'); return }
      setNote(r.warning ?? null)
      onChanged()
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not revert to this turn.') }
    finally { setBusyTurn(null) }
  }

  const fork = async (turn: number, worktree = false) => {
    if (disabled) return
    setBusyTurn(turn); setError(null); setNote(null)
    try {
      const r = await forkThreadFromTurn(thread.id, turn, worktree ? { worktree: true } : undefined)
      if (!r.ok) setError(r.error || 'Could not fork from this turn.')
      else if (worktree && r.worktreeError) setNote(`Fork created on the shared folder — worktree unavailable: ${r.worktreeError}`)
    } finally { setBusyTurn(null) }
  }

  return (
    <>
      <div className="mx-4 border-t border-line" />
      <PanelSection title="Turns" right={<span className="rounded bg-ink-5 px-1.5 text-[12px] tabular-nums text-ink-3">{turns.length}</span>}>
        <div className="space-y-0.5">
          {turns.slice(-8).map((cp) => (
            <div key={cp.turn} className="group flex h-7 items-center gap-2 rounded-lg px-1.5 text-[13px] transition-colors hover:bg-ink-5">
              <Icon name="clock" size={13} className="shrink-0 text-ink-3" />
              <span className="min-w-0 flex-1 truncate text-ink/85">Turn {cp.turn}</span>
              <span className="shrink-0 text-[12px] tabular-nums text-ink-4 group-hover:hidden">
                {busyTurn === cp.turn ? 'Working…' : `${cp.messageCount} msg${cp.messageCount === 1 ? '' : 's'}`}
              </span>
              <span className="hidden shrink-0 items-center gap-1 group-hover:flex">
                <button
                  onClick={() => revert(cp.turn)}
                  disabled={disabled}
                  title={`Restore files and conversation to before turn ${cp.turn}.`}
                  className="rounded-md px-1.5 py-0.5 text-[12px] font-medium text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink disabled:opacity-45"
                >
                  Revert
                </button>
                <button
                  onClick={() => fork(cp.turn)}
                  disabled={disabled}
                  title={`Start a new thread that diverges from turn ${cp.turn} — this thread stays as-is.`}
                  className="rounded-md px-1.5 py-0.5 text-[12px] font-medium text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink disabled:opacity-45"
                >
                  Fork
                </button>
                {canForkWorktree && (
                  <button
                    onClick={() => fork(cp.turn, true)}
                    disabled={disabled}
                    title={`Fork from turn ${cp.turn} into its OWN worktree — the fork edits a separate copy of the snapshot, this folder stays untouched.`}
                    className="rounded-md px-1.5 py-0.5 text-[12px] font-medium text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink disabled:opacity-45"
                  >
                    Fork ⎇
                  </button>
                )}
              </span>
            </div>
          ))}
        </div>
        {busy && <p className="mt-1 px-1.5 text-[12px] text-ink-4">Paused while the agent is working.</p>}
        {error && <p className="mt-1 px-1.5 text-[12px] text-danger">{error}</p>}
        {note && <p className="mt-1 px-1.5 text-[12px] text-warn">{note}</p>}
      </PanelSection>
    </>
  )
}
