import { useEffect, useState } from 'react'
import clsx from 'clsx'
import { api } from '../lib/ipc'
import { Icon } from '../lib/icons'
import type { LoopRecord, LoopStatus } from '../lib/types'

const STATUS_TINT: Record<LoopStatus, string> = {
  running: 'text-accent',
  shipped: 'text-positive',
  blocked: 'text-warn',
  capped: 'text-warn',
  failed: 'text-danger',
  stopped: 'text-ink-4',
}

function upsert(list: LoopRecord[], rec: LoopRecord): LoopRecord[] {
  const i = list.findIndex((p) => p.id === rec.id)
  if (i >= 0) { const next = [...list]; next[i] = rec; return next }
  return [rec, ...list]
}

/**
 * Loop Mode launcher + live monitor for one surface (solo thread, room, or
 * kanban). Starts an autonomous self-verifying loop and renders its record as
 * it advances (phase, cycle, the independent critic's verdict, blockers, log).
 */
export function LoopPanel({ cwd, source, surfaceId, projectId }: { cwd: string; source: 'solo' | 'room' | 'ticket'; surfaceId: string; projectId?: string | null }) {
  const [goal, setGoal] = useState('')
  const [mode, setMode] = useState<'build' | 'ship'>('build')
  const [loops, setLoops] = useState<LoopRecord[]>([])
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    api.listLoops().then((l) => { if (alive) setLoops(l.filter((x) => x.surfaceId === surfaceId)) }).catch(() => {})
    const off = api.onLoopUpdate((rec) => {
      if (rec.surfaceId !== surfaceId) return
      setLoops((prev) => upsert(prev, rec))
    })
    return () => { alive = false; off() }
  }, [surfaceId])

  const start = async () => {
    if (!goal.trim() || !cwd || busy) return
    setBusy(true); setErr(null)
    try {
      const rec = await api.startLoop({ source, surfaceId, goal: goal.trim(), mode, baseCwd: cwd, projectId })
      setGoal('')
      setLoops((prev) => upsert(prev, rec))
    } catch (e: any) {
      setErr(e?.message || 'Could not start loop')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-2">
      {!cwd && <div className="px-1 text-[12px] text-ink-4">Pick a folder to run a loop.</div>}
      <textarea
        value={goal}
        onChange={(e) => setGoal(e.target.value)}
        placeholder="Goal — what should the loop build &amp; verify?"
        rows={2}
        className="w-full resize-none rounded-md border border-line bg-card px-2 py-1.5 text-[13px] text-ink placeholder:text-ink-4 focus:border-ink-3 focus:outline-none"
      />
      {err && <div className="rounded-md bg-danger/10 px-2 py-1 text-[12px] leading-snug text-danger">{err}</div>}
      <div className="flex items-center gap-1.5">
        <div className="flex overflow-hidden rounded-md border border-line">
          {(['build', 'ship'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={clsx('px-2 py-1 text-[12px] font-medium capitalize transition-colors', mode === m ? 'bg-ink text-bg' : 'text-ink-3 hover:bg-ink-5')}
            >{m}</button>
          ))}
        </div>
        <button
          onClick={start}
          disabled={!goal.trim() || !cwd || busy}
          title="Start an autonomous self-verifying loop (roles → gates → independent critic → retry)."
          className="flex flex-1 items-center justify-center gap-1 rounded-md bg-ink px-2 py-1.5 text-[13px] font-medium text-bg transition-opacity hover:opacity-90 disabled:opacity-40"
        >
          <Icon name="play" size={11} /> {busy ? 'Starting…' : 'Start loop'}
        </button>
      </div>

      {loops.length > 0 && (
        <div className="scroll-thin max-h-72 space-y-1.5 overflow-y-auto pt-1">
          {loops.map((l) => <LoopCard key={l.id} loop={l} />)}
        </div>
      )}
    </div>
  )
}

function LoopCard({ loop }: { loop: LoopRecord }) {
  const [open, setOpen] = useState(false)
  const running = loop.status === 'running'
  const v = loop.lastVerdict
  return (
    <div className="rounded-lg border border-line bg-card-2 p-2">
      <div className="flex items-center gap-2">
        {running
          ? <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-accent" />
          : <Icon name={loop.status === 'shipped' ? 'check' : loop.status === 'failed' ? 'warning' : 'stop'} size={12} className={clsx('shrink-0', STATUS_TINT[loop.status])} />}
        <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-ink">{loop.goal}</span>
        {running && <button onClick={() => api.stopLoop(loop.id)} className="shrink-0 rounded px-1.5 py-0.5 text-[12px] text-ink-3 transition-colors hover:bg-ink-5 hover:text-danger">Stop</button>}
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 px-0.5 font-mono text-[12px] text-ink-3">
        <span className={clsx('capitalize', STATUS_TINT[loop.status])}>{loop.status}</span>
        <span>·</span>
        <span>{loop.phase ? loop.phase : '—'}</span>
        <span>·</span>
        <span>cycle {loop.cycle}/{loop.maxCycles}</span>
        {loop.checkpoints > 0 && <><span>·</span><span>{loop.checkpoints} cp</span></>}
        <span>·</span>
        <span className="capitalize">{loop.mode}</span>
      </div>
      {v && (
        <div className={clsx('mt-1 rounded-md px-2 py-1 text-[12px] leading-snug', v.passed ? 'bg-positive/10 text-positive' : 'bg-danger/10 text-danger')}>
          {v.passed ? '✓ verified' : '✗ not verified'} ({Math.round(v.confidence * 100)}%) — {v.summary}
        </div>
      )}
      {loop.blockers.length > 0 && (
        <div className="mt-1 space-y-0.5">
          {loop.blockers.map((b, i) => (
            <div key={i} className="rounded-md bg-warn/10 px-2 py-1 text-[12px] leading-snug text-warn"><span className="font-semibold">{b.kind.replace(/_/g, ' ').toLowerCase()}:</span> {b.issue}</div>
          ))}
        </div>
      )}
      <button onClick={() => setOpen((o) => !o)} className="mt-1 flex items-center gap-1 px-0.5 text-[12px] text-ink-4 transition-colors hover:text-ink-3">
        <Icon name="chevronDown" size={10} className={clsx('transition-transform', open && 'rotate-180')} /> {loop.log.length} log line{loop.log.length === 1 ? '' : 's'}
      </button>
      {open && (
        <div className="scroll-thin mt-1 max-h-32 overflow-y-auto whitespace-pre-wrap rounded-md border border-line bg-card px-2 py-1 font-mono text-[11px] leading-snug text-ink-3">
          {loop.log.slice(-40).join('\n') || '(no log yet)'}
        </div>
      )}
    </div>
  )
}
