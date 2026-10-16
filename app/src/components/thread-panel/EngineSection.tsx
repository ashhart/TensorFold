import { useEffect, useState } from 'react'
import clsx from 'clsx'
import { useStore } from '../../lib/store'
import { api } from '../../lib/ipc'
import type { Thread } from '../../lib/types'

// Engine selector (VST-050): run this thread on TensorFold's native loop or on the installed
// omp coding agent via its headless rpc mode. omp threads keep this timeline, plan panel,
// and Guide steering — omp owns the agent loop itself.

export function EngineSection({ thread }: { thread: Thread }) {
  const setThreadEngine = useStore((s) => s.setThreadEngine)
  const [omp, setOmp] = useState<{ available: boolean; version?: string } | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    api.ompStatus().then((status) => { if (alive) setOmp({ available: status.available, version: status.version }) }).catch(() => { if (alive) setOmp({ available: false }) })
    return () => { alive = false }
  }, [])

  const engine = thread.engine === 'omp' ? 'omp' : 'native'
  const choose = async (next: 'native' | 'omp') => {
    if (next === engine || busy) return
    setBusy(true)
    setError(null)
    try {
      await setThreadEngine(thread.id, next)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const option = (value: 'native' | 'omp', label: string, hint: string, enabled: boolean) => (
    <button
      type="button"
      disabled={!enabled || busy}
      onClick={() => void choose(value)}
      className={clsx(
        'flex-1 rounded-lg border px-2.5 py-2 text-left transition-colors',
        engine === value ? 'border-ink bg-ink text-canvas' : 'border-line bg-card text-ink-2 hover:border-ink-4',
        !enabled && 'cursor-not-allowed opacity-50',
      )}
    >
      <span className="block text-[13px] font-semibold">{label}</span>
      <span className={clsx('mt-0.5 block text-[12px] leading-tight', engine === value ? 'text-canvas/70' : 'text-ink-4')}>{hint}</span>
    </button>
  )

  return (
    <div className="space-y-2 px-4 py-3">
      <div className="flex items-center justify-between">
        <span className="text-[12px] font-semibold uppercase tracking-wider text-ink-3">Engine</span>
        {omp?.available && omp.version ? <span className="font-mono text-[11px] text-ink-4">{omp.version}</span> : null}
      </div>
      <div className="flex gap-2">
        {option('native', 'Native loop', 'TensorFold tools, approvals, guards', true)}
        {option('omp', 'OMP agent', omp === null ? 'checking…' : omp.available ? 'omp runs the loop; TensorFold gates writes' : 'omp CLI not installed', omp?.available === true)}
      </div>
      {error ? <div className="text-[12px] text-danger">{error}</div> : null}
    </div>
  )
}
