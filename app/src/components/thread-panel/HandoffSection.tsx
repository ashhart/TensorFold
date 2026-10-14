import { useState } from 'react'
import { useStore } from '../../lib/store'
import type { Thread } from '../../lib/types'
import { PanelSection } from './primitives'

// VST-036 cross-engine handoff: continue this conversation on a DIFFERENT engine. The host
// flattens the transcript into a seed message and runs it as the new thread's first turn.
export function HandoffSection({ thread, busy }: { thread: Thread; busy: boolean }) {
  const settings = useStore((s) => s.settings)
  const handoffThread = useStore((s) => s.handoffThread)
  const targets = settings?.providers?.filter((p) => p.id !== thread.providerId) ?? []
  const [toId, setToId] = useState('')
  const [working, setWorking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  if (targets.length === 0) return null
  const selected = targets.find((p) => p.id === toId) ?? null

  const go = async () => {
    if (!selected || working || busy) return
    setWorking(true); setError(null)
    try {
      const r = await handoffThread(thread.id, selected.id)
      if (!r.ok) setError(r.error || 'Handoff failed.')
    } finally { setWorking(false) }
  }

  return (
    <>
      <div className="mx-4 border-t border-line" />
      <PanelSection title="Hand off">
        <p className="mb-1.5 px-1 text-[13px] leading-snug text-ink-4">Continue this conversation on a different engine. The transcript seeds its first turn; this thread stays as-is.</p>
        <div className="flex items-center gap-1.5">
          <select
            value={toId}
            onChange={(e) => setToId(e.target.value)}
            className="h-7 min-w-0 flex-1 rounded-md border border-line bg-card px-1.5 text-[13px] text-ink focus:border-ink-3 focus:outline-none"
          >
            <option value="">Choose an engine…</option>
            {targets.map((p) => <option key={p.id} value={p.id}>{p.name || p.id}</option>)}
          </select>
          <button
            onClick={go}
            disabled={!selected || working || busy}
            title={busy ? 'Paused while the agent is working.' : selected ? `Hand off to ${selected.name || selected.id} and run the seed turn now.` : 'Choose a target engine first.'}
            className="h-7 shrink-0 rounded-md bg-ink px-2.5 text-[13px] font-medium text-bg transition-opacity hover:opacity-90 disabled:opacity-40"
          >
            {working ? 'Handing off…' : 'Hand off'}
          </button>
        </div>
        {error && <p className="mt-1 px-1 text-[12px] text-danger">{error}</p>}
      </PanelSection>
    </>
  )
}
