import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useState } from 'react'
import { useStore } from '../../lib/store'
import { Icon } from '../../lib/icons'
import type { Thread } from '../../lib/types'
import { isBuiltInProvider, isExternalProvider, pickPlanRunner } from '../../../shared/provider-trust'

// After an external (planning-only) model replies in a Code thread, offer to carry the plan
// out on the built-in engine, or another local provider until that exists. Reuses the
// cross-engine handoff (VST-049) with the 'run-plan' intent.
export function RunPlanCard({ thread, busy }: { thread: Thread; busy: boolean }) {
  const tx = useAuthoredText()
  const settings = useStore((s) => s.settings)
  const handoffThread = useStore((s) => s.handoffThread)
  const [working, setWorking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const providers = settings?.providers ?? []
  const provider = providers.find((p) => p.id === (thread.providerId ?? settings?.activeProviderId)) ?? null
  const last = [...(thread.timeline ?? [])].reverse().find((item) => item.kind === 'user' || item.kind === 'assistant')
  const runner = pickPlanRunner(providers)
  if (busy || thread.kind !== 'task' || !provider || !isExternalProvider(provider) || last?.kind !== 'assistant' || !last.text?.trim()) return null

  const label = runner && isBuiltInProvider(runner) ? 'the TensorFold engine' : runner?.name || 'a local engine'
  const run = async () => {
    if (!runner || working) return
    setWorking(true); setError(null)
    try {
      const r = await handoffThread(thread.id, runner.id, null, 'run-plan')
      if (!r.ok) setError(r.error || tx("Could not start the plan."))
    } finally { setWorking(false) }
  }

  return (
    <div className="flex flex-wrap items-center gap-2 rounded-xl border border-line bg-card-2 px-3 py-2 text-[13px] text-ink-2 fade-in">
      <Icon name="shield" size={14} className="shrink-0 text-ink-3" />
      <span className="min-w-0 flex-1">
        {runner ? tx("{value0} only planned. Run the plan on this Mac with {value1}.", { value0: provider.name || tx("This provider"), value1: label }) : tx("Add the TensorFold engine or a local provider in Settings to run this plan on this Mac.")}
      </span>
      {runner && (
        <button
          onClick={run}
          disabled={working}
          className="shrink-0 rounded-lg bg-ink px-2.5 py-1 text-[13px] font-medium text-bg transition-opacity hover:opacity-90 disabled:opacity-40"
        >
          {working ? tx("Starting…") : tx("Run this plan with {value0}", { value0: label })}
        </button>
      )}
      {error && <p className="w-full text-[12px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
    </div>
  )
}
