import { useState } from 'react'
import { api } from '../../lib/ipc'
import { MarkdownLink } from '../../lib/markdown-link'
import type { EngineStatus } from '../../lib/types'
import { Toggle } from '../ui'

const PRIVACY_DOC = 'https://github.com/ashhart/TensorFold/blob/v1.0.9/docs/cortheon.md#what-leaves-the-machine'

// Cortheon mode for the built-in engine: the server researches the web and follows a reasoning
// plan for every chat request it answers. Turning it on or off restarts a loaded model.
export function CortheonToggle({ status, activeRuns }: { status: EngineStatus | undefined; activeRuns: number }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const cortheon = status?.cortheon
  if (!status?.available || !cortheon) return null
  const loaded = status.state === 'ready' || status.state === 'starting'
  const blocked = !cortheon.supported || busy || (loaded && activeRuns > 0)
  const detail = !cortheon.supported
    ? "This engine version doesn't include Cortheon yet."
    : loaded && activeRuns > 0
      ? 'Stop active local runs first; changing this restarts the loaded model.'
      : cortheon.enabled && loaded && !cortheon.active
        ? 'Restarting the model with Cortheon on.'
        : 'Follows a reasoning plan and researches before answering, for any app using this engine. It sends search queries built from your chats to a web search service and reads public pages.' + (loaded ? ' Changing it restarts the loaded model.' : '')
  const change = async (enabled: boolean) => {
    setBusy(true)
    setError(null)
    try { await api.engineSetCortheon(enabled) } catch (e) { setError((e as Error).message) } finally { setBusy(false) }
  }
  return (
    <div data-testid="cortheon-toggle" className="flex items-start justify-between gap-4 border-t border-line px-4 py-3">
      <div className="min-w-0">
        <div className="flex items-center gap-2 text-[13px] text-ink">
          Cortheon
          {cortheon.active && <span className="rounded-full bg-info/12 px-1.5 py-px text-[11px] text-info-ink">On in this model</span>}
        </div>
        <p className="mt-0.5 text-[12px] text-ink-2">{detail}{cortheon.supported && <> <MarkdownLink href={PRIVACY_DOC} className="text-info-ink underline underline-offset-2">What leaves this Mac</MarkdownLink></>}</p>
        {error && <p role="alert" className="mt-1 text-[12px] text-danger">{error}</p>}
      </div>
      <Toggle label="Cortheon" checked={cortheon.enabled} disabled={blocked} onChange={(enabled) => void change(enabled)} />
    </div>
  )
}
