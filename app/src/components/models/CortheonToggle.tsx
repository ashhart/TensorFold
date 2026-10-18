import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useState } from 'react'
import { api } from '../../lib/ipc'
import { MarkdownLink } from '../../lib/markdown-link'
import type { EngineStatus } from '../../lib/types'
import { Button, Toggle } from '../ui'
import { assertCortheonReleaseAllowed, CORTHEON_RELEASE_POLICY } from '../../../shared/release-feature-policy'

const PRIVACY_DOC = 'https://github.com/ashhart/TensorFold/blob/v1.0.10/docs/cortheon.md#what-leaves-the-machine'

// Cortheon mode for the built-in engine: the server researches the web and follows a reasoning
// plan for every chat request it answers. The beta.6 release policy disables enabling it.
export function CortheonToggle({ status, activeRuns, blocked: externallyBlocked = false }: { status: EngineStatus | undefined; activeRuns: number; blocked?: boolean }) {
  const tx = useAuthoredText()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const cortheon = status?.cortheon
  const releasePaused = !CORTHEON_RELEASE_POLICY.enabled
  if (!releasePaused && (!status?.available || !cortheon)) return null
  const loaded = status?.state === 'ready' || status?.state === 'starting'
  const blocked = releasePaused || externallyBlocked || !cortheon?.supported || busy || (loaded && activeRuns > 0)
  const detail = releasePaused
    ? CORTHEON_RELEASE_POLICY.reason + (cortheon?.active ? ' Research is active in an existing engine. Turn it off before continuing.' : ' Local chat and the API server remain available.')
    : !cortheon?.supported
      ? tx("This engine version doesn't include Cortheon yet.")
      : loaded && activeRuns > 0
        ? tx("Stop active local runs first; changing this restarts the loaded model.")
        : cortheon?.enabled && loaded && !cortheon?.active
          ? tx("Restarting the model with Cortheon on.")
          : 'Adds a reasoning plan and web research for any app using this engine. Research can send search queries built from your chats to a web search service and read public pages.' + (loaded ? ' Changing it restarts the loaded model.' : '')
  const change = async (enabled: boolean) => {
    setBusy(true)
    setError(null)
    try { assertCortheonReleaseAllowed(enabled); await api.engineSetCortheon(enabled) } catch (e) { setError((e as Error).message) } finally { setBusy(false) }
  }
  return (
    <div data-testid="cortheon-toggle" className="flex items-start justify-between gap-4 border-t border-line px-4 py-3">
      <div className="min-w-0">
        <div className="flex items-center gap-2 text-[13px] text-ink">
          {tx("Cortheon")}{cortheon?.active && <span className="rounded-full bg-info/12 px-1.5 py-px text-[11px] text-info-ink">{tx("On in this model")}</span>}
        </div>
        <p className="mt-0.5 text-[12px] text-ink-2">{detail}{!releasePaused && cortheon?.supported && <> <MarkdownLink href={PRIVACY_DOC} className="text-info-ink underline underline-offset-2">{tx("What leaves this Mac")}</MarkdownLink></>}</p>
        {error && <p role="alert" className="mt-1 text-[12px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
      </div>
      <div className="flex shrink-0 flex-col items-end gap-2">
        <Toggle label={tx("Cortheon")} checked={cortheon?.active === true || (!releasePaused && cortheon?.enabled === true)} disabled={blocked} onChange={(enabled) => void change(enabled)} />
        {releasePaused && cortheon?.active && <Button size="xs" disabled={externallyBlocked || busy || activeRuns > 0} onClick={() => void change(false)}>{tx("Turn off")}</Button>}
      </div>
    </div>
  )
}
