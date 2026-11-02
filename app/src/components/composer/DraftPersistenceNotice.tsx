import { SESSION_ONLY_DRAFT_NOTICE } from '../../lib/conversation-drafts'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { Icon } from '../../lib/icons'
import { OriginalDiagnostic } from '../OriginalDiagnostic'

export function isSessionOnlyDraftNotice(error: string | null, sessionOnly: boolean): boolean {
  return sessionOnly && error === SESSION_ONLY_DRAFT_NOTICE
}

export function DraftPersistenceNotice({ error, sessionOnly, onRetry }: {
  error: string | null
  sessionOnly: boolean
  onRetry: () => void
}) {
  const tx = useAuthoredText()
  if (!error) return null
  if (isSessionOnlyDraftNotice(error, sessionOnly)) return <p role="status" className="flex items-start gap-2 px-4 pt-3 text-[11px] leading-relaxed text-ink-3">
    <Icon name="monitor" size={13} className="mt-0.5 shrink-0" />
    <span>{tx('This draft stays in this tab. Use the desktop app to keep drafts across restarts.')}</span>
  </p>
  return <div role="alert" className="px-4 pt-3 text-[12px] leading-relaxed text-danger">
    <span className="font-medium">{tx('Draft not saved: ')}</span><OriginalDiagnostic>{error}</OriginalDiagnostic>
    <button type="button" onClick={onRetry} className="mt-1 rounded px-1 py-0.5 underline focus-visible:outline-2 focus-visible:outline-accent">{tx('Retry save')}</button>
  </div>
}
