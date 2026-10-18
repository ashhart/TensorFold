import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { Icon } from '../../lib/icons'

/** Quiet autosave feedback; failures retain a separate visible explanation and Retry. */
export function DraftSaveIndicator({ ready = true, saving = false, privateDraft = false, sessionOnly = false, problem = false, label }: {
  ready?: boolean; saving?: boolean; privateDraft?: boolean; sessionOnly?: boolean; problem?: boolean; label?: string;
}) {
  const tx = useAuthoredText()
  const detail = label || (!ready ? tx("Loading draft…") : privateDraft ? tx("Private draft: kept only until you quit") : sessionOnly ? tx("Draft kept for this session; use the desktop app to save across restarts.") : saving ? tx("Saving draft…") : tx("Draft saved on this device"))
  return <span role="img" aria-label={detail} title={detail} tabIndex={0} className={`inline-flex h-8 w-6 shrink-0 items-center justify-center rounded-md outline-none focus-visible:ring-2 focus-visible:ring-accent ${problem ? 'text-danger' : 'text-ink-3'}`}>
    <Icon name={problem || !ready || saving || sessionOnly ? 'clock' : privateDraft ? 'shield' : 'check'} size={13} />
  </span>
}
