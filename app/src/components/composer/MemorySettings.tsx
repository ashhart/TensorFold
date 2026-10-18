import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useState } from 'react'
import type { ThreadMemoryPolicy } from '../../lib/types'

export interface MemorySettingsProps {
  policy: ThreadMemoryPolicy | null
  problem?: string | null
  revision?: number
  incognito: boolean
  busy: boolean
  unsupported: string | null
  isNew: boolean
  onSave: (policy: ThreadMemoryPolicy) => Promise<void>
}

/** Draft choices do not become effective until the host accepts the durable save. */
export function MemorySettings({ policy, problem, revision, incognito, busy, unsupported, isNew, onSave }: MemorySettingsProps) {
  const tx = useAuthoredText()
  const [draftRevision, setDraftRevision] = useState(revision)
  const stale = draftRevision !== revision
  const [draft, setDraft] = useState(policy ? { ...policy } : { useExisting: false, contribute: false })
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const dirty = !policy || draft.useExisting !== policy.useExisting || draft.contribute !== policy.contribute
  const save = async () => {
    setSaving(true); setError(null)
    try { await onSave(draft); setDraftRevision(revision === undefined ? undefined : revision + 1) }
    catch (error) { setError(error instanceof Error ? error.message : tx("Memory choices could not be saved. Your choices are still here.")) }
    finally { setSaving(false) }
  }
  return <div className="space-y-3 px-2.5 py-2 text-[13px] leading-relaxed text-ink-2">
    {problem && <p role="alert" className="text-[12px] text-warn">{problem} {tx("These unchecked draft choices are not an effective policy.")}</p>}
    <label className="flex items-start gap-2.5">
      <input type="checkbox" checked={draft.useExisting} disabled={busy || saving} onChange={(event) => setDraft({ ...draft, useExisting: event.target.checked })} className="mt-1 accent-accent" />
      <span><span className="block font-medium text-ink">{tx("Use saved memory")}</span><span className="text-[12px] text-ink-3">{tx("Recall previous chats and saved project or room memories.")}</span></span>
    </label>
    <label className="flex items-start gap-2.5">
      <input type="checkbox" checked={!incognito && draft.contribute} disabled={incognito || busy || saving} onChange={(event) => setDraft({ ...draft, contribute: event.target.checked })} className="mt-1 accent-accent" />
      <span><span className="block font-medium text-ink">{tx("Propose new memories")}</span><span className="text-[12px] text-ink-3">{tx("Suggest memories and skill improvements for your review; nothing is approved or trained automatically.")}</span></span>
    </label>
    <p className="text-[12px] text-ink-3">{incognito ? tx("Incognito never contributes memories and keeps chat history only while the app is open.") : tx("Chat history is still saved, even with both options off.")} {tx("Providers and tools may retain their own records.")}</p>
    <p className="text-[12px] text-ink-3">{tx("Changes apply to the next turn; they do not erase this conversation or undo learned weights. Start a new chat for fresh context.")}</p>
    {stale && <p role="status" className="text-[12px] text-warn">{tx("Memory choices changed in another window or a new turn started; reopen this menu to review them.")}</p>}
    {busy && <p role="status" className="text-[12px] text-ink-3">{tx("Stop this run to change memory choices.")}</p>}
    {unsupported && <p role="status" className="text-[12px] text-warn">{unsupported} {tx("These controls are not enforced on that route.")}</p>}
    {error && <p role="alert" className="text-[12px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
    <button type="button" disabled={!dirty || saving || busy || stale || !!unsupported} onClick={() => void save()} className="w-full rounded-lg bg-accent px-3 py-2 font-medium text-accent-ink disabled:opacity-40">{saving ? tx("Saving…") : dirty ? isNew ? tx("Use for new chat") : tx("Save choices") : isNew ? tx("Selected for new chat") : tx("Saved")}</button>
  </div>
}
