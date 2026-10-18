import { useEffect, useRef, useState } from 'react'
import type { InputHistoryApi, InputHistoryScope, InputHistorySnapshot } from '../../../shared/input-history'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { Icon } from '../../lib/icons'
import { Menu } from '../ui'

interface PromptHistoryProps {
  scope: InputHistoryScope; port: InputHistoryApi; disabled?: boolean; onInsert: (text: string) => void
}

export function PromptHistoryPanel({ scope, port, disabled, onInsert, close }: PromptHistoryProps & { close: () => void }) {
  const tx = useAuthoredText(), [busy, setBusy] = useState(false)
  const [snapshot, setSnapshot] = useState<InputHistorySnapshot | null>(null), [error, setError] = useState<string | null>(null)
  const pending = useRef(false), active = useRef(true), current = useRef(scope), blocked = useRef(disabled)
  current.current = scope; blocked.current = disabled
  useEffect(() => { active.current = true; return () => { active.current = false } }, [])
  const operation = async (action: (owned: InputHistoryScope) => Promise<void>) => {
    if (pending.current || blocked.current) return
    pending.current = true; setBusy(true); setError(null)
    const owned = current.current
    try { await action(owned) }
    catch (failure) { if (active.current && current.current === owned) { setSnapshot(null); setError(failure instanceof Error ? failure.message : tx('Input history is unavailable.')) } }
    finally { pending.current = false; if (active.current && current.current === owned) setBusy(false) }
  }
  const refresh = () => { void operation(async owned => { const result = await port.getInputHistory(owned); if (active.current && current.current === owned) setSnapshot(result) }) }
  useEffect(refresh, [])
  return <section aria-label={tx('Recent prompts')} className="p-3">
    <div className="flex items-center justify-between gap-2"><p className="font-medium text-[13px] text-ink">{tx('Recent prompts')}</p><button type="button" onClick={close} className="rounded px-2 py-1 text-[12px] text-ink-3 hover:bg-ink-5">{tx('Close')}</button></div>
    <p className="my-2 text-[12px] leading-relaxed text-ink-3">{tx('Only direct inputs from this workspace and model account appear here. Private, connector, phone, bot and remote chats are excluded; files and quoted context are never recalled.')}</p>
    <p className="mb-2 text-[12px] text-ink-3">{tx('Insert appends the prompt to your current draft. Your text and files remain, and nothing is sent.')}</p>
    {busy && <p role="status" className="text-[12px] text-ink-3">{tx('Loading recent prompts…')}</p>}
    {error && <p role="alert" className="text-[12px] text-danger">{tx('Input history could not be read.')} <OriginalDiagnostic>{error}</OriginalDiagnostic> <button type="button" onClick={refresh} className="underline">{tx('Retry')}</button></p>}
    {snapshot && <>
      <div className="max-h-64 overflow-y-auto">{snapshot.entries.length ? snapshot.entries.map(entry => <div key={entry.id} className="mb-2 rounded-lg border border-line p-2"><p dir="auto" className="line-clamp-3 whitespace-pre-wrap break-words text-[12px] text-ink-2">{entry.text}</p><button type="button" disabled={busy || disabled} className="mt-1 text-[12px] text-accent disabled:opacity-40" onClick={() => void operation(async owned => {
        const prompt = await port.recallInputHistory({ scope: owned, id: entry.id, expectedRevision: snapshot.revision })
        if (active.current && !blocked.current && current.current === owned) { onInsert(prompt); close() }
      })}>{tx('Insert')}</button></div>) : <p className="text-[12px] text-ink-3">{tx('No recent prompts in this workspace and account.')}</p>}</div>
      {snapshot.partial && <p className="my-2 text-[12px] text-ink-3">{tx('Showing the latest available prompts from a bounded history scan.')}</p>}
      <button type="button" disabled={busy || disabled || !snapshot.entries.length} className="mt-1 rounded-md px-2 py-1 text-[12px] text-ink-3 hover:bg-ink-5 disabled:opacity-40" onClick={() => void operation(async owned => {
        const result = await port.clearInputHistory({ scope: owned, expectedRevision: snapshot.revision })
        if (active.current && current.current === owned) setSnapshot(result)
      })}>{tx('Clear recent prompts for this workspace and account')}</button>
      <p className="mt-1 text-[11px] text-ink-3">{tx('Clearing hides earlier prompts here; it does not delete your conversations.')}</p>
    </>}
  </section>
}

export function PromptHistory(props: PromptHistoryProps) {
  const tx = useAuthoredText()
  return <Menu align="end" placement="top" width={360} trigger={open => <button type="button" disabled={props.disabled} aria-label={tx('Recent prompts')} title={tx('Recent prompts')} aria-expanded={open} className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-ink-3 hover:bg-ink-5 hover:text-ink disabled:opacity-40"><Icon name="clock" size={15} /></button>}>
    {close => <PromptHistoryPanel {...props} close={close} />}
  </Menu>
}
