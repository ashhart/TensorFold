import { useEffect, useRef, useState, type ReactNode } from 'react'

/** A question asked in the page, where the action is, instead of the browser's own dialog: it can say what the
 *  action means, it looks like the rest of TensorFold Web, and Escape or "Keep" leaves things as they are. */
export function Ask({ text, yes, danger, busy, onYes, onNo }: { text: ReactNode; yes: string; danger?: boolean; busy?: boolean; onYes: () => void; onNo: () => void }) {
  const first = useRef<HTMLButtonElement>(null)
  useEffect(() => { first.current?.focus() }, [])
  return (
    <div className="ask" role="alertdialog" aria-live="polite" onKeyDown={(e) => { if (e.key === 'Escape') onNo() }}>
      <span>{text}</span>
      <span className="actions">
        <button ref={first} className={`btn sm ${danger ? 'danger' : 'primary'}`} aria-busy={busy} onClick={onYes}>{yes}</button>
        <button className="btn sm quiet" onClick={onNo}>Keep as it is</button>
      </span>
    </div>
  )
}

/** A name asked in the page: prefilled, cleaned as typed when `clean` is given, Enter to accept, Escape to leave. */
export function AskName({ text, initial, ok, clean, onOk, onCancel }: { text: ReactNode; initial: string; ok: string; clean?: (s: string) => string; onOk: (name: string) => void; onCancel: () => void }) {
  const [value, setValue] = useState(initial)
  const input = useRef<HTMLInputElement>(null)
  useEffect(() => { input.current?.focus(); input.current?.select() }, [])
  const accept = () => { const v = value.trim(); if (v) onOk(v) }
  return (
    <div className="ask" role="dialog" aria-live="polite" onKeyDown={(e) => { if (e.key === 'Escape') onCancel(); else if (e.key === 'Enter') { e.preventDefault(); accept() } }}>
      <label className="askname"><span>{text}</span><input ref={input} className="input" value={value} onChange={(e) => setValue(clean ? clean(e.target.value) : e.target.value)} aria-label={typeof text === 'string' ? text : 'Name'} /></label>
      <span className="actions">
        <button className="btn sm primary" disabled={!value.trim()} onClick={accept}>{ok}</button>
        <button className="btn sm quiet" onClick={onCancel}>Cancel</button>
      </span>
    </div>
  )
}

/** The usual cleaning of a copy's or a point's name: lower case, letters, digits, dashes and underscores. */
export const cleanName = (s: string) => s.toLowerCase().replace(/[^a-z0-9_-]/g, '-')
