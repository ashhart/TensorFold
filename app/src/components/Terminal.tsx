import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import { useEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import { api } from '../lib/ipc'
import { Icon } from '../lib/icons'
import { shortPath } from '../lib/format'
import type { WorkspaceLocation } from '../lib/types'

interface Line { cmd: string; output: string; code: number }

// A lightweight built-in terminal: run a command in the workspace, see output.
export function Terminal({ cwd, workspace = null, onClose }: { cwd: string; workspace?: WorkspaceLocation | null; onClose: () => void }) {
  const tx = useAuthoredText()
  const [lines, setLines] = useState<Line[]>([])
  const [cmd, setCmd] = useState('')
  const [busy, setBusy] = useState(false)
  const [history, setHistory] = useState<string[]>([])
  const [hIdx, setHIdx] = useState(-1)
  const endRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => { endRef.current?.scrollIntoView({ block: 'end' }) }, [lines, busy])
  useEffect(() => { inputRef.current?.focus() }, [])

  const run = async () => {
    const c = cmd.trim()
    if (!c || busy) return
    setHistory((h) => [...h, c]); setHIdx(-1); setCmd(''); setBusy(true)
    try { const r = await api.exec(cwd, c, workspace); setLines((ls) => [...ls, { cmd: c, output: r.output, code: r.code }]) }
    catch (e: any) { setLines((ls) => [...ls, { cmd: c, output: String(e?.message ?? e), code: 1 }]) }
    finally { setBusy(false); requestAnimationFrame(() => inputRef.current?.focus()) }
  }
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') { e.preventDefault(); run() }
    else if (e.key === 'ArrowUp') { e.preventDefault(); const i = hIdx < 0 ? history.length - 1 : Math.max(0, hIdx - 1); if (history[i] != null) { setHIdx(i); setCmd(history[i]) } }
    else if (e.key === 'ArrowDown') { e.preventDefault(); const i = hIdx + 1; if (i >= history.length) { setHIdx(-1); setCmd('') } else { setHIdx(i); setCmd(history[i]) } }
  }

  return (
    <div className="flex h-full flex-col bg-card-2/40">
      <div className="flex items-center gap-2 border-b border-line px-3 py-1.5">
        <Icon name="terminal" size={13} className="text-ink-3" />
        <span className="text-[13px] font-medium text-ink-2">{tx("Terminal")}</span>
        <span className="truncate text-[12px] text-ink-4">{cwd ? shortPath(cwd, 2) : tx("no folder")}</span>
        <div className="flex-1" />
        {lines.length > 0 && <button onClick={() => setLines([])} className="text-[12px] text-ink-3 transition-colors hover:text-ink">{tx("Clear")}</button>}
        <button onClick={onClose} className="flex h-6 w-6 items-center justify-center rounded text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink" aria-label={tx("Close terminal")}><Icon name="x" size={13} /></button>
      </div>
      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-3 py-2 font-mono text-[13px] leading-[1.5]">
        {lines.length === 0 && <div className="text-ink-4">{tx("Run a command in ")}{cwd ? shortPath(cwd, 2) : tx("the workspace")} {tx("— e.g. ")}<span className="text-ink-3">{tx("npm test")}</span>, <span className="text-ink-3">{tx("git status")}</span>.</div>}
        {lines.map((l, i) => (
          <div key={i} className="mb-1.5">
            <div className="flex gap-1.5"><span className="select-none text-accent">$</span><span className="break-all text-ink">{l.cmd}</span>{l.code !== 0 && <span className="ml-auto shrink-0 rounded bg-danger/10 px-1.5 py-0.5 text-[11px] font-medium text-danger">{tx("exit ")}{l.code}</span>}</div>
            {l.output && <pre className={clsx('mt-0.5 whitespace-pre-wrap break-words', l.code === 0 ? 'text-ink-2' : 'text-danger')}>{l.output}</pre>}
          </div>
        ))}
        {busy && <div className="text-ink-3">{tx("running…")}</div>}
        <div ref={endRef} className="h-px" />
      </div>
      <div className="flex items-center gap-1.5 border-t border-line px-3 py-1.5">
        <span className="select-none font-mono text-[13px] text-accent">$</span>
        <input ref={inputRef} value={cmd} onChange={(e) => setCmd(e.target.value)} onKeyDown={onKey} placeholder={tx("command…")} className="w-full bg-transparent font-mono text-[13px] text-ink outline-none placeholder:text-ink-4" spellCheck={false} />
      </div>
    </div>
  )
}
