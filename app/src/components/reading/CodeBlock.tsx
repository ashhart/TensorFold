import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { memo, useEffect, useRef, useState } from 'react'
import { api } from '../../lib/ipc'
import { highlight } from '../../lib/highlight'

const control = 'min-h-8 rounded-md px-2 py-1 text-[12px] font-medium text-ink-2 hover:bg-ink-5 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent'

// React keeps each block mounted while later Markdown streams in. Choices stay
// local to this block; content updates never reset wrap or collapse preferences.
export const CodeBlock = memo(function CodeBlock({ code, language }: { code: string; language?: string }) {
  const tx = useAuthoredText()
  const [wrap, setWrap] = useState(false), [collapsed, setCollapsed] = useState(false)
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'failed'>('idle')
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null), mounted = useRef(true)
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; if (timer.current) clearTimeout(timer.current) } }, [])
  const copy = async () => {
    try {
      const result = await api.writeClipboard(code)
      if (!mounted.current) return
      setCopyState(result?.ok === false ? 'failed' : 'copied')
    } catch { if (mounted.current) setCopyState('failed') }
    if (timer.current) clearTimeout(timer.current)
    if (mounted.current) timer.current = setTimeout(() => { if (mounted.current) setCopyState('idle') }, 1800)
  }
  const lines = code.split('\n').length
  return <div className="my-3 min-w-0 overflow-hidden rounded-lg border border-line bg-card-2">
    <div className="flex flex-wrap items-center gap-1 border-b border-line px-2 py-1">
      <span className="mr-auto px-1 text-[12px] text-ink-3">{language || tx("Code")} · {lines} {lines === 1 ? tx("line") : tx("lines")}</span>
      <button type="button" className={control} aria-pressed={wrap} onClick={() => setWrap((value) => !value)}>{tx("Wrap")}</button>
      <button type="button" className={control} aria-expanded={!collapsed} onClick={() => setCollapsed((value) => !value)}>{collapsed ? tx("Expand code") : tx("Collapse code")}</button>
      <button type="button" className={control} onClick={() => void copy()}>{copyState === 'copied' ? tx("Copied") : tx("Copy code")}</button>
    </div>
    <div hidden={collapsed}><pre tabIndex={0} aria-label={language ? tx("{value0} block", { value0: language }) : tx("Code block")} className={`scroll-thin max-h-[32rem] overflow-auto p-3 text-[13px] focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent ${wrap ? 'whitespace-pre-wrap break-words' : 'whitespace-pre'}`}><code className="syn-code">{code.length <= 128_000 ? highlight(code, language) : code}</code></pre></div>
    <span role="status" aria-live="polite" className={copyState === 'failed' ? 'block px-3 py-1 text-[12px] text-danger' : 'sr-only'}>{copyState === 'copied' ? tx("Code copied to clipboard.") : copyState === 'failed' ? tx("Copy failed. Select the code and copy it manually.") : ''}</span>
  </div>
})
