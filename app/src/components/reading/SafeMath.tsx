import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { memo, useMemo, useState } from 'react'
import { renderBoundedMath } from './safe-math'

export const SafeMath = memo(function SafeMath({ expression, display = false }: { expression: string; display?: boolean }) {
  const tx = useAuthoredText()
  const [render, setRender] = useState(true)
  const result = useMemo(() => render ? renderBoundedMath(expression, display) : null, [expression, display, render])
  return <span className={display ? 'my-3 block max-w-full overflow-x-auto rounded-lg border border-line p-3' : 'inline'}>
    {result?.ok ? <span dangerouslySetInnerHTML={{ __html: result.html }} /> : <code className="whitespace-pre-wrap break-words">{display ? '$$' : '$'}{expression}{display ? '$$' : '$'}</code>}
    {result && !result.ok && <span className="ml-1 text-[12px] text-ink-3">{result.reason}</span>}
    <button type="button" onClick={() => setRender((value) => !value)} aria-pressed={render} className="ml-1 rounded px-1 text-[11px] text-ink-3 hover:text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent">{render ? tx("Math source") : tx("Render math")}</button>
  </span>
})
