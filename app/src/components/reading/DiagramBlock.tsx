import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { memo, useId, useMemo, useState } from 'react'
import { api } from '../../lib/ipc'
import { diagramLabelLines, diagramSvg, layoutDiagram, parseDiagram } from './diagram'

export const DiagramBlock = memo(function DiagramBlock({ source }: { source: string }) {
  const tx = useAuthoredText()
  const [showSource, setShowSource] = useState(false), [zoom, setZoom] = useState(100)
  const [copyState, setCopyState] = useState('')
  const parsed = useMemo(() => parseDiagram(source), [source]), id = useId().replace(/:/g, '')
  const layout = useMemo(() => parsed.ok ? layoutDiagram(parsed.model) : null, [parsed])
  const copy = async () => {
    if (!parsed.ok) return
    try { const result = await api.writeClipboard(diagramSvg(parsed.model)); setCopyState(result?.ok === false ? 'Copy failed.' : 'SVG copied. Paste it into an .svg file.') } catch { setCopyState('Copy failed.') }
  }
  return <section className="my-3 min-w-0 overflow-hidden rounded-lg border border-line bg-card-2" aria-label={tx("Diagram block")}>
    <div className="flex flex-wrap items-center gap-2 border-b border-line p-2 text-[12px] text-ink-2"><span className="mr-auto">{parsed.ok ? parsed.model.kind === 'sequence' ? tx("Sequence diagram") : tx("Flow diagram") : tx("Diagram source")}</span>
      <button type="button" aria-pressed={showSource} onClick={() => setShowSource((value) => !value)} className="rounded px-2 py-1 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent">{showSource ? tx("Show diagram") : tx("Show source")}</button>
      {parsed.ok && <><label>{tx("Zoom ")}<select aria-label={tx("Diagram zoom")} value={zoom} onChange={(event) => setZoom(Number(event.target.value))} className="rounded border border-line bg-card px-1 py-1">{[50, 75, 100, 150, 200].map((value) => <option key={value} value={value}>{value}%</option>)}</select></label><button type="button" onClick={() => void copy()} className="rounded px-2 py-1 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent">{tx("Copy SVG")}</button></>}
    </div>
    {!parsed.ok && <p role="status" className="px-3 pt-3 text-[12px] text-ink-3">{<OriginalDiagnostic>{parsed.error}</OriginalDiagnostic>} {tx("Source remains available below.")}</p>}
    {showSource || !parsed.ok ? <pre tabIndex={0} className="scroll-thin max-h-80 overflow-auto whitespace-pre-wrap break-words p-3 text-[12px] text-ink-2">{source}</pre> : layout && parsed.ok && <div className="scroll-thin max-h-[32rem] overflow-auto p-3"><svg role="img" aria-label={parsed.model.kind === 'sequence' ? tx("Sequence diagram") : tx("Flow diagram")} viewBox={`0 0 ${layout.width} ${layout.height}`} width={layout.width * zoom / 100} height={layout.height * zoom / 100} className="text-ink-2">
      <defs><marker id={`${id}-arrow`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="currentColor" /></marker></defs>
      <title>{parsed.model.kind === 'sequence' ? tx("Sequence") : tx("Flow")} {tx("diagram")}</title>
      {parsed.model.kind === 'sequence' && layout.points.map((point) => <line key={`life-${point.node.id}`} x1={point.x} y1={point.y + 34} x2={point.x} y2={layout.height - 10} stroke="currentColor" strokeDasharray="3 4" opacity=".4" />)}
      {layout.arrows.map((arrow, index) => <g key={index}><path d={arrow.path} fill="none" stroke="currentColor" strokeDasharray={arrow.edge.dashed ? '4 4' : undefined} markerEnd={`url(#${id}-arrow)`} /><text x={arrow.labelX} y={arrow.labelY} textAnchor={parsed.model.direction === 'TD' && parsed.model.kind === 'flow' ? 'start' : 'middle'} fill="currentColor" fontSize={11}>{arrow.edge.label}</text></g>)}
      {layout.points.map((point) => {
        const lines = diagramLabelLines(point.node.label)
        return <g key={point.node.id}><rect x={point.x - 95} y={point.y - 34} width={190} height={68} rx={6} fill="var(--card)" stroke="currentColor" /><text textAnchor="middle" fill="currentColor" fontSize={12}>{lines.map((line, index) => <tspan key={index} x={point.x} y={point.y + 4 + (index - (lines.length - 1) / 2) * 14}>{line}</tspan>)}<title>{point.node.label}</title></text></g>
      })}
    </svg><ol className="sr-only">{parsed.model.nodes.map((node) => <li key={node.id}>{node.label}</li>)}{parsed.model.edges.map((edge, index) => <li key={index}>{parsed.model.nodes.find((node) => node.id === edge.from)?.label} {tx("to ")}{parsed.model.nodes.find((node) => node.id === edge.to)?.label}: {edge.label}</li>)}</ol></div>}
    {copyState && <p role="status" className="px-3 pb-2 text-[12px] text-ink-3">{copyState}</p>}
  </section>
})
