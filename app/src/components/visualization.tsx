import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import clsx from 'clsx'
import { Icon } from '../lib/icons'
import type {
  TimelineItem,
  VisualizationBar,
  VisualizationEdge,
  VisualizationMetric,
  VisualizationNode,
  VisualizationSpec,
  VisualizationTimelineEvent,
  VisualizationTone,
} from '../lib/types'

const KINDS = new Set(['flow', 'bar', 'stats', 'timeline'])
const TONES = new Set(['accent', 'muted', 'positive', 'warning', 'danger'])

type Segment = { type: 'markdown'; text: string } | { type: 'visualization'; spec: VisualizationSpec }

function text(value: unknown, fallback = '', max = 160): string {
  return String(value ?? fallback).replace(/\s+/g, ' ').trim().slice(0, max)
}

function tone(value: unknown): VisualizationTone | undefined {
  const v = String(value ?? '')
  return TONES.has(v) ? v as VisualizationTone : undefined
}

function num(value: unknown, fallback = 0): number {
  const n = Number(value)
  return Number.isFinite(n) ? n : fallback
}

export function normalizeVisualizationSpec(input: unknown): VisualizationSpec | null {
  if (!input || typeof input !== 'object') return null
  const src: any = (input as any).spec && typeof (input as any).spec === 'object' ? (input as any).spec : input
  const kind = KINDS.has(src.kind) ? src.kind : 'flow'
  const spec: VisualizationSpec = {
    kind,
    title: text(src.title, 'Visualization', 120) || 'Visualization',
    subtitle: text(src.subtitle, '', 180) || undefined,
    caption: text(src.caption, '', 240) || undefined,
  }

  if (Array.isArray(src.nodes)) {
    spec.nodes = src.nodes.slice(0, 18).map((n: any, i: number) => ({
      id: text(n?.id, `node-${i + 1}`, 48) || `node-${i + 1}`,
      title: text(n?.title ?? n?.label, `Node ${i + 1}`, 80) || `Node ${i + 1}`,
      subtitle: text(n?.subtitle ?? n?.detail, '', 100) || undefined,
      tone: tone(n?.tone),
      column: Number.isFinite(Number(n?.column)) ? Math.max(0, Math.min(8, Math.floor(Number(n.column)))) : undefined,
      row: Number.isFinite(Number(n?.row)) ? Math.max(0, Math.min(8, Math.floor(Number(n.row)))) : undefined,
    }))
  }

  if (Array.isArray(src.edges)) {
    spec.edges = src.edges.slice(0, 28).map((e: any) => ({
      from: text(e?.from, '', 48),
      to: text(e?.to, '', 48),
      label: text(e?.label, '', 60) || undefined,
    })).filter((e: VisualizationEdge) => e.from && e.to)
  }

  if (Array.isArray(src.bars)) {
    spec.bars = src.bars.slice(0, 12).map((b: any, i: number) => ({
      label: text(b?.label, `Item ${i + 1}`, 80) || `Item ${i + 1}`,
      value: num(b?.value),
      max: b?.max == null ? undefined : Math.max(0, num(b.max)),
      detail: text(b?.detail, '', 90) || undefined,
      tone: tone(b?.tone),
    }))
  }

  if (Array.isArray(src.metrics)) {
    spec.metrics = src.metrics.slice(0, 8).map((m: any, i: number) => ({
      label: text(m?.label, `Metric ${i + 1}`, 70) || `Metric ${i + 1}`,
      value: text(m?.value, '0', 70) || '0',
      detail: text(m?.detail, '', 100) || undefined,
      tone: tone(m?.tone),
    }))
  }

  if (Array.isArray(src.events)) {
    spec.events = src.events.slice(0, 12).map((ev: any, i: number) => ({
      title: text(ev?.title, `Event ${i + 1}`, 90) || `Event ${i + 1}`,
      time: text(ev?.time, '', 60) || undefined,
      detail: text(ev?.detail, '', 140) || undefined,
      tone: tone(ev?.tone),
    }))
  }

  if (Array.isArray(src.legend)) {
    spec.legend = src.legend.slice(0, 6).map((l: any) => ({
      label: text(l?.label, '', 70),
      tone: tone(l?.tone),
    })).filter((l: any) => l.label)
  }

  return spec
}

export function splitVisualizationMarkdown(markdown: string): Segment[] {
  const segments: Segment[] = []
  const re = /```(?:vontra-widget|vontra-viz|visualization)(?:\s+json)?\s*\n([\s\S]*?)```/gi
  let last = 0
  let match: RegExpExecArray | null
  while ((match = re.exec(markdown))) {
    if (match.index > last) segments.push({ type: 'markdown', text: markdown.slice(last, match.index) })
    try {
      const spec = normalizeVisualizationSpec(JSON.parse(match[1].trim()))
      if (spec) segments.push({ type: 'visualization', spec })
      else segments.push({ type: 'markdown', text: match[0] })
    } catch {
      segments.push({ type: 'markdown', text: match[0] })
    }
    last = match.index + match[0].length
  }
  if (last < markdown.length) segments.push({ type: 'markdown', text: markdown.slice(last) })
  return segments.length ? segments : [{ type: 'markdown', text: markdown }]
}

function toneNodeClass(t?: VisualizationTone) {
  switch (t) {
    case 'accent': return 'border-accent/35 bg-accent text-accent-ink'
    case 'positive': return 'border-[color:var(--add-line)] bg-[color:var(--add-bg)] text-[color:var(--add-ink)]'
    case 'warning': return 'border-warn/35 bg-warn/10 text-ink'
    case 'danger': return 'border-danger/35 bg-danger/10 text-ink'
    default: return 'border-line-strong bg-card-2 text-ink'
  }
}

function toneDotClass(t?: VisualizationTone) {
  switch (t) {
    case 'accent': return 'bg-accent'
    case 'positive': return 'bg-positive'
    case 'warning': return 'bg-warn'
    case 'danger': return 'bg-danger'
    default: return 'bg-ink-4'
  }
}

function toneBarClass(t?: VisualizationTone) {
  switch (t) {
    case 'positive': return 'bg-positive'
    case 'warning': return 'bg-warn'
    case 'danger': return 'bg-danger'
    case 'muted': return 'bg-ink-3'
    default: return 'bg-accent'
  }
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: value < 10 ? 1 : 0 }).format(value)
}

function buildFlowLayout(nodes: VisualizationNode[], edges: VisualizationEdge[]) {
  const width = 184
  const height = 68
  const gapX = 78
  const gapY = 48
  const padX = 24
  const padTop = 28
  const col = new Map<string, number>()
  const row = new Map<string, number>()
  const explicitCols = new Set<string>()
  for (const n of nodes) {
    if (n.column != null) {
      col.set(n.id, n.column)
      explicitCols.add(n.id)
    }
  }
  for (const n of nodes) if (!col.has(n.id)) col.set(n.id, 0)
  for (let i = 0; i < nodes.length; i++) {
    for (const e of edges) {
      if (!col.has(e.from) || !col.has(e.to)) continue
      if (explicitCols.has(e.to)) continue
      col.set(e.to, Math.max(col.get(e.to) ?? 0, (col.get(e.from) ?? 0) + 1))
    }
  }
  const byCol = new Map<number, VisualizationNode[]>()
  for (const n of nodes) {
    const c = Math.max(0, col.get(n.id) ?? 0)
    const list = byCol.get(c) ?? []
    list.push(n)
    byCol.set(c, list)
  }
  for (const [c, list] of byCol) {
    list.sort((a, b) => (a.row ?? 999) - (b.row ?? 999) || nodes.indexOf(a) - nodes.indexOf(b))
    list.forEach((n, i) => row.set(n.id, n.row == null ? i : n.row))
    byCol.set(c, list)
  }
  const maxCol = Math.max(0, ...Array.from(col.values()))
  const maxRow = Math.max(0, ...Array.from(row.values()))
  const canvasW = padX * 2 + (maxCol + 1) * width + maxCol * gapX
  const canvasH = padTop + (maxRow + 1) * height + maxRow * gapY + 18
  const pos = new Map<string, { x: number; y: number; w: number; h: number; node: VisualizationNode }>()
  for (const n of nodes) {
    const x = padX + (col.get(n.id) ?? 0) * (width + gapX)
    const y = padTop + (row.get(n.id) ?? 0) * (height + gapY)
    pos.set(n.id, { x, y, w: width, h: height, node: n })
  }
  return { pos, width: canvasW, height: canvasH }
}

function FlowVisualization({ spec }: { spec: VisualizationSpec }) {
  const fallback: VisualizationNode[] = [{ id: 'idea', title: spec.title, subtitle: spec.subtitle, tone: 'accent' }]
  const nodes = spec.nodes?.length ? spec.nodes : fallback
  const edges = spec.edges ?? []
  const { pos, width, height } = buildFlowLayout(nodes, edges)

  return (
    <div className="scroll-thin overflow-x-auto pb-1">
      <div className="relative" style={{ width, height, minWidth: '100%' }}>
        <svg className="pointer-events-none absolute inset-0 text-ink-3" viewBox={`0 0 ${width} ${height}`} width={width} height={height} aria-hidden="true">
          <defs>
            <marker id={`arrow-${spec.title.replace(/\W+/g, '-')}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="rgb(var(--ink) / 0.34)" />
            </marker>
          </defs>
          {edges.map((edge, i) => {
            const a = pos.get(edge.from)
            const b = pos.get(edge.to)
            if (!a || !b) return null
            const start = { x: a.x + a.w, y: a.y + a.h / 2 }
            const end = { x: b.x, y: b.y + b.h / 2 }
            const c = Math.max(24, Math.abs(end.x - start.x) * 0.42)
            const path = end.x > start.x
              ? `M ${start.x} ${start.y} C ${start.x + c} ${start.y}, ${end.x - c} ${end.y}, ${end.x - 8} ${end.y}`
              : `M ${a.x + a.w / 2} ${a.y + a.h} L ${b.x + b.w / 2} ${b.y - 8}`
            return (
              <g key={`${edge.from}-${edge.to}-${i}`}>
                <path d={path} fill="none" stroke="rgb(var(--ink) / 0.32)" strokeWidth="1.4" markerEnd={`url(#arrow-${spec.title.replace(/\W+/g, '-')})`} />
                {edge.label && (
                  <text x={(start.x + end.x) / 2} y={(start.y + end.y) / 2 - 8} textAnchor="middle" className="fill-ink-3 text-[11px] font-medium">
                    {edge.label}
                  </text>
                )}
              </g>
            )
          })}
        </svg>
        {nodes.map((node) => {
          const p = pos.get(node.id)
          if (!p) return null
          const accent = node.tone === 'accent'
          return (
            <div
              key={node.id}
              className={clsx('absolute flex flex-col justify-center overflow-hidden rounded-lg border px-3 text-center shadow-card', toneNodeClass(node.tone))}
              style={{ left: p.x, top: p.y, width: p.w, height: p.h }}
            >
              <div className="truncate text-[13px] font-semibold leading-snug">{node.title}</div>
              {node.subtitle && <div className={clsx('mt-0.5 line-clamp-2 text-[13px] leading-tight', accent ? 'text-accent-ink/72' : 'text-ink-3')}>{node.subtitle}</div>}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function BarVisualization({ bars }: { bars: VisualizationBar[] }) {
  const tx = useAuthoredText()
  const rows: VisualizationBar[] = bars.length ? bars : [{ label: tx("No data"), value: 0 }]
  const max = Math.max(1, ...rows.map((b) => b.max ?? b.value))
  return (
    <div className="space-y-3">
      {rows.map((bar, i) => {
        const denom = bar.max && bar.max > 0 ? bar.max : max
        const ratio = (bar.value / denom) * 100
        const pct = Number.isFinite(ratio) ? Math.max(0, Math.min(100, ratio)) : 0
        return (
          <div key={`${bar.label}-${i}`}>
            <div className="mb-1 flex items-center gap-3 text-[13px]">
              <span className="min-w-0 flex-1 truncate font-medium text-ink">{bar.label}</span>
              {bar.detail && <span className="truncate text-ink-3">{bar.detail}</span>}
              <span className="font-mono text-ink-2">{formatNumber(bar.value)}</span>
            </div>
            <div className="h-2.5 overflow-hidden rounded-full bg-ink-5">
              <div className={clsx('h-full rounded-full', toneBarClass(bar.tone))} style={{ width: `${pct}%` }} />
            </div>
          </div>
        )
      })}
    </div>
  )
}

function StatsVisualization({ metrics }: { metrics: VisualizationMetric[] }) {
  const tx = useAuthoredText()
  const rows: VisualizationMetric[] = metrics.length ? metrics : [{ label: tx("Metric"), value: '0' }]
  return (
    <div className="grid grid-cols-2 gap-2.5 md:grid-cols-4">
      {rows.map((metric, i) => (
        <div key={`${metric.label}-${i}`} className="rounded-lg border border-line bg-card-2 px-3 py-2.5">
          <div className="flex items-center gap-1.5 text-[12px] font-medium uppercase text-ink-3">
            <span className={clsx('h-2 w-2 rounded-full', toneDotClass(metric.tone))} />
            <span className="truncate">{metric.label}</span>
          </div>
          <div className="mt-1 truncate text-[20px] font-semibold leading-tight text-ink">{metric.value}</div>
          {metric.detail && <div className="mt-0.5 truncate text-[13px] text-ink-3">{metric.detail}</div>}
        </div>
      ))}
    </div>
  )
}

function TimelineVisualization({ events }: { events: VisualizationTimelineEvent[] }) {
  const rows = events.length ? events : [{ title: 'No events' }]
  return (
    <div className="space-y-0">
      {rows.map((event, i) => (
        <div key={`${event.title}-${i}`} className="relative pl-6">
          {i < rows.length - 1 && <div className="absolute left-[7px] top-4 h-full w-px bg-line-strong" />}
          <span className={clsx('absolute left-0 top-1.5 h-3.5 w-3.5 rounded-full border-2 border-card', toneDotClass(event.tone))} />
          <div className="pb-4">
            <div className="flex items-baseline gap-2">
              <div className="min-w-0 flex-1 truncate text-[13px] font-medium text-ink">{event.title}</div>
              {event.time && <div className="shrink-0 font-mono text-[12px] text-ink-3">{event.time}</div>}
            </div>
            {event.detail && <div className="mt-0.5 text-[13px] leading-relaxed text-ink-3">{event.detail}</div>}
          </div>
        </div>
      ))}
    </div>
  )
}

function Legend({ spec }: { spec: VisualizationSpec }) {
  if (!spec.legend?.length && !spec.caption) return null
  return (
    <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-line pt-3 text-[13px] text-ink-3">
      {spec.legend?.map((item, i) => (
        <span key={`${item.label}-${i}`} className="inline-flex items-center gap-1.5">
          <span className={clsx('h-3 w-3 rounded-[4px] border border-line', toneDotClass(item.tone))} />
          {item.label}
        </span>
      ))}
      {spec.caption && <span className="min-w-[180px] flex-1">{spec.caption}</span>}
    </div>
  )
}

export function VisualizationCard({ item, spec: rawSpec }: { item?: TimelineItem; spec?: VisualizationSpec }) {
  const tx = useAuthoredText()
  const spec = normalizeVisualizationSpec(rawSpec ?? item?.visualization ?? item?.args)
  const failed = item?.state === 'error'
  const loading = item?.state === 'pending' || item?.state === 'running'

  if (!spec) {
    return (
      <div className="fade-in rounded-xl border border-danger/25 bg-danger/8 px-3.5 py-3 text-[13px] text-ink">
        {tx("Unable to render visualization.")}</div>
    )
  }

  return (
    <div className={clsx('fade-in overflow-hidden rounded-xl border bg-card shadow-card', failed ? 'border-danger/25' : 'border-line')}>
      <div className="flex items-center gap-2 border-b border-line bg-card-2 px-3.5 py-2.5">
        <Icon name="grid" size={14} className="text-ink-3" />
        <span className="text-[13px] font-medium text-ink-2">{tx("Widget from ")}{item?.tool || tx("assistant")}</span>
        <code className="rounded-md border border-line bg-card px-1.5 py-0.5 font-mono text-[12px] text-ink-3">{spec.kind}</code>
        <span className="min-w-0 flex-1 truncate text-[12px] text-ink-3">{item?.meta}</span>
        {loading && <span className="h-3 w-3 animate-spin rounded-full border-2 border-ink-5 border-t-accent" />}
        {failed && <Icon name="warning" size={14} className="text-danger" />}
      </div>
      <div className="px-4 py-4">
        <div className="mb-4">
          <h3 className="truncate text-[15px] font-semibold text-ink">{spec.title}</h3>
          {spec.subtitle && <p className="mt-0.5 text-[13px] leading-relaxed text-ink-3">{spec.subtitle}</p>}
        </div>
        {spec.kind === 'flow' && <FlowVisualization spec={spec} />}
        {spec.kind === 'bar' && <BarVisualization bars={spec.bars ?? []} />}
        {spec.kind === 'stats' && <StatsVisualization metrics={spec.metrics ?? []} />}
        {spec.kind === 'timeline' && <TimelineVisualization events={spec.events ?? []} />}
        <Legend spec={spec} />
        {failed && item?.output && <pre className="mt-3 whitespace-pre-wrap rounded-lg border border-danger/25 bg-danger/8 px-3 py-2 font-mono text-[12px] text-danger">{item.output}</pre>}
      </div>
    </div>
  )
}
