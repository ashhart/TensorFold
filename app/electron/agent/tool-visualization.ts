import type { VisualizationKind, VisualizationSpec, VisualizationTone } from './types'

const VISUALIZATION_KINDS = new Set<VisualizationKind>(['flow', 'bar', 'stats', 'timeline'])
const VISUALIZATION_TONES = new Set<VisualizationTone>(['accent', 'muted', 'positive', 'warning', 'danger'])

function cleanText(value: unknown, fallback = '', max = 160): string {
  return String(value ?? fallback).replace(/\s+/g, ' ').trim().slice(0, max)
}

function cleanTone(value: unknown): VisualizationTone | undefined {
  const v = String(value ?? '')
  return VISUALIZATION_TONES.has(v as VisualizationTone) ? v as VisualizationTone : undefined
}

export function filterNonSensitiveRetrievalKinds(value: unknown): string[] {
  const raw = Array.isArray(value)
    ? value
    : typeof value === 'string'
      ? value.split(',')
      : ['code', 'artifact']
  const allowed = new Set(['code', 'artifact'])
  const next = raw.map((item) => String(item).trim()).filter((item) => allowed.has(item))
  return next.length ? [...new Set(next)] : ['code', 'artifact']
}

function finiteNumber(value: unknown, fallback = 0): number {
  const n = Number(value)
  return Number.isFinite(n) ? n : fallback
}

export function normalizeVisualizationSpec(args: Record<string, any>): VisualizationSpec {
  const src = args.spec && typeof args.spec === 'object' ? args.spec : args
  const kind = VISUALIZATION_KINDS.has(src.kind) ? src.kind as VisualizationKind : 'flow'
  const title = cleanText(src.title, kind === 'bar' ? 'Chart' : 'Visualization', 120) || 'Visualization'
  const spec: VisualizationSpec = {
    kind,
    title,
    subtitle: cleanText(src.subtitle, '', 180) || undefined,
    caption: cleanText(src.caption, '', 240) || undefined,
  }

  if (Array.isArray(src.nodes)) {
    spec.nodes = src.nodes.slice(0, 18).map((n: any, i: number) => ({
      id: cleanText(n?.id, `node-${i + 1}`, 48) || `node-${i + 1}`,
      title: cleanText(n?.title ?? n?.label, `Node ${i + 1}`, 80) || `Node ${i + 1}`,
      subtitle: cleanText(n?.subtitle ?? n?.detail, '', 100) || undefined,
      tone: cleanTone(n?.tone),
      column: Number.isFinite(Number(n?.column)) ? Math.max(0, Math.min(8, Math.floor(Number(n.column)))) : undefined,
      row: Number.isFinite(Number(n?.row)) ? Math.max(0, Math.min(8, Math.floor(Number(n.row)))) : undefined,
    }))
  }

  if (Array.isArray(src.edges)) {
    spec.edges = src.edges.slice(0, 28).map((e: any) => ({
      from: cleanText(e?.from, '', 48),
      to: cleanText(e?.to, '', 48),
      label: cleanText(e?.label, '', 60) || undefined,
    })).filter((e: any) => e.from && e.to)
  }

  if (Array.isArray(src.bars)) {
    spec.bars = src.bars.slice(0, 12).map((b: any, i: number) => ({
      label: cleanText(b?.label, `Item ${i + 1}`, 80) || `Item ${i + 1}`,
      value: finiteNumber(b?.value),
      max: b?.max == null ? undefined : Math.max(0, finiteNumber(b.max)),
      detail: cleanText(b?.detail, '', 90) || undefined,
      tone: cleanTone(b?.tone),
    }))
  }

  if (Array.isArray(src.metrics)) {
    spec.metrics = src.metrics.slice(0, 8).map((m: any, i: number) => ({
      label: cleanText(m?.label, `Metric ${i + 1}`, 70) || `Metric ${i + 1}`,
      value: cleanText(m?.value, '0', 70) || '0',
      detail: cleanText(m?.detail, '', 100) || undefined,
      tone: cleanTone(m?.tone),
    }))
  }

  if (Array.isArray(src.events)) {
    spec.events = src.events.slice(0, 12).map((ev: any, i: number) => ({
      title: cleanText(ev?.title, `Event ${i + 1}`, 90) || `Event ${i + 1}`,
      time: cleanText(ev?.time, '', 60) || undefined,
      detail: cleanText(ev?.detail, '', 140) || undefined,
      tone: cleanTone(ev?.tone),
    }))
  }

  if (Array.isArray(src.legend)) {
    spec.legend = src.legend.slice(0, 6).map((l: any) => ({
      label: cleanText(l?.label, '', 70),
      tone: cleanTone(l?.tone),
    })).filter((l: any) => l.label)
  }

  return spec
}
