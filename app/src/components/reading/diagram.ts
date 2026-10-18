export interface DiagramNode { id: string; label: string }
export interface DiagramEdge { from: string; to: string; label: string; dashed: boolean }
export interface DiagramModel { kind: 'flow' | 'sequence'; direction: 'LR' | 'TD'; nodes: DiagramNode[]; edges: DiagramEdge[] }
export type DiagramParse = { ok: true; model: DiagramModel } | { ok: false; error: string }
export const DIAGRAM_LIMITS = { characters: 8192, lines: 64, nodes: 18, edges: 28, label: 80 } as const
const id = '[A-Za-z][A-Za-z0-9_]{0,31}'
const endpoint = `(${id})(?:\\[("[^"\\]\\n]*"|[^\\]\\n]*)\\])?`
const declaration = new RegExp(`^${endpoint}$`)
const flowEdge = new RegExp(`^${endpoint}\\s*-->\\s*(?:\\|([^|]*)\\|\\s*)?${endpoint}$`)
const sequenceEdge = new RegExp(`^(${id})(-->>|->>)(${id})\\s*:\\s*(.+)$`)
const participant = new RegExp(`^participant\\s+(${id})(?:\\s+as\\s+(.+))?$`)
const label = (value: string) => value.replace(/^"|"$/g, '').trim()

export function parseDiagram(source: string): DiagramParse {
  if (source.length > DIAGRAM_LIMITS.characters) return { ok: false, error: 'Diagram exceeds the 8,192 character limit.' }
  const lines = source.replace(/\r\n?/g, '\n').split('\n').map((line) => line.trim()).filter(Boolean)
  if (lines.length > DIAGRAM_LIMITS.lines) return { ok: false, error: 'Diagram exceeds the 64 line limit.' }
  const heading = lines.shift()
  const direction = /^(?:flowchart|graph) (LR|TD|TB)$/.exec(heading ?? '')
  const kind = heading === 'sequenceDiagram' ? 'sequence' : direction ? 'flow' : null
  if (!kind) return { ok: false, error: 'Supported syntax starts with flowchart LR, flowchart TD or sequenceDiagram.' }
  const nodes: DiagramNode[] = [], edges: DiagramEdge[] = []
  let error = ''
  const add = (nodeId: string, raw?: string) => {
    const nodeLabel = label(raw ?? nodeId)
    if (!nodeLabel || nodeLabel.length > DIAGRAM_LIMITS.label || /[<>\u0000-\u001f]/.test(nodeLabel)) { error = 'Node labels must be plain text with up to 80 characters.'; return }
    const found = nodes.find((node) => node.id === nodeId)
    if (found) { if (raw && found.label !== nodeId && found.label !== nodeLabel) error = 'A node was declared with conflicting labels.'; else if (raw) found.label = nodeLabel }
    else nodes.push({ id: nodeId, label: nodeLabel })
  }
  for (const line of lines) {
    if (line.startsWith('%%')) continue
    if (kind === 'flow') {
      const edge = flowEdge.exec(line), node = declaration.exec(line)
      if (edge) { add(edge[1], edge[2]); add(edge[4], edge[5]); edges.push({ from: edge[1], to: edge[4], label: label(edge[3] ?? ''), dashed: false }) }
      else if (node) add(node[1], node[2])
      else return { ok: false, error: 'Only node declarations and one A --> B or A -->|label| B edge per line are supported.' }
    } else {
      const edge = sequenceEdge.exec(line), actor = participant.exec(line)
      if (actor) add(actor[1], actor[2])
      else if (edge) { add(edge[1]); add(edge[3]); edges.push({ from: edge[1], to: edge[3], label: label(edge[4]), dashed: edge[2] === '-->>' }) }
      else return { ok: false, error: 'Only participant declarations and A->>B: message or A-->>B: reply are supported.' }
    }
    if (error) return { ok: false, error }
    if (nodes.length > DIAGRAM_LIMITS.nodes || edges.length > DIAGRAM_LIMITS.edges) return { ok: false, error: 'Diagram exceeds 18 nodes or 28 edges.' }
    if (edges.some((edge) => edge.from === edge.to)) return { ok: false, error: 'Self-loop arrows are not supported. The source is shown instead.' }
    if (edges.some((edge) => edge.label.length > DIAGRAM_LIMITS.label || /[<>\u0000-\u001f]/.test(edge.label))) return { ok: false, error: 'Edge labels must be plain text with up to 80 characters.' }
  }
  if (!nodes.length) return { ok: false, error: 'Diagram has no nodes yet.' }
  return { ok: true, model: { kind, direction: direction?.[1] === 'LR' ? 'LR' : 'TD', nodes, edges } }
}

export interface DiagramLayout { width: number; height: number; points: { x: number; y: number; node: DiagramNode }[]; arrows: { path: string; labelX: number; labelY: number; edge: DiagramEdge }[] }
export function diagramLabelLines(value: string): string[] {
  const lines: string[] = []
  let remaining = value
  while (remaining.length > 26) {
    const space = remaining.lastIndexOf(' ', 26), split = space > 8 ? space : 26
    lines.push(remaining.slice(0, split)); remaining = remaining.slice(split).trimStart()
  }
  if (remaining) lines.push(remaining)
  return lines
}
export function layoutDiagram(model: DiagramModel): DiagramLayout {
  const horizontal = model.kind === 'sequence' || model.direction === 'LR'
  const points = model.nodes.map((node, index) => ({ node, x: horizontal ? 110 + index * 220 : 130, y: horizontal ? 50 : 50 + index * 120 }))
  const width = horizontal ? Math.max(240, model.nodes.length * 220) : 300 + model.edges.length * 14
  const height = model.kind === 'sequence' ? 150 + model.edges.length * 65 : horizontal ? 150 + model.edges.length * 35 : Math.max(140, model.nodes.length * 120)
  const arrows = model.edges.map((edge, index) => {
    const from = points.find((point) => point.node.id === edge.from)!, to = points.find((point) => point.node.id === edge.to)!
    if (model.kind === 'sequence') { const y = 115 + index * 65; return { path: `M${from.x} ${y} L${to.x} ${y}`, labelX: (from.x + to.x) / 2, labelY: y - 8, edge } }
    if (horizontal) { const lane = 115 + index * 35; return { path: `M${from.x} ${from.y + 34} L${from.x} ${lane} L${to.x} ${lane} L${to.x} ${to.y + 34}`, labelX: (from.x + to.x) / 2, labelY: lane - 8, edge } }
    const lane = 260 + index * 14
    return { path: `M${from.x + 95} ${from.y} L${lane} ${from.y} L${lane} ${to.y} L${to.x + 95} ${to.y}`, labelX: lane + 8, labelY: (from.y + to.y) / 2, edge }
  })
  return { width, height, points, arrows }
}
const xml = (value: string) => value.replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' })[char]!)
export function diagramSvg(model: DiagramModel): string {
  const layout = layoutDiagram(model)
  const boxes = layout.points.map((point) => {
    const lines = diagramLabelLines(point.node.label)
    return `<g><rect x="${point.x - 95}" y="${point.y - 34}" width="190" height="68" rx="6" fill="white" stroke="currentColor"/><text text-anchor="middle">${lines.map((line, index) => `<tspan x="${point.x}" y="${point.y + 4 + (index - (lines.length - 1) / 2) * 14}">${xml(line)}</tspan>`).join('')}</text></g>`
  }).join('')
  const arrows = layout.arrows.map((arrow) => `<g><path d="${arrow.path}" fill="none" stroke="currentColor" marker-end="url(#tf-diagram-arrow)"${arrow.edge.dashed ? ' stroke-dasharray="4 4"' : ''}/><text x="${arrow.labelX}" y="${arrow.labelY}" text-anchor="${model.direction === 'TD' && model.kind === 'flow' ? 'start' : 'middle'}">${xml(arrow.edge.label)}</text></g>`).join('')
  const lifelines = model.kind === 'sequence' ? layout.points.map((point) => `<line x1="${point.x}" y1="${point.y + 34}" x2="${point.x}" y2="${layout.height - 10}" stroke="currentColor" stroke-dasharray="3 4" opacity=".4"/>`).join('') : ''
  return `<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${model.kind === 'sequence' ? 'Sequence' : 'Flow'} diagram" viewBox="0 0 ${layout.width} ${layout.height}" width="${layout.width}" height="${layout.height}" color="#222" font-family="sans-serif" font-size="12"><defs><marker id="tf-diagram-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>${lifelines}${arrows}${boxes}</svg>`
}
