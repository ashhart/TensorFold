import type { EngineLearningFact } from './types'

export const LEARNING_STATES = ['queued', 'learning', 'learned', 'missed', 'replaced', 'deleting', 'deleted'] as const
export function summarizeLearning(facts: EngineLearningFact[]) {
  return {
    total: facts.length,
    learned: facts.filter((fact) => fact.state === 'learned').length,
    recalled: facts.filter((fact) => fact.state === 'learned' && fact.recalled === true).length,
    missed: facts.filter((fact) => fact.state === 'missed').length,
    pending: facts.filter((fact) => fact.state === 'queued' || fact.state === 'learning' || fact.state === 'deleting').length,
  }
}
export function buildLearningLayout(facts: EngineLearningFact[], limit = 60) {
  const count = Math.max(0, Math.floor(limit))
  const visible = count ? facts.slice(-count) : []
  const rows = new Map<string, number>()
  const nodes = visible.map((fact) => {
    const column = LEARNING_STATES.indexOf(fact.state)
    const row = rows.get(fact.state) ?? 0
    rows.set(fact.state, row + 1)
    return { fact, x: 80 + Math.max(0, column) * 160, y: 76 + row * 58 }
  })
  const byId = new Map(nodes.map((node) => [node.fact.id, node]))
  const seen = new Set<string>()
  const edges: { from: typeof nodes[number]; to: typeof nodes[number] }[] = []
  for (const from of nodes) for (const id of from.fact.links) {
    const to = byId.get(id)
    if (!to || to === from) continue
    const key = [from.fact.id, id].sort((a, b) => a - b).join(':')
    if (seen.has(key)) continue
    seen.add(key); edges.push({ from, to })
  }
  return { nodes, edges, width: LEARNING_STATES.length * 160, height: Math.max(210, 110 + Math.max(0, ...rows.values()) * 58), hidden: facts.length - nodes.length }
}
export function filterLearningFacts(facts: EngineLearningFact[], query: string) {
  const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean)
  return [...facts].reverse().filter((fact) => words.every((word) => `${fact.id} ${fact.text} ${fact.state} ${fact.source}`.toLowerCase().includes(word)))
}
