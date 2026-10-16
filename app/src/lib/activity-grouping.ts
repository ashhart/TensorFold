// Pure helpers for the thread activity display (Tasks #34 + #40).
//
// Task #34 — reasoning collapse: `reasoningDisplayState` decides whether a
// reasoning block renders expanded (full thinking text) or as a compact
// "Thinking" indicator, given the user's click state, whether the block is
// still streaming, and the default-collapse preference.
//
// Task #40 — tool grouping: `groupToolRuns` folds consecutive tool timeline
// items of the same tool into a single run so the renderer can collapse them
// into one "activity line" instead of N tool cards. `activityGroupLabel` and
// `activityGroupStateSummary` produce the summary line + state badge.
//
// Everything here is deliberately framework-free (no React) so it can be unit
// tested under jiti/node without a DOM.
import type { TimelineItem, ToolState } from './types'

export interface ReasoningDisplayState {
  // The full thinking text is shown.
  expanded: boolean
  // The compact spinner+label indicator is shown (active, collapsed by default).
  compact: boolean
}

// Decide how a reasoning block should render.
// - `open`: the user has clicked to expand it.
// - `active`: the block is still streaming (model is thinking).
// - `forceExpanded`: when true, an active block always expands (legacy
//   behaviour / "collapse reasoning by default" turned OFF). When the flag is
//   undefined we treat it as `true` for backward compatibility.
export function reasoningDisplayState({
  open,
  active,
  forceExpanded,
}: {
  open?: boolean
  active?: boolean
  forceExpanded?: boolean
}): ReasoningDisplayState {
  // A user click always wins.
  if (open) return { expanded: true, compact: false }
  // While active, expand only when not collapsing by default.
  if (active) {
    const expanded = forceExpanded !== false
    return { expanded, compact: !expanded }
  }
  // Inactive + not clicked open: collapsed header only (no compact spinner).
  return { expanded: false, compact: false }
}

// Two timeline items belong to the same activity run when they are both tool
// items invoking the same tool. Anything non-tool (user/assistant/reasoning/
// error/notice/visualization) breaks the run.
function sameToolRun(a: TimelineItem, b: TimelineItem): boolean {
  return a.kind === 'tool' && b.kind === 'tool' && (a.tool || '') === (b.tool || '')
}

// Group a flat timeline into consecutive runs. Single (non-grouped) items come
// back as length-1 arrays so callers can render them normally; runs of 2+
// same-tool items come back as a single array to be collapsed into one line.
export function groupToolRuns(timeline: TimelineItem[]): TimelineItem[][] {
  const groups: TimelineItem[][] = []
  for (const item of timeline) {
    const last = groups[groups.length - 1]
    if (last && sameToolRun(last[last.length - 1], item)) last.push(item)
    else groups.push([item])
  }
  return groups
}

// Pluralize a noun for a given count ("query" -> "3 queries").
function plural(count: number, singular: string, pluralForm?: string): string {
  const word = count === 1 ? singular : pluralForm ?? `${singular}s`
  return `${count} ${word}`
}

// Verb + count summary for a grouped tool run, e.g. "Search the web · 3 queries".
// Falls back to the item title for unknown tools.
export function activityGroupLabel(items: TimelineItem[]): string {
  const count = items.length
  const tool = items[0]?.tool || ''
  switch (tool) {
    case 'web_search': return `Search the web · ${plural(count, 'query', 'queries')}`
    case 'web_fetch': return `Fetch URLs · ${plural(count, 'request')}`
    case 'search': return `Search · ${plural(count, 'query', 'queries')}`
    case 'read_file': return `Read files · ${plural(count, 'file')}`
    case 'write_file': return `Create files · ${plural(count, 'change')}`
    case 'edit_file': return `Edit files · ${plural(count, 'change')}`
    case 'list_dir': return `List directories · ${plural(count, 'listing')}`
    case 'run_command': return `Run commands · ${plural(count, 'command')}`
    default: {
      const verb = items[0]?.title || tool || 'Activity'
      return `${verb} · ${plural(count, 'step')}`
    }
  }
}

export interface ActivityGroupStateSummary {
  ok: number
  error: number
  pending: number // running / pending / awaiting (in-flight)
  denied: number
  // Whether any item in the group is awaiting approval (auto-expand trigger).
  awaiting: boolean
  // A short badge string, e.g. "2 ok, 1 pending" or "" when uniform & ok.
  text: string
}

const IN_FLIGHT = new Set<ToolState>(['running', 'pending', 'awaiting'])

// Summarize the mix of states across a grouped tool run.
export function activityGroupStateSummary(items: TimelineItem[]): ActivityGroupStateSummary {
  let ok = 0, error = 0, pending = 0, denied = 0, awaiting = false
  for (const item of items) {
    const state = item.state
    if (state === 'awaiting') awaiting = true
    if (state === 'ok') ok++
    else if (state === 'error') error++
    else if (state === 'denied') denied++
    else if (state && IN_FLIGHT.has(state)) pending++
  }
  const parts: string[] = []
  if (ok) parts.push(`${ok} ok`)
  if (error) parts.push(`${error} error${error === 1 ? '' : 's'}`)
  if (pending) parts.push(`${pending} pending`)
  if (denied) parts.push(`${denied} denied`)
  // A single state (e.g. all ok) is implied by the icon, so only surface a
  // badge when the run is mixed or has anything non-ok worth flagging.
  const text = parts.length > 1 || error || pending || denied ? parts.join(', ') : ''
  return { ok, error, pending, denied, awaiting, text }
}
