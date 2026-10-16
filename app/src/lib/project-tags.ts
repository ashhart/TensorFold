// Project tags — pure helpers for tagging projects (Work / Personal / custom) and
// filtering the Code & Chats sidebar by tag. Kept free of React/store so it can be
// unit-tested in plain node and reused from both the Sidebar filter and the
// ProjectView tag editor.
import type { Project } from './types'

// Built-in preset tags offered as one-click chips. Lowercase canonical form.
export const TAG_PRESETS = ['work', 'personal'] as const
export type PresetTag = (typeof TAG_PRESETS)[number]

// The palette a user can pick from when colouring a tag. Each entry carries the chip classes
// (bg+text), the icon text colour (for the tinted folder), and a solid swatch (for the picker).
// `name` is the value stored in settings.tagColors.
export interface TagColorDef { name: string; label: string; chip: string; icon: string; swatch: string }
export const TAG_COLORS: TagColorDef[] = [
  { name: 'green', label: 'Green', chip: 'bg-positive/15 text-positive', icon: 'text-positive', swatch: 'bg-positive' },
  { name: 'purple', label: 'Purple', chip: 'bg-accent-soft text-accent', icon: 'text-accent', swatch: 'bg-accent' },
  { name: 'orange', label: 'Orange', chip: 'bg-warn/15 text-warn', icon: 'text-warn', swatch: 'bg-warn' },
  { name: 'red', label: 'Red', chip: 'bg-danger/15 text-danger', icon: 'text-danger', swatch: 'bg-danger' },
  { name: 'neutral', label: 'Grey', chip: 'bg-ink-5 text-ink-2', icon: 'text-ink-3', swatch: 'bg-ink-3' },
]

// Preset defaults: Personal is green, Work is purple; any other tag defaults to neutral.
function defaultColorName(tag: string): string {
  const t = (tag || '').trim().toLowerCase()
  if (t === 'personal') return 'green'
  if (t === 'work') return 'purple'
  return 'neutral'
}

// Resolve the palette entry for a tag: the user's chosen colour (settings.tagColors[tag]) wins,
// else the preset default, else neutral. Pure.
export function tagColorDef(tag: string, colors?: Record<string, string> | null): TagColorDef {
  const chosen = colors?.[(tag || '').trim().toLowerCase()]
  const name = chosen && TAG_COLORS.some((c) => c.name === chosen) ? chosen : defaultColorName(tag)
  return TAG_COLORS.find((c) => c.name === name) ?? TAG_COLORS[TAG_COLORS.length - 1]
}

// Chip (bg+text) classes for a tag. Optional `colors` = settings.tagColors for user-chosen hues.
export function tagTone(tag: string, colors?: Record<string, string> | null): string {
  return tagColorDef(tag, colors).chip
}

// Just the text colour for a tag — used to TINT the project's folder icon (so a coloured folder
// replaces a space-hungry text chip on the project row).
export function tagIconColor(tag: string, colors?: Record<string, string> | null): string {
  return tagColorDef(tag, colors).icon
}

// Special filter sentinels. 'all' = no filtering; 'untagged' = only items with no
// tag (projects without tags + ungrouped chats/tasks, which never carry one).
export const ALL_FILTER = 'all'
export const UNTAGGED_FILTER = 'untagged'

// Normalize a raw, user-typed tag into its canonical stored form: trimmed,
// lower-cased, internal whitespace collapsed to single spaces. Returns '' for a
// blank/whitespace-only input so callers can reject it. Reserved sentinels
// ('all'/'untagged') are also rejected (-> '') so a user can't shadow them.
export function normalizeTag(raw: string | null | undefined): string {
  const t = (raw ?? '').trim().toLowerCase().replace(/\s+/g, ' ')
  if (!t || t === ALL_FILTER || t === UNTAGGED_FILTER) return ''
  return t
}

// Add a tag to a list, normalized + deduped. No-op (returns the same logical set)
// when the tag is blank or already present. Never mutates the input array.
export function addTag(tags: readonly string[] | undefined, raw: string): string[] {
  const next = normalizeTag(raw)
  const base = tags ?? []
  if (!next || base.includes(next)) return [...base]
  return [...base, next]
}

// Remove a tag from a list (normalized match). Never mutates the input array.
export function removeTag(tags: readonly string[] | undefined, raw: string): string[] {
  const target = normalizeTag(raw)
  return (tags ?? []).filter((t) => t !== target)
}

// Toggle a tag's membership — add if absent, remove if present.
export function toggleTag(tags: readonly string[] | undefined, raw: string): string[] {
  const target = normalizeTag(raw)
  if (!target) return [...(tags ?? [])]
  return (tags ?? []).includes(target) ? removeTag(tags, target) : addTag(tags, target)
}

// Does a project match the active sidebar filter?
//  - 'all'      -> always
//  - 'untagged' -> only projects with no tags
//  - <tag>      -> the project carries that (normalized) tag
export function projectMatchesTag(project: Pick<Project, 'tags'>, filter: string): boolean {
  if (filter === ALL_FILTER) return true
  const tags = project.tags ?? []
  if (filter === UNTAGGED_FILTER) return tags.length === 0
  return tags.includes(normalizeTag(filter))
}

// All distinct tags across the given projects, ordered by frequency (most-used
// first) then alphabetically for a stable tie-break — drives the filter chip bar.
export function collectTags(projects: ReadonlyArray<Pick<Project, 'tags'>>): string[] {
  const counts = new Map<string, number>()
  for (const p of projects) {
    for (const raw of p.tags ?? []) {
      const t = normalizeTag(raw)
      if (t) counts.set(t, (counts.get(t) ?? 0) + 1)
    }
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([t]) => t)
}

// Are there any untagged projects? (Used to decide whether to offer the
// "Untagged" filter chip.)
export function hasUntaggedProjects(projects: ReadonlyArray<Pick<Project, 'tags'>>): boolean {
  return projects.some((p) => (p.tags ?? []).length === 0)
}
