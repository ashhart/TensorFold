// Memory & context helpers (Wave 7). Two pure, testable pieces:
//  1) harvestMemoryCandidates — when compaction folds the middle of a transcript, salvage
//     durable facts/preferences/decisions worth remembering BEFORE they're summarized away,
//     as project-memory candidates the user approves. Bridges compaction -> durable memory.
//  2) selectRelevantMemory — query-driven recall: inject only the memory entries relevant to
//     the current turn, not the whole bucket (keeps the prompt tight + on-topic).

import { contentText } from './content.ts'
import type { ContentBlock } from './types'

interface MsgLike { role: string; content?: string | ContentBlock[] | null }

// Phrases that signal a durable preference / rule / decision worth keeping.
const DURABLE = /\b(always|never|prefer|i like|i hate|we use|we always|going forward|from now on|remember (?:that|to)?|my .{1,30} is|please (?:always|never)|stick to|default to|the convention is|we decided|let'?s go with)\b/i
// Ephemeral chatter to skip even if it trips a phrase.
const EPHEMERAL = /\b(today|right now|just now|for this (?:run|task|message)|temporarily|ignore that|never ?mind)\b/i

function sentences(text: string): string[] {
  return text.split(/(?<=[.!?])\s+|\n+/).map((s) => s.trim()).filter(Boolean)
}

/**
 * Salvage candidate durable facts from messages about to be folded by compaction.
 * Heuristic + conservative: only USER statements that look like a lasting preference,
 * rule, or decision (and not obviously ephemeral). Deduped, length-bounded, capped.
 * Each string becomes a pending project-memory candidate the user approves.
 */
export function harvestMemoryCandidates(folded: MsgLike[], cap = 5): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const m of folded) {
    const text = contentText(m.content)
    if (m.role !== 'user' || !text) continue
    for (const s of sentences(text)) {
      if (s.length < 8 || s.length > 240) continue
      if (!DURABLE.test(s) || EPHEMERAL.test(s)) continue
      const key = s.toLowerCase().replace(/\s+/g, ' ')
      if (seen.has(key)) continue
      seen.add(key)
      out.push(s)
      if (out.length >= cap) return out
    }
  }
  return out
}

export function harvestPostRunMemoryCandidates(
  messages: MsgLike[],
  options: { existingTexts?: string[]; cap?: number } = {},
): string[] {
  const existing = new Set((options.existingTexts ?? []).map(memoryKey).filter(Boolean))
  const candidates = harvestMemoryCandidates(messages.slice(-30), Math.max(1, options.cap ?? 3))
  const out: string[] = []
  for (const candidate of candidates) {
    const key = memoryKey(candidate)
    if (!key) continue
    let duplicate = false
    for (const known of existing) {
      if (known.includes(key) || key.includes(known)) {
        duplicate = true
        break
      }
    }
    if (duplicate) continue
    existing.add(key)
    out.push(candidate)
  }
  return out
}

interface EntryLike { id: string; text: string }

const STOP = new Set('the a an of to and or in on for with is are be it this that you i we my your our as at by from'.split(' '))
function tokens(text: string): Set<string> {
  return new Set((text.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((t) => t.length > 2 && !STOP.has(t)))
}

function memoryKey(text: string): string {
  return (text.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((t) => !STOP.has(t)).join(' ')
}

/**
 * Rank memory entries by token overlap with the current query and return the top-k that
 * actually share content (score > 0). Query-driven recall: only inject what's relevant
 * this turn instead of the whole memory bucket.
 */
export function selectRelevantMemory<T extends EntryLike>(entries: T[], query: string, k = 5): T[] {
  const q = tokens(query)
  if (!q.size) return []
  const scored = entries
    .map((e) => {
      const et = tokens(e.text)
      let overlap = 0
      for (const t of et) if (q.has(t)) overlap++
      // Normalize by entry length so a long entry doesn't always win on raw overlap.
      return { e, score: overlap === 0 ? 0 : overlap / Math.sqrt(et.size || 1) }
    })
    .filter((s) => s.score > 0)
    .sort((a, b) => b.score - a.score)
  return scored.slice(0, k).map((s) => s.e)
}
