import type { ConversationSearchHighlight } from './conversation-search'

export const MAX_SEARCH_QUERY_CHARS = 512
export const MAX_SEARCH_TEXT_CHARS = 2_000_000
const CJK = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u
const CJK_RUN = /([\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]+)/u

/** Local, deterministic case/diacritic folding; originals are never replaced for display. */
export function normalizeSearchText(value: string): string {
  return value.slice(0, MAX_SEARCH_TEXT_CHARS).normalize('NFKD').replace(/\p{M}/gu, '').toLowerCase().replace(/ς/gu, 'σ')
}

function* searchParts(value: string): Generator<string> {
  for (const word of normalizeSearchText(String(value ?? '')).matchAll(/[\p{L}\p{N}_/-]+/gu)) {
    yield* word[0].split(CJK_RUN).filter(Boolean)
  }
}

export function searchTerms(value: string, maxTerms = 20_000): string[] {
  const terms = new Set<string>()
  for (const part of searchParts(value)) {
    if (CJK.test(part)) {
      const chars = Array.from(part)
      if (chars.length === 1) terms.add(chars[0])
      else for (let i = 0; i + 1 < chars.length && terms.size < maxTerms; i++) terms.add(chars[i] + chars[i + 1])
    } else if (part.length >= 2) terms.add(Array.from(part).slice(0, 64).join(''))
    if (terms.size >= maxTerms) return [...terms].slice(0, maxTerms)
  }
  return [...terms]
}

/** Index bounded prefixes at word and compound/path boundaries, plus internal CJK grams. */
export function indexSearchTokens(value: string, maxTokens: number): string[] {
  const tokens = new Set<string>()
  const seenWindows = new Set<string>()
  const prefixes = (part: string, start: number) => {
    // Slice before Array.from so long compounds do not repeatedly copy their full suffix.
    const chars = Array.from(part.slice(start, start + 128)).slice(0, 64)
    const window = chars.join('')
    if (seenWindows.has(window)) return
    seenWindows.add(window)
    for (let length = 2; length <= chars.length && tokens.size < maxTokens; length++) tokens.add(chars.slice(0, length).join(''))
  }
  for (const part of searchParts(value)) {
    if (CJK.test(part)) {
      const chars = Array.from(part)
      for (let index = 0; index < chars.length && tokens.size < maxTokens; index++) {
        tokens.add(chars[index])
        if (index + 1 < chars.length && tokens.size < maxTokens) tokens.add(chars[index] + chars[index + 1])
      }
    } else {
      prefixes(part, 0)
      for (const separator of part.matchAll(/[_/-]/gu)) {
        if (tokens.size >= maxTokens) break
        // Keep both '/folder/file' and 'folder/file' queries, without changing query OR semantics.
        prefixes(part, separator.index)
        prefixes(part, separator.index + 1)
      }
    }
    if (tokens.size >= maxTokens) break
  }
  return [...tokens].slice(0, maxTokens)
}

export function normalizedTermRanges(normalized: string, terms: string[], maxRanges = 64): ConversationSearchHighlight[] {
  const ranges: ConversationSearchHighlight[] = []
  for (const term of terms) {
    let at = normalized.indexOf(term)
    while (at >= 0) {
      // Compound/path separators are boundaries; ordinary Latin/Cyrillic infixes remain excluded.
      const previous = at > 0 ? Array.from(normalized.slice(Math.max(0, at - 2), at)).at(-1) ?? '' : ''
      if (CJK.test(term) || CJK.test(previous) || /^[_/-]/u.test(term) || !/[\p{L}\p{N}]/u.test(previous)) {
        ranges.push({ start: at, end: at + term.length })
        if (ranges.length >= maxRanges) return ranges.sort((a, b) => a.start - b.start)
      }
      at = normalized.indexOf(term, at + Math.max(1, term.length))
    }
  }
  return ranges.sort((a, b) => a.start - b.start)
}

export function matchingTerms(value: string, terms: string[]): string[] {
  const normalized = normalizeSearchText(value)
  return terms.filter((term) => normalizedTermRanges(normalized, [term], 1).length > 0)
}

/** Convert folded offsets back to original UTF-16 coordinates, including decomposed marks. */
export function originalSearchRanges(value: string, terms: string[]): ConversationSearchHighlight[] {
  let normalized = ''
  const starts: number[] = [], ends: number[] = []
  let offset = 0
  for (const char of value.slice(0, MAX_SEARCH_TEXT_CHARS)) {
    const folded = normalizeSearchText(char)
    if (!folded && ends.length) ends[ends.length - 1] = offset + char.length
    for (let i = 0; i < folded.length; i++) { starts.push(offset); ends.push(offset + char.length) }
    normalized += folded
    offset += char.length
  }
  const merged: ConversationSearchHighlight[] = []
  for (const range of normalizedTermRanges(normalized, terms)) {
    const original = { start: starts[range.start], end: ends[range.end - 1] }
    const previous = merged.at(-1)
    if (previous && original.start <= previous.end) previous.end = Math.max(previous.end, original.end)
    else merged.push(original)
  }
  return merged
}

export function searchExcerpt(value: string, terms: string[], maxLength = 240): { excerpt: string; highlights: ConversationSearchHighlight[] } {
  const ranges = originalSearchRanges(value, terms)
  let start = Math.max(0, (ranges[0]?.start ?? 0) - 60)
  if (start && /[\uDC00-\uDFFF]/u.test(value[start])) start--
  let end = Math.min(value.length, start + maxLength)
  if (end < value.length && /[\uDC00-\uDFFF]/u.test(value[end])) end--
  const prefix = start > 0 ? '…' : ''
  return {
    excerpt: prefix + value.slice(start, end) + (end < value.length ? '…' : ''),
    highlights: ranges.filter((range) => range.start < end && range.end > start).slice(0, 24)
      .map((range) => ({ start: Math.max(start, range.start) - start + prefix.length, end: Math.min(end, range.end) - start + prefix.length })),
  }
}
