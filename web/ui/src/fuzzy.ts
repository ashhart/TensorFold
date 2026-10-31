// Forgiving matching for the palette: case and accents ignored, each word of the query found in the label as a
// piece of it, as its letters in order ("bnch" finds Bench), or with one typo ("nemtoron" finds Nemotron).
// A higher score is a better match; null is no match.

const norm = (s: string) => s.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase()

/** Damerau-Levenshtein distance, stopping early once it exceeds ``max``. */
function within(a: string, b: string, max: number): boolean {
  if (Math.abs(a.length - b.length) > max) return false
  const d = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array(b.length).fill(0)])
  for (let j = 1; j <= b.length; j++) d[0][j] = j
  for (let i = 1; i <= a.length; i++) {
    let best = Infinity
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + 1)
      best = Math.min(best, d[i][j])
    }
    if (best > max) return false
  }
  return d[a.length][b.length] <= max
}

function wordScore(w: string, text: string, words: string[]): number | null {
  const at = text.indexOf(w)
  if (at >= 0) return 100 - Math.min(at, 40) + (at === 0 || /[\s\-_./:(]/.test(text[at - 1]) ? 30 : 0)
  // the letters in order: fewer and shorter gaps score higher
  let i = 0, gaps = 0, last = -1
  for (let k = 0; k < text.length && i < w.length; k++) {
    if (text[k] === w[i]) { if (last >= 0 && k > last + 1) gaps += 1; last = k; i += 1 }
  }
  if (i === w.length && w.length >= 2) return 60 - gaps * 8
  if (w.length >= 4) {
    for (const t of words) {
      if (within(w, t.slice(0, w.length + 1), 1) || within(w, t, 1)) return 35
    }
  }
  return null
}

export function score(query: string, label: string): number | null {
  const text = norm(label)
  const words = text.split(/[\s\-_./:()]+/).filter(Boolean)
  let total = 0
  for (const w of norm(query).split(/\s+/).filter(Boolean)) {
    const s = wordScore(w, text, words)
    if (s == null) return null
    total += s
  }
  return total - text.length * 0.05              // a shorter label wins a tie
}

/** The items matching ``query``, best first; all of them, in their order, for an empty query. */
export function rank<T>(query: string, items: T[], label: (t: T) => string): T[] {
  if (!query.trim()) return items
  return items.map((t, i) => ({ t, i, s: score(query, label(t)) }))
    .filter((x): x is { t: T; i: number; s: number } => x.s != null)
    .sort((a, b) => b.s - a.s || a.i - b.i).map((x) => x.t)
}
