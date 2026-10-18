/** Detect duplicate object keys before JSON.parse discards their earlier values. */
export function strictBenchmarkJson(raw: string): unknown {
  let at = 0
  const bad = (): never => { throw new Error('The local benchmark receipt contains invalid JSON.') }
  const space = () => { while (/\s/.test(raw[at] ?? '') && at < raw.length) at++ }
  const string = (): string => {
    if (raw[at] !== '"') return bad()
    const start = at++
    while (at < raw.length) {
      if (raw[at] === '\\') { at += 2; continue }
      if (raw[at++] === '"') { try { return JSON.parse(raw.slice(start, at)) } catch { return bad() } }
    }
    return bad()
  }
  let nodes = 0
  const value = (depth: number): void => {
    if (depth > 24 || ++nodes > 10_000) bad()
    space()
    if (raw[at] === '{') {
      at++; space(); const keys = new Set<string>()
      if (raw[at] === '}') { at++; return }
      for (;;) {
        space(); const key = string()
        if (keys.has(key)) bad(); keys.add(key); space()
        if (raw[at++] !== ':') bad()
        value(depth + 1); space()
        const end = raw[at++]
        if (end === '}') return
        if (end !== ',') bad()
      }
    }
    if (raw[at] === '[') {
      at++; space()
      if (raw[at] === ']') { at++; return }
      for (;;) { value(depth + 1); space(); const end = raw[at++]; if (end === ']') return; if (end !== ',') bad() }
    }
    if (raw[at] === '"') { string(); return }
    const token = /^(?:true|false|null|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)/.exec(raw.slice(at))?.[0]
    if (!token) bad()
    at += token!.length
  }
  value(0); space()
  if (at !== raw.length) bad()
  try { return JSON.parse(raw) } catch { return bad() }
}
