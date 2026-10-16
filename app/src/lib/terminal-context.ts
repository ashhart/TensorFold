// Composer terminal-context injection (VST-039): turn a terminal's recent scrollback into a
// bounded, composer-insertable reference so the user can say "look at what just happened in
// this terminal" without pasting a wall of output. Pure — the renderer fetches the backlog
// and this shapes it (mirrors diff-view's buildAskWhyReference).

// Strip ANSI escape sequences (colour, cursor moves, OSC) so the reference reads as plain
// text — the model doesn't need the terminal's rendering codes, and they bloat the context.
// eslint-disable-next-line no-control-regex
const ANSI = /\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]/g

export function stripAnsi(text: string): string {
  return text.replace(ANSI, '')
}

export interface TerminalContextOptions {
  // Keep only the last N non-empty lines (the tail is what's relevant). Default 40.
  maxLines?: number
  // Hard cap on the fenced excerpt's characters, trimmed from the FRONT (recent wins). Default 4000.
  maxChars?: number
}

// The last `maxLines` meaningful lines of a terminal's scrollback, ANSI-stripped and
// bounded, as a fenced block. Returns '' when there's nothing worth referencing.
export function terminalTail(raw: string, options: TerminalContextOptions = {}): string {
  const maxLines = options.maxLines ?? 40
  const maxChars = options.maxChars ?? 4000
  const cleaned = stripAnsi(raw).replace(/\r\n/g, '\n').replace(/\r/g, '\n')
  const lines = cleaned.split('\n')
  // Drop trailing blank lines (a shell prompt often leaves one), keep interior blanks.
  while (lines.length && lines[lines.length - 1].trim() === '') lines.pop()
  const tail = lines.slice(-maxLines).join('\n')
  if (!tail.trim()) return ''
  return tail.length > maxChars ? `…${tail.slice(tail.length - maxChars + 1)}` : tail
}

// A composer-insertable reference naming the terminal and quoting its recent output. `label`
// is the terminal's title; `raw` is its backlog. Empty output → a short note instead of an
// empty fence, so the reference is never misleading.
export function buildTerminalReference(label: string, raw: string, options: TerminalContextOptions = {}): string {
  const name = (label || 'terminal').trim()
  const tail = terminalTail(raw, options)
  if (!tail) return `Regarding the “${name}” terminal (no recent output captured): `
  return `Regarding recent output from the “${name}” terminal:\n\n\`\`\`\n${tail}\n\`\`\`\n\n`
}
