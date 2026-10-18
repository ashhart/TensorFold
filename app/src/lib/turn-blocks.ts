// One turn, one block: folds a thread timeline into per-turn rows for the conversation view.
// A turn runs from a user message to the next one. Its work (tool calls, reasoning, interim
// text, run bookkeeping) collapses into one row; anything that needs the user, the final
// answer and the files it edited stay visible. Pure so it is tested without a DOM.
import { basename } from './format'
import { authoredText } from './preferences/language-packs'
import { currentUiLanguage } from './preferences/current-locale'
import { normalizeLanguage } from '../../shared/languages'
import type { TimelineItem } from './types'

export interface FileEdit {
  path: string
  additions: number
  deletions: number
  /** False when a command changed the file and no line diff was recorded. */
  counted: boolean
}

export interface TurnSummary {
  key: string
  /** 1-based user turn, the number turn checkpoints use; 0 for items before the first message. */
  index: number
  work: TimelineItem[]
  steps: number
  startedAt: number | null
  endedAt: number | null
  answerId: string | null
  edits: FileEdit[]
  running: boolean
  questionPending?: boolean
}

export type TurnRow =
  | { type: 'item'; item: TimelineItem; turn: TurnSummary }
  | { type: 'work'; turn: TurnSummary }
  | { type: 'edits'; turn: TurnSummary }

const MACHINERY_TITLES = new Set(['Run contract', 'Run receipt', 'Guard', 'Verify-on-stop', 'Plan check', 'Open plan items'])
const MACHINERY_TEXT = [
  /^Compacting context/i,
  /^Context (too large|compacted|overflow)/i,
  /^Provider hiccup/i,
  /^Restarting turn to apply your guide/i,
  /^Caught the model repeating itself/i,
  /^The model hit its output limit/i,
  /^Goal not met yet/i,
  /^Run (step|time) budget reached/i,
]

/** Run bookkeeping that belongs in the collapsed details, not in the transcript. */
export function isMachineryNotice(item: TimelineItem): boolean {
  if (item.kind !== 'notice') return false
  if (item.runContract || item.runReceipt) return true
  if (item.title && MACHINERY_TITLES.has(item.title)) return true
  const text = (item.text ?? '').trim()
  return MACHINERY_TEXT.some((re) => re.test(text))
}

/** A tool item the user has to act on: an approval, a question, or a confirm-first setting. */
export function needsUser(item: TimelineItem): boolean {
  if (item.kind !== 'tool') return false
  if (item.state === 'awaiting') return true
  if ((item.tool === 'ask_user' || item.tool === 'ask_question') && item.state !== 'ok' && item.state !== 'error' && item.state !== 'denied') return true
  const action = item.appAction
  return action?.kind === 'change_setting' && (!action.state || action.state === 'pending')
}

const IN_FLIGHT = new Set(['pending', 'running', 'awaiting'])
const STEP_KINDS = new Set<TimelineItem['kind']>(['tool', 'visualization'])

function normalizePath(path: string): string {
  return path.trim().replace(/^\.\//, '')
}

/** Files a turn's tools changed, with line counts from their recorded diffs. */
export function turnFileEdits(items: TimelineItem[]): FileEdit[] {
  const byPath = new Map<string, FileEdit>()
  const touch = (raw: string): FileEdit | null => {
    const path = normalizePath(raw)
    if (!path) return null
    let entry = byPath.get(path)
    if (!entry) { entry = { path, additions: 0, deletions: 0, counted: false }; byPath.set(path, entry) }
    return entry
  }
  for (const item of items) {
    if (item.kind !== 'tool' || (item.state !== 'ok' && item.state !== 'error')) continue
    if (item.diff?.length && item.path) {
      const entry = touch(item.path)
      if (entry) {
        entry.counted = true
        for (const line of item.diff) {
          if (line.type === 'add') entry.additions++
          else if (line.type === 'del') entry.deletions++
        }
      }
    }
    for (const changed of item.changedPaths ?? []) touch(String(changed ?? ''))
  }
  return [...byPath.values()]
}

/** The turn's answer: its last assistant text with no tool call or reasoning after it. */
function answerIndex(items: TimelineItem[]): number {
  for (let i = items.length - 1; i >= 0; i--) {
    const kind = items[i].kind
    if (kind === 'assistant') return i
    if (kind === 'tool' || kind === 'reasoning') return -1
  }
  return -1
}

function timeBounds(items: TimelineItem[]): { startedAt: number | null; endedAt: number | null } {
  const receipt = items.find((item) => item.runReceipt)?.runReceipt
  if (receipt && receipt.finishedAt >= receipt.startedAt) return { startedAt: receipt.startedAt, endedAt: receipt.finishedAt }
  const times = items.map((item) => item.createdAt).filter((at): at is number => typeof at === 'number' && Number.isFinite(at))
  if (!times.length) return { startedAt: null, endedAt: null }
  return { startedAt: Math.min(...times), endedAt: Math.max(...times) }
}

function segments(timeline: TimelineItem[]): TimelineItem[][] {
  const out: TimelineItem[][] = [[]]
  for (const item of timeline) {
    if (item.kind === 'user') out.push([item])
    else out[out.length - 1].push(item)
  }
  return out[0].length ? out : out.slice(1)
}

export function buildTurnRows(timeline: TimelineItem[], opts: { running?: boolean } = {}): TurnRow[] {
  const rows: TurnRow[] = []
  const segs = segments(timeline)
  let users = 0
  segs.forEach((items, segIndex) => {
    const user = items[0]?.kind === 'user' ? items[0] : null
    if (user) users++
    const body = user ? items.slice(1) : items
    const running = !!opts.running && segIndex === segs.length - 1
    const answerAt = answerIndex(body)
    const work: TimelineItem[] = []
    const visible: TimelineItem[] = []
    body.forEach((item, i) => {
      const folds = (item.kind === 'assistant' && i !== answerAt)
        || item.kind === 'reasoning'
        || (item.kind === 'tool' && !needsUser(item) && !item.question)
        || isMachineryNotice(item)
      if (folds) work.push(item)
      else visible.push(item)
    })
    const bounds = timeBounds(items)
    const turn: TurnSummary = {
      key: user?.id ?? `pre-${body[0]?.id ?? segIndex}`,
      index: user ? users : 0,
      work,
      steps: body.filter((item) => STEP_KINDS.has(item.kind)).length,
      startedAt: user?.createdAt ?? bounds.startedAt,
      endedAt: bounds.endedAt,
      answerId: answerAt >= 0 ? body[answerAt].id : null,
      edits: turnFileEdits(body),
      running,
      questionPending: body.some((item) => item.question?.state === 'pending'),
    }
    if (user) rows.push({ type: 'item', item: user, turn })
    if (work.length || running) rows.push({ type: 'work', turn })
    let editsPlaced = !turn.edits.length
    for (const item of visible) {
      rows.push({ type: 'item', item, turn })
      if (item.id === turn.answerId && !editsPlaced && !running) { rows.push({ type: 'edits', turn }); editsPlaced = true }
    }
    if (!editsPlaced && !running) rows.push({ type: 'edits', turn })
  })
  return rows
}

/** Present-tense label for one in-flight work item. */
export function stepLabel(item: TimelineItem | undefined, locale: unknown = currentUiLanguage()): string {
  if (normalizeLanguage(locale) !== 'en') {
    const tx = (source: string, params = {}) => authoredText(locale, source, params)
    if (!item) return ''
    if (item.kind === 'reasoning') return tx('Thinking')
    if (item.kind === 'assistant') return tx('Writing')
    if (item.kind === 'visualization') return tx('Drawing a chart')
    const name = item.path ? basename(item.path) : ''
    switch (item.tool) {
      case 'edit_file': return name ? tx('Editing {name}', { name }) : tx('Editing')
      case 'write_file': return name ? tx('Creating {name}', { name }) : tx('Creating a file')
      case 'read_file': return name ? tx('Reading {name}', { name }) : tx('Reading')
      case 'run_command': return item.command ? tx('Running {command}', { command: item.command }) : tx('Running a command')
      case 'search': return tx('Searching the code')
      case 'list_dir': return tx('Looking through files')
      case 'web_search': return tx('Searching the web')
      case 'web_fetch': return tx('Reading a page')
      default: return item.title || tx('Working')
    }
  }
  if (!item) return ''
  if (item.kind === 'reasoning') return 'Thinking'
  if (item.kind === 'assistant') return 'Writing'
  if (item.kind === 'visualization') return 'Drawing a chart'
  const file = item.path ? basename(item.path) : ''
  switch (item.tool) {
    case 'edit_file': return file ? `Editing ${file}` : 'Editing'
    case 'write_file': return file ? `Creating ${file}` : 'Creating a file'
    case 'read_file': return file ? `Reading ${file}` : 'Reading'
    case 'run_command': return item.command ? `Running ${item.command}` : 'Running a command'
    case 'search': return 'Searching the code'
    case 'list_dir': return 'Looking through files'
    case 'web_search': return 'Searching the web'
    case 'web_fetch': return 'Reading a page'
    default: return item.title || 'Working'
  }
}

/** What the live row says it is doing right now. */
export function currentStepLabel(turn: TurnSummary, status?: string, locale: unknown = currentUiLanguage()): string {
  const tx = (source: string) => authoredText(locale, source)
  if (normalizeLanguage(locale) !== 'en') {
    if (status === 'awaiting') return tx(turn.questionPending ? 'Waiting for your answer' : 'Waiting for approval')
    const item = [...turn.work].reverse().find(item => item.kind === 'tool' && IN_FLIGHT.has(String(item.state ?? '')))
    if (item) return stepLabel(item, locale)
    if (status === 'thinking') return tx('Thinking')
    if (status === 'streaming') return tx('Writing')
    return stepLabel(turn.work[turn.work.length - 1], locale) || tx('Starting')
  }
  if (status === 'awaiting') return turn.questionPending ? 'Waiting for your answer' : 'Waiting for approval'
  const inFlight = [...turn.work].reverse().find((item) => item.kind === 'tool' && IN_FLIGHT.has(String(item.state ?? '')))
  if (inFlight) return stepLabel(inFlight)
  if (status === 'thinking') return 'Thinking'
  if (status === 'streaming') return 'Writing'
  return stepLabel(turn.work[turn.work.length - 1]) || 'Starting'
}

export function compactDuration(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const rs = s % 60
  if (m < 60) return rs ? `${m}m ${rs}s` : `${m}m`
  const h = Math.floor(m / 60)
  return m % 60 ? `${h}h ${m % 60}m` : `${h}h`
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`

/** "Worked for 12s · 4 steps" when done; "Working · 12s · Editing App.tsx" while live. */
export function workRowLabel(turn: TurnSummary, opts: { now: number; status?: string; phaseLabel?: string; locale?: string }): string {
  const locale = normalizeLanguage(opts.locale ?? currentUiLanguage())
  if (locale !== 'en') {
    const tx = (source: string, params = {}) => authoredText(locale, source, params)
    const duration = (ms: number) => new Intl.NumberFormat(locale, { style: 'unit', unit: 'second', unitDisplay: 'short', maximumFractionDigits: 0 }).format(Math.max(0, Math.round(ms / 1000)))
    if (turn.running) return [opts.phaseLabel || tx('Working'), turn.startedAt == null ? '' : duration(opts.now - turn.startedAt), opts.phaseLabel ? '' : currentStepLabel(turn, opts.status, locale)].filter(Boolean).join(' · ')
    const elapsed = turn.startedAt != null && turn.endedAt != null && turn.endedAt > turn.startedAt ? duration(turn.endedAt - turn.startedAt) : ''
    const thought = turn.steps === 0 && turn.work.some(item => item.kind === 'reasoning')
    const head = elapsed ? tx(thought ? 'Thought for {duration}' : 'Worked for {duration}', { duration: elapsed }) : tx(thought ? 'Thought' : 'Worked')
    return turn.steps > 0 ? `${head} · ${tx('Steps: {count}', { count: turn.steps })}` : head
  }
  if (turn.running) {
    const elapsed = turn.startedAt != null ? compactDuration(opts.now - turn.startedAt) : null
    if (opts.phaseLabel) return [opts.phaseLabel, elapsed].filter(Boolean).join(' · ')
    return ['Working', elapsed, currentStepLabel(turn, opts.status)].filter(Boolean).join(' · ')
  }
  const duration = turn.startedAt != null && turn.endedAt != null && turn.endedAt > turn.startedAt
    ? compactDuration(turn.endedAt - turn.startedAt)
    : null
  const onlyThought = turn.steps === 0 && turn.work.some((item) => item.kind === 'reasoning')
  const head = onlyThought
    ? (duration ? `Thought for ${duration}` : 'Thought')
    : (duration ? `Worked for ${duration}` : 'Worked')
  return turn.steps > 0 ? `${head} · ${plural(turn.steps, 'step')}` : head
}

export function editTotals(edits: FileEdit[]): { files: number; additions: number; deletions: number } {
  return edits.reduce((acc, edit) => ({ files: acc.files + 1, additions: acc.additions + edit.additions, deletions: acc.deletions + edit.deletions }), { files: 0, additions: 0, deletions: 0 })
}

/** Old saved threads carry a "~N tok/s" chars/4 estimate in meta; it is not shown any more. */
export function isEstimatedThroughputMeta(meta: string | undefined): boolean {
  return /^~\d+ tok\/s$/.test((meta ?? '').trim())
}

const SANDBOX_META = /^(sandbox=|artifacts=|redacted=|macos-sandbox-exec$|unsandboxed$)/

/** The tool line's short meta ("exit 0 · 240ms"); sandbox bookkeeping shows only when expanded. */
export function shortToolMeta(meta: string | undefined): string {
  return (meta ?? '').split(' · ').filter((part) => part && !SANDBOX_META.test(part.trim())).join(' · ')
}
