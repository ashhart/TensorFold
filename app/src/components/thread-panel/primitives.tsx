import type { ReactNode } from 'react'
import clsx from 'clsx'
import { useStore } from '../../lib/store'
import { FileGlyph } from '../../lib/fileicon'
import { basename } from '../../lib/format'
import type { Thread } from '../../lib/types'

const STATUS_TINT: Record<string, string> = { A: 'text-positive', M: 'text-warn', D: 'text-danger', R: 'text-accent', '?': 'text-ink-3' }

// Plain-language summary of what the agent is doing right now.
export function workingLine(status: string | undefined, thread: Thread): string {
  if (status === 'thinking') return 'Thinking…'
  if (status === 'streaming') return 'Writing a response…'
  if (status === 'awaiting') return 'Waiting for your approval'
  if (status === 'tool') {
    const last = [...thread.timeline].reverse().find((i) => i.kind === 'tool')
    const f = last?.path ? basename(last.path) : ''
    if (last?.tool === 'edit_file') return `Editing ${f}`
    if (last?.tool === 'write_file') return `Creating ${f}`
    if (last?.tool === 'read_file') return `Reading ${f}`
    if (last?.tool === 'run_command') return 'Running a command'
    if (last?.tool === 'search') return 'Searching the code'
    if (last?.tool === 'list_dir') return 'Looking through files'
    return 'Working…'
  }
  return 'Idle'
}

// What was produced (Outputs) vs consulted (Sources) — friendly, non-developer view.
export function deriveActivity(thread: Thread) {
  const outputs = new Map<string, 'Created' | 'Edited'>()
  const sources = new Set<string>()
  let commands = 0
  for (const it of thread.timeline) {
    if (it.kind !== 'tool') continue
    if (it.state && it.state !== 'ok') continue
    if (it.tool === 'write_file' && it.path) { if (!outputs.has(it.path)) outputs.set(it.path, 'Created') }
    else if (it.tool === 'edit_file' && it.path) { if (outputs.get(it.path) !== 'Created') outputs.set(it.path, 'Edited') }
    else if (it.tool === 'read_file' && it.path) { if (!outputs.has(it.path)) sources.add(it.path) }
    else if (it.tool === 'run_command') commands++
  }
  // A file that was edited is an Output, not also a Source.
  return { outputs: [...outputs.entries()].map(([path, verb]) => ({ path, verb })), sources: [...sources].filter((p) => !outputs.has(p)), commands }
}

export function FileRow({ path, tag, tagClass }: { path: string; tag?: string; tagClass?: string }) {
  const addFileMention = useStore((s) => s.addFileMention)
  return (
    <button onClick={() => addFileMention(path)} title={`${path} — click to reference in the composer`} className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-[13px] transition-colors hover:bg-ink-5">
      <FileGlyph path={path} size={15} />
      <span className="min-w-0 flex-1 truncate text-ink-2">{basename(path)}</span>
      {tag && <span className={clsx('shrink-0 text-[12px] font-medium', tagClass)}>{tag}</span>}
    </button>
  )
}

export function estTokens(thread: Thread): number {
  let chars = 0
  for (const it of thread.timeline) chars += (it.text?.length ?? 0) + (it.output?.length ?? 0)
  for (const m of thread.apiMessages) chars += (typeof m.content === 'string' ? m.content.length : 0)
  return Math.round(chars / 4)
}

export const kfmt = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(n >= 10_000 ? 0 : 1)}k` : String(n))

export function labelize(value: string): string {
  return value.replace(/_/g, ' ')
}

export function RunMiniStat({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="min-w-0 rounded-lg bg-card px-2 py-1.5" title={title ?? value}>
      <div className="text-[11px] font-semibold uppercase tracking-[0.06em] text-ink-4">{label}</div>
      <div className="mt-0.5 truncate text-[13px] font-medium text-ink-2">{value}</div>
    </div>
  )
}

// Panel section: sentence-case 12px muted label, dense body.
export function PanelSection({ title, right, children }: { title: string; right?: ReactNode; children: ReactNode }) {
  return (
    <div className="px-3.5 py-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[13px] font-normal text-ink-3">{title}</span>
        {right}
      </div>
      {children}
    </div>
  )
}

export { STATUS_TINT }
