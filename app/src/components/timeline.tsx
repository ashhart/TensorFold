import { memo, useMemo, useState } from 'react'
import clsx from 'clsx'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { remarkCitationAutolinks } from '../lib/markdown-citations'
import { Icon, type IconName } from '../lib/icons'
import { FileGlyph } from '../lib/fileicon'
import { mdComponents } from '../lib/highlight'
import { useStore } from '../lib/store'
import { reasoningDisplayState } from '../lib/activity-grouping'
import { api } from '../lib/ipc'
import { speakText, speechSupported, stopSpeaking } from '../lib/voice'
import { pairWordDiffs, type WordSpan } from '../lib/diff-words'
import { VisualizationCard, splitVisualizationMarkdown } from './visualization'
import { isEstimatedThroughputMeta, shortToolMeta } from '../lib/turn-blocks'
import type { Attachment, DiffLine, TimelineItem, ToolArtifactSummary, ToolState } from '../lib/types'
import { AttachmentChips } from './AttachmentChips'
import { runRecovery } from '../lib/run-recovery'

const FILE_TOOLS = new Set(['read_file', 'write_file', 'edit_file'])

export const TOOL_ICON: Record<string, IconName> = {
  list_dir: 'list', read_file: 'file', write_file: 'file', edit_file: 'edit',
  search: 'search', web_search: 'globe', web_fetch: 'globe', run_command: 'terminal',
}

// Best-effort icon for a tool (shared with the grouped activity line).
export function toolIcon(tool?: string): IconName {
  return TOOL_ICON[tool || ''] || 'terminal'
}

export function Spinner({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className="animate-spin text-accent">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2.5" strokeOpacity="0.2" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  )
}

export function StateGlyph({ state }: { state?: ToolState }) {
  if (state === 'running' || state === 'pending') return <Spinner />
  if (state === 'awaiting') return <span className="flex h-3.5 w-3.5 items-center justify-center"><Icon name="shield" size={14} className="text-ink-2" /></span>
  if (state === 'ok') return <Icon name="check" size={15} className="text-ink-2" />
  if (state === 'error') return <Icon name="warning" size={14} className="text-danger" />
  if (state === 'denied') return <Icon name="x" size={14} className="text-ink-3" />
  return null
}

function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      const res = await api.writeClipboard(text)
      if (res && res.ok === false) return
    } catch {
      return // clipboard denied/unavailable — degrade quietly rather than throw an unhandled rejection
    }
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1200)
  }
  return (
    <button onClick={copy} className="inline-flex h-7 items-center gap-1.5 rounded-lg border border-line bg-card px-2 text-[13px] font-medium text-ink-2 shadow-card transition-colors hover:bg-ink-5 hover:text-ink">
      <Icon name={copied ? 'check' : 'copy'} size={12} />{copied ? 'Copied' : label}
    </button>
  )
}

function diffText(diff: DiffLine[]): string {
  return diff.map((line) => {
    if (line.type === 'hunk') return line.text
    if (line.type === 'add') return `+${line.text}`
    if (line.type === 'del') return `-${line.text}`
    return ` ${line.text || ' '}`
  }).join('\n')
}

export function UserBubble({ text, attachments }: { text: string; attachments?: Attachment[] }) {
  return (
    <div className="flex flex-col items-end gap-1.5 fade-in">
      <AttachmentChips items={attachments} />
      {text && (
        <div className="max-w-[82%] whitespace-pre-wrap rounded-[0.8rem] bg-bubble-out px-3.5 py-2 text-[14px] leading-relaxed text-bubble-out-ink">
          {text}
        </div>
      )}
    </div>
  )
}

export function CueBubble({ text, state }: { text: string; state?: ToolState }) {
  const applied = state === 'ok'
  return (
    <div className="flex justify-end fade-in">
      <div className={clsx('max-w-[82%] rounded-[18px] rounded-br-md border px-4 py-2.5 text-[14px] leading-relaxed',
        applied ? 'border-line bg-card text-ink-2' : 'border-accent/30 bg-accent-soft text-ink')}>
        <div className="mb-1 flex items-center justify-end gap-1.5 text-[13px] font-medium text-ink-3">
          <Icon name="spark" size={11} className={applied ? 'text-ink-3' : 'text-accent'} />
          <span>Guide</span>
          <span className="font-normal">{applied ? 'applied' : 'queued'}</span>
        </div>
        <div className="whitespace-pre-wrap">{text}</div>
      </div>
    </div>
  )
}

export const AssistantMessage = memo(function AssistantMessage({ text, active, meta, footer }: { text: string; active?: boolean; meta?: string; footer?: React.ReactNode }) {
  const voice = useStore((s) => s.settings?.voice)
  const [speaking, setSpeaking] = useState(false)
  const segments = splitVisualizationMarkdown(text || '')
  const simple = segments.length === 1 && segments[0].type === 'markdown'
  const canSpeak = !!voice?.enabled && !!voice.ttsEnabled && speechSupported() && text.trim().length > 0
  const read = (summarize: boolean) => {
    setSpeaking(true)
    const started = speakText(text, voice, {
      summarize,
      onEnd: () => setSpeaking(false),
      onError: () => setSpeaking(false),
    })
    if (!started) setSpeaking(false)
  }
  const stop = () => { stopSpeaking(); setSpeaking(false) }
  return (
    <div className={clsx('group relative fade-in max-w-none', simple ? 'prose-tf' : 'space-y-4')}>
      {segments.map((segment, i) => (
        segment.type === 'markdown'
          ? <div key={i} className={simple ? undefined : 'prose-tf'}><ReactMarkdown remarkPlugins={[remarkGfm, remarkCitationAutolinks]} components={mdComponents}>{segment.text}</ReactMarkdown></div>
          : <VisualizationCard key={i} spec={segment.spec} />
      ))}
      {!active && meta && !isEstimatedThroughputMeta(meta) && <div className="mt-1.5 text-[13px] leading-4 text-ink-4">{meta}</div>}
      {!active && footer && text.trim().length === 0 && <div className="mt-2">{footer}</div>}
      {!active && text.trim().length > 0 && (
        <div className="mt-2 flex min-h-7 items-center gap-1">
          {footer && <div className="mr-2 min-w-0 flex-1">{footer}</div>}
          <div className="flex shrink-0 items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
          <CopyButton text={text} />
          {speaking ? (
            <button onClick={stop} className="inline-flex h-7 items-center gap-1.5 rounded-lg border border-line bg-card px-2 text-[13px] font-medium text-ink-2 shadow-card transition-colors hover:bg-ink-5 hover:text-ink">
              <Icon name="stop" size={12} />Stop
            </button>
          ) : canSpeak ? (
            <>
              <button onClick={() => read(false)} className="inline-flex h-7 items-center gap-1.5 rounded-lg border border-line bg-card px-2 text-[13px] font-medium text-ink-2 shadow-card transition-colors hover:bg-ink-5 hover:text-ink">
                <Icon name="speaker" size={12} />Read
              </button>
              <button onClick={() => read(true)} className="inline-flex h-7 items-center gap-1.5 rounded-lg border border-line bg-card px-2 text-[13px] font-medium text-ink-2 shadow-card transition-colors hover:bg-ink-5 hover:text-ink">
                <Icon name="spark" size={12} />Brief
              </button>
            </>
          ) : null}
          </div>
        </div>
      )}
      {active && <span className="caret" />}
    </div>
  )
})

// `forceExpanded` controls active-block behaviour (Task #34): when false the
// streaming reasoning collapses to a compact "Thinking" indicator until the
// user clicks. Undefined keeps the legacy always-expand-while-active behaviour.
export function Reasoning({ text, active, forceExpanded }: { text: string; active?: boolean; forceExpanded?: boolean }) {
  const [open, setOpen] = useState(false)
  const { expanded } = reasoningDisplayState({ open, active, forceExpanded })
  return (
    <div className="fade-in">
      <button onClick={() => setOpen((o) => !o)} aria-expanded={expanded} className="group flex items-center gap-1.5 text-[13px] text-ink-3 transition-colors hover:text-ink-2">
        <Icon name="spark" size={13} className={active ? 'text-accent' : 'text-ink-3'} />
        <span className={active ? 'shimmer font-medium' : 'font-medium'}>{active ? 'Thinking' : 'Thought process'}</span>
        <Icon name={expanded ? 'chevronDown' : 'chevronRight'} size={13} className="opacity-60" />
      </button>
      {expanded && (
        <div className="mt-2 border-l-2 border-line pl-3.5 text-[13px] leading-relaxed text-ink-3">
          <div className="whitespace-pre-wrap">{text}{active && <span className="caret" />}</div>
        </div>
      )}
    </div>
  )
}

export function ErrorBlock({ text }: { text: string }) {
  const setView = useStore((s) => s.setView)
  const recovery = runRecovery(text)
  return (
    <div role="alert" className="fade-in flex items-start gap-2.5 rounded-xl border border-danger/25 bg-danger/8 px-3.5 py-3 text-[13px] text-ink">
      <Icon name="warning" size={16} className="mt-0.5 shrink-0 text-danger" />
      <div className="min-w-0 flex-1">
        <div className="font-medium text-ink">{recovery.title}</div>
        <p className="mt-1 leading-relaxed text-ink-2">{recovery.guidance}</p>
        <details className="mt-2 text-ink-3">
          <summary className="w-fit cursor-pointer rounded-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent">Technical details</summary>
          <div className="mt-1.5 max-h-56 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-card p-2 font-mono text-[12px]">{text}</div>
        </details>
        <div className="mt-3 flex flex-wrap gap-2">
          {recovery.destination && <button type="button" className="inline-flex h-7 items-center rounded-lg border border-line bg-card px-2 text-[13px] font-medium text-ink-2 transition-colors hover:bg-ink-5" onClick={() => {
            if (recovery.destination === 'models') setView({ name: 'models' })
            else setView({ name: 'settings', section: recovery.destination === 'security' ? 'security' : 'models' })
          }}>{recovery.action}</button>}
          <CopyButton text={text} label="Copy details" />
        </div>
      </div>
    </div>
  )
}

function WordSpanText({ spans, kind }: { spans: WordSpan[]; kind: 'add' | 'del' }) {
  return (
    <>
      {spans.map((s, idx) =>
        s.changed ? (
          <mark
            key={idx}
            className={clsx(
              'rounded-[2px] px-0.5',
              kind === 'add' ? 'bg-[var(--add-ink)]/20 text-[var(--add-ink)]' : 'bg-[var(--del-ink)]/20 text-[var(--del-ink)]',
            )}
          >
            {s.text || ' '}
          </mark>
        ) : (
          <span key={idx}>{s.text}</span>
        ),
      )}
    </>
  )
}

function DiffView({ diff }: { diff: DiffLine[] }) {
  // Pair adjacent del+add lines so we can highlight the changed words within
  // each line, not just which lines changed. Computed once per diff render.
  const wordPairs = useMemo(() => pairWordDiffs(diff), [diff])
  return (
    <div className="scroll-thin overflow-x-auto bg-card-2 font-mono text-[13px] leading-[1.55]">
      {diff.map((l, i) => {
        if (l.type === 'hunk') return (
          <div key={i} className="select-none px-3 py-1 text-[12px] text-ink-3">{l.text}</div>
        )
        const wd = wordPairs.get(i)
        return (
          <div key={i} className={clsx('flex',
            l.type === 'add' && 'bg-[var(--add-bg)]',
            l.type === 'del' && 'bg-[var(--del-bg)]')}>
            <span className="w-10 shrink-0 select-none px-1 text-right text-ink-4">{l.oldNo ?? ''}</span>
            <span className="w-10 shrink-0 select-none px-1 text-right text-ink-4">{l.newNo ?? ''}</span>
            <span className={clsx('w-4 shrink-0 select-none text-center',
              l.type === 'add' && 'text-[var(--add-ink)]', l.type === 'del' && 'text-[var(--del-ink)]', l.type === 'ctx' && 'text-ink-4')}>
              {l.type === 'add' ? '+' : l.type === 'del' ? '−' : ''}
            </span>
            <span className={clsx('whitespace-pre pr-3',
              l.type === 'add' && 'text-[var(--add-ink)]', l.type === 'del' && 'text-[var(--del-ink)]', l.type === 'ctx' && 'text-ink-2')}>
              {wd && l.type === 'del' ? <WordSpanText spans={wd.removed} kind="del" /> :
               wd && l.type === 'add' ? <WordSpanText spans={wd.added} kind="add" /> :
               (l.text || ' ')}
            </span>
          </div>
        )
      })}
    </div>
  )
}

// A tool call reads as a single dim line — icon · title · target ·
// state — that expands to its diff/output inline. No card chrome; the transcript
// stays prose-first and the tool runs recede until you want them.
export function ToolCard({ item, foldDiff }: { item: TimelineItem; foldDiff?: boolean }) {
  const isDiff = !!item.diff?.length
  const isCmd = item.tool === 'run_command'
  // Diffs (the edits you care about) open by default unless a turn's edits card already
  // lists them; noisy command output stays folded until clicked.
  const [open, setOpen] = useState(isDiff && !foldDiff)
  const resolveAppAction = useStore((s) => s.resolveAppAction)
  const view = useStore((s) => s.view)
  const threadId = view.name === 'thread' ? view.id : ''
  const settingProposal = item.appAction?.kind === 'change_setting' && item.appAction.payload?.key && item.appAction.payload?.value ? item.appAction : null

  const icon = toolIcon(item.tool)
  const hasBody = isDiff || (item.output && item.output.trim().length > 0)

  return (
    <div className="fade-in">
      <button
        onClick={() => hasBody && setOpen((o) => !o)}
        className={clsx('group flex w-full items-center gap-2 py-[3px] text-left', !hasBody && 'cursor-default')}
      >
        {FILE_TOOLS.has(item.tool || '') && item.path
          ? <FileGlyph path={item.path} size={14} />
          : <Icon name={icon} size={14} className="shrink-0 text-ink-3" />}
        <span className={clsx('shrink-0 text-[13px]', item.state === 'error' ? 'text-danger' : 'text-ink-2')}>{item.title}</span>
        <code className="min-w-0 flex-1 truncate font-mono text-[13px] text-ink-3">{item.command || item.path}</code>
        {shortToolMeta(item.meta) && <span className="shrink-0 font-mono text-[12px] text-ink-4">{shortToolMeta(item.meta)}</span>}
        <span className="flex w-4 shrink-0 items-center justify-center"><StateGlyph state={item.state} /></span>
        {hasBody && <Icon name={open ? 'chevronDown' : 'chevronRight'} size={13} className="shrink-0 text-ink-4 opacity-0 transition-opacity group-hover:opacity-100" />}
      </button>

      {item.approval && item.state !== 'awaiting' && (
        <div className="mt-0.5 flex items-start gap-1.5 pl-[22px] text-[13px] leading-relaxed text-ink-3">
          <Icon name="shield" size={12} className="mt-0.5 shrink-0 text-ink-3" />
          <span>{item.approval}</span>
        </div>
      )}

      {open && hasBody && (
        // Opaque recessed surface (not a translucent tint): on the frosted glass
        // panels a see-through body lets the wallpaper bleed under the mono diff/
        // output text and muddies it, so the tool body stays a solid card-2.
        <div className="ml-[22px] mt-1.5 overflow-hidden rounded-lg border border-line bg-card-2">
          {isDiff ? (
            <>
              <DiffView diff={item.diff!} />
              <div className="flex justify-end border-t border-line bg-card px-2.5 py-1.5">
                <CopyButton text={diffText(item.diff!)} label="Copy diff" />
              </div>
            </>
          ) : (
            <>
              <pre className="scroll-thin max-h-[280px] overflow-auto px-3.5 py-2.5 font-mono text-[13px] leading-[1.5] text-ink-2">
                {isCmd && <span className="select-none text-ink-4">$ </span>}
                {isCmd && <span className="text-ink">{item.command}</span>}
                {isCmd && '\n'}
                {item.output}
              </pre>
              <div className="flex items-center gap-2 border-t border-line bg-card px-2.5 py-1.5">
                <div className="min-w-0 flex-1">
                  {item.artifacts ? <ArtifactLine artifacts={item.artifacts} /> : item.meta && item.meta !== shortToolMeta(item.meta) ? <div className="truncate font-mono text-[12px] text-ink-4">{item.meta}</div> : null}
                </div>
                <CopyButton text={`${isCmd && item.command ? `$ ${item.command}\n` : ''}${item.output ?? ''}`} />
              </div>
            </>
          )}
        </div>
      )}

      {/* #42 — confirm-first chip for a chat-requested setting change. */}
      {settingProposal && (!settingProposal.state || settingProposal.state === 'pending') && threadId && (
        <div className="ml-[22px] mt-1.5 flex items-center gap-2 rounded-xl border border-line-strong bg-card px-3 py-2.5 shadow-card">
          <Icon name="settings" size={14} className="text-ink-2" />
          <span className="flex-1 text-[13px] text-ink-2">Set <span className="font-medium text-ink">{String(settingProposal.payload?.key)}</span> to <span className="font-mono text-ink">{String(settingProposal.payload?.value)}</span>?</span>
          <button onClick={() => resolveAppAction(threadId, item.id, 'dismiss')} className="rounded-lg px-2.5 py-1.5 text-[13px] font-medium text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink">Dismiss</button>
          <button onClick={() => resolveAppAction(threadId, item.id, 'apply')} className="rounded-lg bg-accent px-3 py-1.5 text-[13px] font-medium text-accent-ink transition-colors hover:bg-accent-strong">Apply</button>
        </div>
      )}
      {settingProposal && (settingProposal.state === 'applied' || settingProposal.state === 'dismissed') && (
        <div className="ml-[22px] mt-1 text-[12px] text-ink-4">{settingProposal.state === 'applied' ? 'Applied.' : 'Dismissed.'}</div>
      )}
    </div>
  )
}

// What the sandbox captured for a command, shown only inside the expanded card.
function ArtifactLine({ artifacts }: { artifacts: ToolArtifactSummary }) {
  const count = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`
  const parts = [
    `Sandbox ${artifacts.profile}`,
    artifacts.outputFiles ? count(artifacts.outputFiles, 'output file') : '',
    artifacts.screenshots.length ? count(artifacts.screenshots.length, 'screenshot') : '',
    artifacts.diffs.length ? count(artifacts.diffs.length, 'changed artifact') : '',
    artifacts.patch ? 'patch saved' : '',
  ].filter(Boolean)
  const title = [`${artifacts.id} · ${artifacts.implementation}`, ...artifacts.diffs.map((diff) => `${diff.status} ${diff.path}`), ...artifacts.screenshots].join('\n')
  return <div title={title} className="truncate text-[13px] text-ink-4">{parts.join(' · ')}</div>
}

export { VisualizationCard }
