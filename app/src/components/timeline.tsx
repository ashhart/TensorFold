import { MessageQuoteAction } from './composer/MessageQuotes'
import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import { memo, useEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { remarkCitationAutolinks } from '../lib/markdown-citations'
import { Icon, type IconName } from '../lib/icons'
import { FileGlyph } from '../lib/fileicon'
import { mdComponents } from '../lib/highlight'
import { remarkBoundedMath } from './reading/math-markdown'
import { useStore } from '../lib/store'
import { reasoningDisplayState } from '../lib/activity-grouping'
import { api } from '../lib/ipc'
import { speakText, speechSupported, stopSpeaking } from '../lib/voice'
import { DiffView } from './thread-panel/DiffView'
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
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true" className="animate-spin text-accent motion-reduce:animate-none">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2.5" strokeOpacity="0.2" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  )
}

export function StateGlyph({ state }: { state?: ToolState }) {
  if (state === 'running') return <Spinner />
  if (state === 'pending') return <Icon name="clock" size={14} className="text-ink-3" />
  if (state === 'awaiting') return <span className="flex h-3.5 w-3.5 items-center justify-center"><Icon name="shield" size={14} className="text-ink-2" /></span>
  if (state === 'ok') return <Icon name="check" size={15} className="text-ink-2" />
  if (state === 'error') return <Icon name="warning" size={14} className="text-danger" />
  if (state === 'denied') return <Icon name="x" size={14} className="text-ink-3" />
  return null
}

function CopyButton({ text, label }: { text: string; label?: string }) {
  const tx = useAuthoredText()
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
    <button type="button" onClick={copy} className="focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent inline-flex h-7 items-center gap-1.5 rounded-lg border border-line bg-card px-2 text-[13px] font-medium text-ink-2 shadow-card transition-colors hover:bg-ink-5 hover:text-ink">
      <Icon name={copied ? 'check' : 'copy'} size={12} />{copied ? tx("Copied") : label ?? tx("Copy")}
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

export function UserBubble({ text, attachments, contentRef }: { text: string; attachments?: Attachment[]; contentRef?: React.Ref<HTMLDivElement> }) {
  return (
    <div className="flex w-full flex-col items-end gap-1.5 fade-in">
      <AttachmentChips items={attachments} />
      {text && (
        <div ref={contentRef} className="max-w-[82%] select-text whitespace-pre-wrap break-words rounded-[0.8rem] bg-bubble-out px-3.5 py-2 text-[14px] leading-relaxed text-bubble-out-ink">
          {text}
        </div>
      )}
    </div>
  )
}

export function CueBubble({ text, state }: { text: string; state?: ToolState }) {
  const tx = useAuthoredText()
  const applied = state === 'ok'
  return (
    <div className="flex justify-end fade-in">
      <div className={clsx('max-w-[82%] rounded-[18px] rounded-br-md border px-4 py-2.5 text-[14px] leading-relaxed',
        applied ? 'border-line bg-card text-ink-2' : 'border-accent/30 bg-accent-soft text-ink')}>
        <div className="mb-1 flex items-center justify-end gap-1.5 text-[13px] font-medium text-ink-3">
          <Icon name="spark" size={11} className={applied ? 'text-ink-3' : 'text-accent'} />
          <span>{tx("Guide")}</span>
          <span className="font-normal">{applied ? tx("applied") : tx("queued")}</span>
        </div>
        <div className="whitespace-pre-wrap">{text}</div>
      </div>
    </div>
  )
}

export const AssistantMessage = memo(function AssistantMessage({ text, active, meta, footer, onQuote }: { text: string; active?: boolean; meta?: string; footer?: React.ReactNode; onQuote?(selection?: string): void }) {
  const tx = useAuthoredText()
  const contentRef = useRef<HTMLDivElement>(null)
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
      <div ref={contentRef}>{segments.map((segment, i) => (
        segment.type === 'markdown'
          ? <div key={i} className={simple ? undefined : 'prose-tf'}><ReactMarkdown remarkPlugins={[remarkGfm, remarkCitationAutolinks, remarkBoundedMath]} components={mdComponents}>{segment.text}</ReactMarkdown></div>
          : <VisualizationCard key={i} spec={segment.spec} />
      ))}</div>
      {!active && meta && !isEstimatedThroughputMeta(meta) && <div className="mt-1.5 text-[13px] leading-4 text-ink-4">{meta}</div>}
      {!active && footer && text.trim().length === 0 && <div className="mt-2">{footer}</div>}
      {!active && text.trim().length > 0 && (
        <div className="mt-2 flex min-h-7 items-center gap-1">
          {footer && <div className="mr-2 min-w-0 flex-1">{footer}</div>}
          <div className="flex shrink-0 items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
          <CopyButton text={text} />
          {onQuote && <MessageQuoteAction contentRef={contentRef} onQuote={onQuote} label={tx('Quote assistant message')} />}
          {speaking ? (
            <button onClick={stop} className="inline-flex h-7 items-center gap-1.5 rounded-lg border border-line bg-card px-2 text-[13px] font-medium text-ink-2 shadow-card transition-colors hover:bg-ink-5 hover:text-ink">
              <Icon name="stop" size={12} />{tx("Stop")}</button>
          ) : canSpeak ? (
            <>
              <button onClick={() => read(false)} className="inline-flex h-7 items-center gap-1.5 rounded-lg border border-line bg-card px-2 text-[13px] font-medium text-ink-2 shadow-card transition-colors hover:bg-ink-5 hover:text-ink">
                <Icon name="speaker" size={12} />{tx("Read")}</button>
              <button onClick={() => read(true)} className="inline-flex h-7 items-center gap-1.5 rounded-lg border border-line bg-card px-2 text-[13px] font-medium text-ink-2 shadow-card transition-colors hover:bg-ink-5 hover:text-ink">
                <Icon name="spark" size={12} />{tx("Brief")}</button>
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
  const tx = useAuthoredText()
  const [open, setOpen] = useState(false)
  const { expanded } = reasoningDisplayState({ open, active, forceExpanded })
  return (
    <div className="fade-in">
      <button onClick={() => setOpen((o) => !o)} aria-expanded={expanded} className="group flex items-center gap-1.5 text-[13px] text-ink-3 transition-colors hover:text-ink-2">
        <Icon name="spark" size={13} className={active ? 'text-accent' : 'text-ink-3'} />
        <span className={active ? 'shimmer font-medium' : 'font-medium'}>{active ? tx("Thinking") : tx("Thought process")}</span>
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
  const tx = useAuthoredText()
  const setView = useStore((s) => s.setView)
  const recovery = runRecovery(text)
  return (
    <div role="alert" className="fade-in flex items-start gap-2.5 rounded-xl border border-danger/25 bg-danger/8 px-3.5 py-3 text-[13px] text-ink">
      <Icon name="warning" size={16} className="mt-0.5 shrink-0 text-danger" />
      <div className="min-w-0 flex-1">
        <div className="font-medium text-ink">{recovery.title}</div>
        <p className="mt-1 leading-relaxed text-ink-2">{recovery.guidance}</p>
        <details className="mt-2 text-ink-3">
          <summary className="w-fit cursor-pointer rounded-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent">{tx("Technical details")}</summary>
          <div className="mt-1.5 max-h-56 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-card p-2 font-mono text-[12px]">{text}</div>
        </details>
        <div className="mt-3 flex flex-wrap gap-2">
          {recovery.destination && <button type="button" className="inline-flex h-7 items-center rounded-lg border border-line bg-card px-2 text-[13px] font-medium text-ink-2 transition-colors hover:bg-ink-5" onClick={() => {
            if (recovery.destination === 'models') setView({ name: 'models' })
            else setView({ name: 'settings', section: recovery.destination === 'security' ? 'security' : 'models' })
          }}>{recovery.action}</button>}
          <CopyButton text={text} label={tx("Copy details")} />
        </div>
      </div>
    </div>
  )
}

// Tool headers retain a readable target and state while long output opens on demand.
export function ToolCard({ item, foldDiff, forceExpanded = false }: { item: TimelineItem; foldDiff?: boolean; forceExpanded?: boolean }) {
  const tx = useAuthoredText()
  const isDiff = !!item.diff?.length
  const isCmd = item.tool === 'run_command'
  // Diffs (the edits you care about) open by default unless a turn's edits card already
  // lists them; noisy command output stays folded until clicked.
  const [open, setOpen] = useState(item.state === 'error' || (isDiff && !foldDiff))
  useEffect(() => { if (item.state === 'error') setOpen(true) }, [item.state])
  const effectiveOpen = open || forceExpanded
  const resolveAppAction = useStore((s) => s.resolveAppAction)
  const view = useStore((s) => s.view)
  const threadId = view.name === 'thread' ? view.id : ''
  const settingProposal = item.appAction?.kind === 'change_setting' && item.appAction.payload?.key && item.appAction.payload?.value ? item.appAction : null

  const icon = toolIcon(item.tool)
  const hasBody = isDiff || !!item.output?.trim() || !!(isCmd && item.command)
  const stateLabel = item.state ? { pending: 'Queued', awaiting: 'Needs approval', running: 'Running', ok: 'Completed', error: 'Failed', denied: 'Denied' }[item.state] : ''
  const bodyId = `tool-output-${item.id}`

  return (
    <div className="fade-in">
      <button
        type="button"
        onClick={() => hasBody && setOpen((o) => !o)}
        disabled={!hasBody}
        aria-expanded={hasBody ? effectiveOpen : undefined}
        aria-controls={hasBody ? bodyId : undefined}
        title={[item.title, item.command || item.path].filter(Boolean).join(': ')}
        className="group flex w-full items-center gap-2 rounded-lg px-1.5 py-1.5 text-left transition-colors hover:bg-surface-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent disabled:cursor-default"
      >
        {FILE_TOOLS.has(item.tool || '') && item.path
          ? <FileGlyph path={item.path} size={14} />
          : <Icon name={icon} size={14} className="shrink-0 text-ink-3" />}
        <span className="min-w-0 flex-1">
          <span className={clsx('block text-[13px] font-medium', item.state === 'error' ? 'text-danger' : 'text-ink-2')}>{item.title}</span>
          {(item.command || item.path) && <code className="block truncate font-mono text-[12px] text-ink-3">{item.command || item.path}</code>}
        </span>
        <span className="shrink-0 text-right">
          <span className={clsx('flex items-center justify-end gap-1 text-[11px]', item.state === 'error' ? 'text-danger' : item.state === 'running' || item.state === 'awaiting' ? 'text-accent' : 'text-ink-3')}>
            <StateGlyph state={item.state} />{tx(stateLabel)}
          </span>
          {shortToolMeta(item.meta) && <span title={shortToolMeta(item.meta)} className="block max-w-40 truncate font-mono text-[11px] text-ink-4">{shortToolMeta(item.meta)}</span>}
        </span>
        {hasBody && <Icon name={effectiveOpen ? 'chevronDown' : 'chevronRight'} size={14} className="shrink-0 text-ink-3" />}
      </button>

      {item.approval && item.state !== 'awaiting' && (
        <div className="mt-0.5 flex items-start gap-1.5 pl-[22px] text-[13px] leading-relaxed text-ink-3">
          <Icon name="shield" size={12} className="mt-0.5 shrink-0 text-ink-3" />
          <span>{item.approval}</span>
        </div>
      )}

      {effectiveOpen && hasBody && <div id={bodyId} className="mt-1.5"><ToolContent item={item} /></div>}

      {/* #42 — confirm-first chip for a chat-requested setting change. */}
      {settingProposal && (!settingProposal.state || settingProposal.state === 'pending') && threadId && (
        <div className="mt-1.5 flex flex-wrap items-center gap-2 rounded-xl border border-line-strong bg-card px-3 py-2.5 shadow-card">
          <Icon name="settings" size={14} className="text-ink-2" />
          <span className="flex-1 text-[13px] text-ink-2">{tx("Set ")}<span className="font-medium text-ink">{String(settingProposal.payload?.key)}</span> {tx("to ")}<span className="font-mono text-ink">{String(settingProposal.payload?.value)}</span>?</span>
          <button onClick={() => resolveAppAction(threadId, item.id, 'dismiss')} className="rounded-lg px-2.5 py-1.5 text-[13px] font-medium text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink">{tx("Dismiss")}</button>
          <button onClick={() => resolveAppAction(threadId, item.id, 'apply')} className="rounded-lg bg-accent px-3 py-1.5 text-[13px] font-medium text-accent-ink transition-colors hover:bg-accent-strong">{tx("Apply")}</button>
        </div>
      )}
      {settingProposal && (settingProposal.state === 'applied' || settingProposal.state === 'dismissed') && (
        <div className="ml-[22px] mt-1 text-[12px] text-ink-4">{settingProposal.state === 'applied' ? tx("Applied.") : tx("Dismissed.")}</div>
      )}
    </div>
  )
}

function ToolContent({ item }: { item: TimelineItem }) {
  const tx = useAuthoredText()
  const [wrap, setWrap] = useState(true)
  const [expanded, setExpanded] = useState(false)
  const isDiff = !!item.diff?.length
  const isCmd = item.tool === 'run_command'
  const output = `${isCmd && item.command ? `$ ${item.command}\n` : ''}${item.output ?? ''}`
  const lines = isDiff ? item.diff!.length : output.split('\n').length
  const controlClass = 'rounded-md px-2 py-1 text-[12px] transition-colors hover:bg-surface-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent'
  return (
    <div className="min-w-0 overflow-hidden rounded-lg border border-line bg-card-2">
      <div className="flex flex-wrap items-center gap-1 border-b border-line bg-card px-2 py-1.5">
        <span className="mr-auto px-1 text-[12px] text-ink-3">{tx(isDiff ? 'Diff' : 'Output')} · {tx('Lines: {count}', { count: lines })}</span>
        <button type="button" aria-pressed={wrap} onClick={() => setWrap(!wrap)} className={clsx(controlClass, wrap ? 'bg-accent-soft text-accent' : 'text-ink-3')}>{tx('Wrap lines')}</button>
        {lines > 12 && <button type="button" aria-expanded={expanded} onClick={() => setExpanded(!expanded)} className={clsx(controlClass, 'text-ink-3')}>{tx(expanded ? 'Collapse output' : 'Expand output')}</button>}
        <CopyButton text={isDiff ? diffText(item.diff!) : output} label={tx(isDiff ? 'Copy diff' : 'Copy output')} />
      </div>
      <div className={clsx('scroll-thin overflow-auto', expanded ? 'max-h-[min(60vh,640px)]' : 'max-h-[280px]')}>
        {isDiff ? <DiffView diff={item.diff!} wrap={wrap} label={tx('Tool diff')} /> : (
          <pre tabIndex={0} aria-label={tx('Tool output')} className={clsx('min-w-0 px-3.5 py-2.5 font-mono text-[12px] leading-5 text-ink-2 focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-accent', wrap ? 'whitespace-pre-wrap [overflow-wrap:anywhere]' : 'whitespace-pre')}>
            {output}{!item.output && <span className="text-ink-3">{tx(item.state === 'running' || item.state === 'pending' ? 'Waiting for output…' : 'No text output.')}</span>}
          </pre>
        )}
      </div>
      {(item.artifacts || (item.meta && item.meta !== shortToolMeta(item.meta))) && (
        <div className="border-t border-line px-3 py-1.5">
          {item.artifacts ? <ArtifactLine artifacts={item.artifacts} /> : <div className="break-words font-mono text-[11px] text-ink-3">{item.meta}</div>}
        </div>
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
