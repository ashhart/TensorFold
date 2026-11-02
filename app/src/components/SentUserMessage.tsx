import { MessageQuoteAction, SentMessageQuotes } from './composer/MessageQuotes'
import { currentIntlLocale } from '../lib/preferences/current-locale'
import { OriginalDiagnostic } from './OriginalDiagnostic'
import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import { useRef, useState } from 'react'
import { api } from '../lib/ipc'
import { useStore } from '../lib/store'
import { Icon } from '../lib/icons'
import type { Thread, TimelineItem } from '../lib/types'
import type { ThreadMessageEditPreparation } from '../../shared/thread-message-edit'
import { UserBubble } from './timeline'

export function SentUserMessage({ thread, item, busy, onQuote }: { thread: Pick<Thread, 'id' | 'orchestratorHostId'>; item: TimelineItem; busy: boolean; onQuote?(selection?: string): void }) {
  const tx = useAuthoredText()
  const contentRef = useRef<HTMLDivElement>(null)
  const action = useRef<'prepare' | 'save' | null>(null)
  const [copyState, setCopyState] = useState<'idle' | 'copied'>('idle')
  const [preparation, setPreparation] = useState<ThreadMessageEditPreparation | null>(null)
  const [text, setText] = useState(item.text ?? '')
  const [pending, setPending] = useState<'prepare' | 'save' | null>(null)
  const [error, setError] = useState('')
  const unsupported = thread.orchestratorHostId ? tx("Edit this conversation on its owning host.")
      : item.memorySource === 'external' ? tx("Messages received from an integration retain their original source.")
        : ''
  const copy = async () => {
    setError('')
    try {
      let copied = false
      try { copied = (await api.writeClipboard(item.text ?? ''))?.ok !== false } catch { /* browser fallback below */ }
      if (!copied) {
        if (!navigator.clipboard?.writeText) throw new Error(tx("Clipboard access is unavailable. Select the message text and copy it with your keyboard."))
        await navigator.clipboard.writeText(item.text ?? '')
      }
      setCopyState('copied')
      window.setTimeout(() => setCopyState('idle'), 1200)
    } catch (cause) { setError(cause instanceof Error ? cause.message : tx("Could not copy this message. Select its text and copy it with your keyboard.")) }
  }
  const edit = async () => {
    if (action.current || busy) return
    setError('')
    if (unsupported) { setError(unsupported); return }
    action.current = 'prepare'; setPending('prepare')
    try {
      const prepared = await api.prepareMessageEdit({ threadId: thread.id, itemId: item.id })
      setText(prepared.expectedMessageText)
      setPreparation(prepared)
    } catch (cause) { setError(cause instanceof Error ? cause.message : tx("Could not prepare this message for editing.")) }
    finally { action.current = null; setPending(null) }
  }
  const save = async () => {
    if (!preparation || action.current || busy) return
    action.current = 'save'; setPending('save'); setError('')
    try {
      const updated = await api.editMessage({ threadId: thread.id, itemId: preparation.itemId, expectedConversationRevision: preparation.expectedConversationRevision, expectedHistoryToken: preparation.expectedHistoryToken, expectedMessageText: preparation.expectedMessageText, text })
      // The host publishes the same authoritative epoch; accepting the return value is
      // idempotent and also works when an event is delayed behind the IPC response.
      useStore.getState().applyThreadUpdate(updated)
      if (updated.id !== thread.id) useStore.getState().setView({ name: 'thread', id: updated.id })
      setPreparation(null)
    } catch (cause) { setError(cause instanceof Error ? cause.message : tx("Could not save the edited message. Your current conversation is unchanged.")) }
    finally { action.current = null; setPending(null) }
  }
  const cancel = () => { if (action.current === 'save') return; setPreparation(null); setText(item.text ?? ''); setError('') }
  return <div className="group/user flex flex-col items-end gap-2">
    {preparation ? <div className="w-full max-w-[min(100%,42rem)] rounded-xl border border-accent/40 bg-card p-3">
      <label className="mb-2 block text-[13px] font-medium text-ink" htmlFor={`edit-user-${item.id}`}>{tx("Edit sent message")}</label>
      <textarea id={`edit-user-${item.id}`} aria-label={tx("Edit sent message")} autoFocus value={text} disabled={pending === 'save'} onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => { if (event.nativeEvent?.isComposing || event.keyCode === 229) return; if (event.key === 'Escape') { event.preventDefault(); cancel() } if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') { event.preventDefault(); void save() } }}
        className="min-h-28 w-full resize-y select-text rounded-lg border border-line bg-canvas p-3 text-[14px] leading-relaxed text-ink outline-none focus:border-accent focus:ring-1 focus:ring-accent disabled:opacity-60" />
      <p className="mt-2 text-[12px] leading-relaxed text-ink-2">{preparation.mode === 'branch-task' ? <>{tx("Saving creates an edited branch with its own history and future checkpoints; the original conversation and all ")}{preparation.affectedItems} {tx("later items remain unchanged.")}</> : <>{tx("Saving keeps ")}{preparation.affectedItems} {tx("later conversation ")}{preparation.affectedItems === 1 ? tx("item") : tx("items")} {tx("in Earlier revisions and removes them from future model context.")}</>} {tx("No new request is sent.")}</p>
      <p className="mt-1 text-[12px] leading-relaxed text-ink-2">{preparation.mode === 'branch-task' ? tx("The branch shares the current workspace files. ") : ''}{tx("File changes, saved memory, sent messages and other external actions are not undone.")}</p>
      {item.attachments?.length ? <p className="mt-1 text-[12px] text-ink-2">{tx("The original ")}{item.attachments.length === 1 ? tx("attachment stays") : tx("attachments stay")} {tx("with this message.")}</p> : null}
      {busy && <p role="status" className="mt-2 text-[12px] text-ink-2">{tx("Stop the active run before saving this edit.")}</p>}
      <div className="mt-3 flex justify-end gap-2">
        <button type="button" onClick={cancel} disabled={pending === 'save'} className="rounded-lg border border-line px-3 py-1.5 text-[13px] text-ink-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-50">{tx("Cancel")}</button>
        <button type="button" onClick={() => void save()} disabled={!!pending || busy || text === preparation.expectedMessageText || (!text.trim() && !item.attachments?.length)} className="rounded-lg bg-accent px-3 py-1.5 text-[13px] font-medium text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-50">{pending === 'save' ? tx("Saving…") : preparation.mode === 'branch-task' ? tx("Save as edited branch") : tx("Save changes")}</button>
      </div>
    </div> : <>
      <UserBubble text={item.text ?? ''} attachments={item.attachments} contentRef={contentRef} />
      {!!item.quotedMessages?.length && <SentMessageQuotes quotes={item.quotedMessages} threadId={thread.id} />}
      <div className="flex flex-wrap items-center gap-1">
        {onQuote && !!item.text?.trim() && <MessageQuoteAction contentRef={contentRef} onQuote={onQuote} label={tx('Quote user message')} />}
        <button type="button" onClick={() => void copy()} aria-label={tx("Copy user message")} className="inline-flex h-7 items-center gap-1.5 rounded-lg px-2 text-[12px] text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"><Icon name={copyState === 'copied' ? 'check' : 'copy'} size={12} />{copyState === 'copied' ? tx("Copied") : tx("Copy")}</button>
        <button type="button" onClick={() => void edit()} disabled={busy || !!pending} title={unsupported || (busy ? tx("Stop the active run before editing a sent message.") : tx("Edit this message and preserve later messages as an earlier revision."))} aria-label={tx("Edit user message")} className="inline-flex h-7 items-center gap-1.5 rounded-lg px-2 text-[12px] text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-40"><Icon name="edit" size={12} />{pending === 'prepare' ? tx("Opening…") : tx("Edit")}</button>
      </div>
    </>}
    {error && <p role="alert" className="max-w-[42rem] text-[12px] leading-relaxed text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
  </div>
}

/** Review-only evidence; tool cards deliberately have no rerun/approval actions here. */
export function EarlierMessageRevisions({ thread }: { thread: Thread }) {
  const tx = useAuthoredText()
  const revisions = thread.messageEditRevisions ?? []
  if (!revisions.length) return null
  return <><EditedTaskSource thread={thread} /><details className="rounded-xl border border-line bg-card/60 p-3 text-[13px] text-ink-2">
    <summary className="cursor-pointer rounded-sm font-medium text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent">{tx("Earlier revisions (")}{revisions.length})</summary>
    <p className="mt-2 text-[12px] leading-relaxed text-ink-2">{tx("These messages are retained for review and excluded from the model’s active conversation; external actions were not undone.")}</p>
    <div className="mt-3 space-y-3">{[...revisions].reverse().map((revision) => <details key={revision.id} className="rounded-lg border border-line p-2.5">
      <summary className="cursor-pointer rounded-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent">{tx("Before edit on ")}{new Date(revision.editedAt).toLocaleString(currentIntlLocale())} · {revision.timeline.length} {tx("preserved ")}{revision.timeline.length === 1 ? tx("item") : tx("items")}</summary>
      <div className="mt-3 max-h-96 space-y-3 overflow-auto select-text">{revision.timeline.map((item) => <div key={item.id} className="border-l-2 border-line pl-3">
        <div className="text-[12px] font-medium text-ink-2">{item.kind === 'user' ? tx("You") : item.kind === 'assistant' ? tx("Assistant") : item.title || item.tool || item.kind}</div>
        <div className="mt-1 whitespace-pre-wrap break-words text-[13px] leading-relaxed text-ink">{item.text || item.output || item.command || (item.args ? JSON.stringify(item.args, null, 2) : '')}</div>
        {!!item.attachments?.length && <p className="mt-1 text-[12px] text-ink-3">{tx("Attachments: ")}{item.attachments.map((attachment) => attachment.name).join(', ')}</p>}
      </div>)}</div>
    </details>)}</div>
  </details></>
}

function EditedTaskSource({ thread }: { thread: Thread }) {
  const tx = useAuthoredText()
  const source = thread.messageEditSource
  const available = useStore((state) => !!source && state.threads.some((candidate) => candidate.id === source.threadId))
  if (!source) return null
  return <div role="status" className="mb-3 rounded-xl border border-accent/30 bg-accent-soft p-3 text-[13px] leading-relaxed text-ink">
    <p>{tx("This is an edited branch; the original conversation and checkpoints are unchanged. Current workspace files are shared and external actions were not undone.")}</p>
    {available && <button type="button" className="mt-2 rounded-lg border border-line bg-card px-2.5 py-1.5 text-[12px] font-medium text-ink-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent" onClick={() => useStore.getState().setView({ name: 'thread', id: source.threadId })}>{tx("Open original conversation")}</button>}
  </div>
}
