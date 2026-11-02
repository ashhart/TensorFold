import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { DraftSaveIndicator } from '../composer/DraftSaveIndicator'
import { DraftPersistenceNotice, isSessionOnlyDraftNotice } from '../composer/DraftPersistenceNotice'
import { Button } from '../ui'
import { actionLabel } from './task-context'
import { sourceName } from './mappings'
import type { AssistantItem, SourceKey } from './types'
import type { UiPreferences } from '../../../shared/ui-preferences'
import { isComposerSendKey } from '../../lib/composer-keyboard'
import type { RefObject } from 'react'

export function Composer({ taskText, setTaskText, canDirectReply, selectedItem, selectedSource, quickActions, taskBusy, sendingReply, onSendReply, onStartTask, onDraftTask, draftKey, textareaRef, draftReady = true, draftStatus, draftProblem = false, onRetryDraft, sendKey, intent, onIntentChange, sessionOnly = false, contextLabel, reviewReply = false }: {
  reviewReply?: boolean
  contextLabel?: string
  sendKey?: UiPreferences['sendKey']
  taskText: string
  setTaskText: (text: string) => void
  canDirectReply: boolean
  selectedItem: AssistantItem | null
  selectedSource: SourceKey
  quickActions: string[]
  taskBusy: boolean
  sendingReply: boolean
  onSendReply: () => void
  onStartTask: () => void
  onDraftTask: (text: string) => void
  draftKey?: string
  textareaRef?: RefObject<HTMLTextAreaElement>
  draftReady?: boolean
  draftStatus?: string
  draftProblem?: boolean
  onRetryDraft?: () => void
  intent?: 'reply' | 'task'
  onIntentChange?: (intent: 'reply' | 'task') => void
  sessionOnly?: boolean
}) {
  const tx = useAuthoredText()
  const reply = canDirectReply && intent !== 'task'
  const problem = draftProblem && !isSessionOnlyDraftNotice(draftStatus ?? null, sessionOnly)
  const suggestions = [...new Map(quickActions.map((action) => [actionLabel(action), action])).entries()]
  const changeIntent = (next: 'reply' | 'task') => { onIntentChange?.(next); textareaRef?.current?.focus() }
  return (
    <section aria-label={tx('Conversation draft')} className="shrink-0 border-t border-line bg-card px-3 py-2 sm:px-5 md:py-3">
      <div className="mx-auto w-full max-w-[760px]">
        <div className="mb-2 flex flex-wrap items-center gap-2">
          {canDirectReply && onIntentChange ? <div aria-label={tx('Compose mode')} className="flex rounded-lg bg-surface p-0.5">
            <button type="button" aria-pressed={reply} onClick={() => changeIntent('reply')} className={`rounded-md px-2.5 py-1 text-[12px] focus-visible:outline-2 focus-visible:outline-accent ${reply ? 'bg-card font-medium text-ink shadow-sm' : 'text-ink-3'}`}>{tx('Reply')}</button>
            <button type="button" aria-pressed={!reply} onClick={() => changeIntent('task')} className={`rounded-md px-2.5 py-1 text-[12px] focus-visible:outline-2 focus-visible:outline-accent ${!reply ? 'bg-card font-medium text-ink shadow-sm' : 'text-ink-3'}`}>{tx('Ask TensorFold')}</button>
          </div> : <span className="text-[12px] font-medium text-ink-2">{tx('Ask TensorFold')}</span>}
          <span className="min-w-0 text-[12px] text-ink-3">{contextLabel ?? (reply && selectedItem ? tx('To {recipient} via {source}', { recipient: selectedItem.from, source: sourceName(selectedItem.source) }) : selectedItem ? tx('Uses this conversation as context') : tx('Uses the selected inbox sources'))}</span>
        </div>
        <div className="overflow-hidden rounded-xl border border-line-strong bg-canvas focus-within:border-accent/60 focus-within:ring-2 focus-within:ring-accent/10">
          <textarea
            key={draftKey}
            ref={textareaRef}
            disabled={!draftReady}
            value={taskText}
            onChange={(event) => setTaskText(event.target.value)}
            aria-label={reply ? tx('Reply draft') : tx('Assistant task draft')}
            onKeyDown={(event) => {
              if (!isComposerSendKey(event, sendKey)) return
              event.preventDefault()
              if (!draftReady || !taskText.trim() || taskBusy || sendingReply) return
              if (reply) onSendReply()
              else onStartTask()
            }}
            rows={2}
            placeholder={reply ? tx('Write your reply. You will confirm before it is sent.') : selectedItem ? tx('Ask for a summary, a reply draft, or a next step…') : selectedSource === 'all' ? tx('What would you like to get done?') : tx('Ask about this source…')}
            className="block max-h-[180px] min-h-[64px] w-full resize-y bg-transparent px-3 py-2.5 text-[14px] leading-relaxed text-ink outline-none placeholder:text-ink-3 disabled:opacity-60 md:min-h-[80px]"
          />
          <div className="flex flex-wrap items-center justify-between gap-2 px-2.5 pb-2">
            <div className="flex min-w-0 items-center gap-1.5 text-[11px] text-ink-3">
              <DraftSaveIndicator ready={draftReady} saving={draftStatus?.startsWith('Saving')} sessionOnly={sessionOnly} problem={problem} label={draftStatus} />
              <span className="hidden md:inline">{sendKey === 'modified-enter' ? tx('⌘ / Ctrl + Enter to submit') : tx('Enter to submit · Shift + Enter for a new line')}</span>
            </div>
            <Button variant="primary" size="sm" icon={reply ? 'send' : 'spark'} onClick={reply ? onSendReply : onStartTask} disabled={!draftReady || sendingReply || taskBusy || !taskText.trim()}>{reply ? sendingReply ? reviewReply ? tx('Preparing…') : tx('Sending…') : reviewReply ? tx('Review reply') : tx('Send reply') : taskBusy ? tx('Starting…') : tx('Start assistant task')}</Button>
          </div>
        </div>
        <DraftPersistenceNotice error={draftProblem ? draftStatus ?? null : null} sessionOnly={sessionOnly} onRetry={() => onRetryDraft?.()} />
        {suggestions.length > 0 && <details className="mt-2 text-[12px] text-ink-3">
          <summary className="w-fit cursor-pointer rounded px-1 py-0.5 hover:text-ink focus-visible:outline-2 focus-visible:outline-accent">{tx('Suggested tasks')}</summary>
          <div className="mt-1.5 flex flex-wrap gap-1.5">{suggestions.map(([label, action]) => <button key={label} type="button" disabled={!draftReady} onClick={(event) => { onDraftTask(action); event.currentTarget.closest('details')?.removeAttribute('open') }} title={action} className="rounded-lg bg-surface px-2.5 py-1.5 text-[12px] font-medium text-ink-2 hover:bg-surface-hover focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-50">{tx(label)}</button>)}</div>
        </details>}
      </div>
    </section>
  )
}
