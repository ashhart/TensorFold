import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { DraftSaveIndicator } from '../composer/DraftSaveIndicator'
import { Button } from '../ui'
import { actionLabel } from './task-context'
import type { AssistantItem, SourceKey } from './types'
import type { UiPreferences } from '../../../shared/ui-preferences'
import { isComposerSendKey } from '../../lib/composer-keyboard'
import type { RefObject } from 'react'

// Bubble composer: your message becomes a blue bubble (right).
export function Composer({ taskText, setTaskText, canDirectReply, selectedItem, selectedSource, quickActions, taskBusy, sendingReply, onSendReply, onStartTask, onDraftTask, draftKey, textareaRef, draftReady = true, draftStatus, draftProblem = false, onRetryDraft, sendKey }: {
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
}) {
  const tx = useAuthoredText()
  return (
    <div className="border-t border-line bg-canvas px-3 py-3 sm:px-6">
      <div className="mx-auto w-full max-w-[720px]">
        <div className="flex flex-wrap items-end gap-2 rounded-2xl border border-line bg-card px-3 py-2 shadow-card">
          <textarea
            key={draftKey}
            ref={textareaRef}
            disabled={!draftReady}
            value={taskText}
            onChange={(e) => setTaskText(e.target.value)}
            aria-label={canDirectReply ? tx("Reply or ask TensorFold") : tx("Ask TensorFold")}
            onKeyDown={(e) => {
              if (!isComposerSendKey(e, sendKey)) return
              e.preventDefault()
              if (!draftReady || !taskText.trim() || taskBusy || sendingReply) return
              if (canDirectReply) onSendReply()
              else onStartTask()
            }}
            rows={1}
            placeholder={canDirectReply ? (sendKey === 'modified-enter' ? tx("Reply to {value0}… (Command or Control + Enter to send · or Ask TensorFold)", { value0: selectedItem!.from }) : tx("Reply to {value0}… (Enter to send · or Ask TensorFold)", { value0: selectedItem!.from })) : selectedItem ? tx("Reply about: {value0}…", { value0: selectedItem.title.slice(0, 40) }) : (selectedSource === 'all' ? tx("Ask TensorFold to triage your day…") : tx("Ask TensorFold to work on {value0}…", { value0: selectedSource }))}
            className="max-h-[160px] min-h-[36px] min-w-0 basis-full resize-none bg-transparent py-1.5 text-[14px] leading-relaxed text-ink outline-none placeholder:text-ink-4 sm:flex-1 sm:basis-auto"
          />
          <DraftSaveIndicator ready={draftReady} saving={draftStatus?.startsWith('Saving')} problem={draftProblem} label={draftStatus} />
          {canDirectReply ? (
            <>
              <Button variant="default" size="sm" icon="spark" onClick={onStartTask} disabled={!draftReady || taskBusy || sendingReply || !taskText.trim()} title={tx("Let TensorFold draft or handle this")}>{taskBusy ? tx("Working") : tx("Ask TensorFold")}</Button>
              <Button variant="primary" size="sm" icon="send" onClick={onSendReply} disabled={!draftReady || sendingReply || taskBusy || !taskText.trim()}>{sendingReply ? tx("Sending") : tx("Send")}</Button>
            </>
          ) : (
            <Button variant="primary" size="sm" icon="spark" onClick={onStartTask} disabled={!draftReady || taskBusy || sendingReply || !taskText.trim()}>{taskBusy ? tx("Working") : tx("Ask TensorFold")}</Button>
          )}
        </div>
        {draftStatus && draftProblem && <div role="alert" className={`mt-1.5 text-[12px] ${draftProblem ? 'text-danger' : 'text-ink-3'}`}>{draftStatus}{draftProblem && onRetryDraft && <> <button type="button" className="underline" onClick={onRetryDraft}>{tx("Retry save")}</button></>}</div>}
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {quickActions.slice(0, 3).map((action) => (
            <button key={action} disabled={!draftReady} onClick={() => onDraftTask(action)} className="rounded-lg bg-surface px-2 py-1 text-[12px] font-medium text-ink-3 transition-colors hover:bg-surface-hover hover:text-ink disabled:opacity-50">{actionLabel(action)}</button>
          ))}
        </div>
      </div>
    </div>
  )
}
