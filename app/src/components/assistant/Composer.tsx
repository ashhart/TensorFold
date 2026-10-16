import { Button } from '../ui'
import { actionLabel } from './task-context'
import type { AssistantItem, SourceKey } from './types'
import { isTextComposition } from '../../lib/composer-keyboard'

// Bubble composer: your message becomes a blue bubble (right).
export function Composer({ taskText, setTaskText, canDirectReply, selectedItem, selectedSource, quickActions, taskBusy, sendingReply, onSendReply, onStartTask, onDraftTask }: {
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
}) {
  return (
    <div className="border-t border-line bg-canvas px-3 py-3 sm:px-6">
      <div className="mx-auto w-full max-w-[720px]">
        <div className="flex flex-wrap items-end gap-2 rounded-2xl border border-line bg-card px-3 py-2 shadow-card">
          <textarea
            value={taskText}
            onChange={(e) => setTaskText(e.target.value)}
            aria-label={canDirectReply ? 'Reply or ask TensorFold' : 'Ask TensorFold'}
            onKeyDown={(e) => {
              if (isTextComposition(e) || e.repeat || e.key !== 'Enter' || e.shiftKey) return
              e.preventDefault()
              if (!taskText.trim() || taskBusy || sendingReply) return
              if (canDirectReply) onSendReply()
              else onStartTask()
            }}
            rows={1}
            placeholder={canDirectReply ? `Reply to ${selectedItem!.from}… (Enter to send · or Ask TensorFold)` : selectedItem ? `Reply about: ${selectedItem.title.slice(0, 40)}…` : (selectedSource === 'all' ? 'Ask TensorFold to triage your day…' : `Ask TensorFold to work on ${selectedSource}…`)}
            className="max-h-[160px] min-h-[36px] min-w-0 basis-full resize-none bg-transparent py-1.5 text-[14px] leading-relaxed text-ink outline-none placeholder:text-ink-4 sm:flex-1 sm:basis-auto"
          />
          {canDirectReply ? (
            <>
              <Button variant="default" size="sm" icon="spark" onClick={onStartTask} disabled={taskBusy || sendingReply || !taskText.trim()} title="Let TensorFold draft or handle this">{taskBusy ? 'Working' : 'Ask TensorFold'}</Button>
              <Button variant="primary" size="sm" icon="send" onClick={onSendReply} disabled={sendingReply || taskBusy || !taskText.trim()}>{sendingReply ? 'Sending' : 'Send'}</Button>
            </>
          ) : (
            <Button variant="primary" size="sm" icon="spark" onClick={onStartTask} disabled={taskBusy || sendingReply || !taskText.trim()}>{taskBusy ? 'Working' : 'Ask TensorFold'}</Button>
          )}
        </div>
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {quickActions.slice(0, 3).map((action) => (
            <button key={action} onClick={() => onDraftTask(action)} className="rounded-lg bg-surface px-2 py-1 text-[12px] font-medium text-ink-3 transition-colors hover:bg-surface-hover hover:text-ink">{actionLabel(action)}</button>
          ))}
        </div>
      </div>
    </div>
  )
}
