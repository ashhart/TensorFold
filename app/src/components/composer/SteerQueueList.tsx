import clsx from 'clsx'
import { Icon } from '../../lib/icons'

/** A staged follow-up in the steer queue above the composer. */
export type Steer = { id: string; text: string; status: 'queued' | 'guiding' | 'guided' }

export interface SteerQueueListProps {
  /** The staged messages, newest last. */
  queued: Steer[]
  /** Whether the active engine can take live mid-turn input (affects the Guide tooltip). */
  liveSteering: boolean
  /** Inject a queued message into the running turn now. */
  onGuide: (id: string, body: string) => void
  /** Remove a message from the queue. */
  onRemove: (id: string) => void
}

export function SteerQueueList({ queued, liveSteering, onGuide, onRemove }: SteerQueueListProps) {
  return (
    <div className="mb-2 space-y-1.5">
      {queued.map((q) => (
        <div
          key={q.id}
          className={clsx(
            'group flex items-start gap-2.5 rounded-2xl border px-3.5 py-2.5 fade-in transition-colors',
            q.status === 'guided' ? 'border-positive/25 bg-positive/5'
              : q.status === 'guiding' ? 'border-accent/30 bg-accent-soft'
                : 'border-line bg-card shadow-card',
          )}
        >
          {/* Status glyph + label */}
          <span className="mt-0.5 flex shrink-0 items-center gap-1.5">
            {q.status === 'guided'
              ? <Icon name="check" size={13} className="text-positive" />
              : q.status === 'guiding'
                ? <span className="relative flex h-2 w-2"><span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-60" /><span className="relative inline-flex h-2 w-2 rounded-full bg-accent" /></span>
                : <Icon name="clock" size={13} className="text-ink-3" />}
          </span>
          <div className="min-w-0 flex-1">
            <div className="mb-0.5 flex items-center gap-1.5 text-[13px] font-medium text-ink-3">
              {q.status === 'guided' ? 'Guided' : q.status === 'guiding' ? 'Guiding' : 'Queued'}
            </div>
            {/* Full message text, wrapping (not truncated) so the whole steer is readable. */}
            <div className={clsx('whitespace-pre-wrap break-words text-[13px] leading-snug', q.status === 'guided' ? 'text-ink-3 line-through' : 'text-ink-2')}>{q.text}</div>
          </div>
          {/* Actions: Guide (queued) → disabled Guiding → none (guided). Always removable. */}
          <div className="flex shrink-0 items-center gap-1">
            {q.status === 'queued' && (
              <button
                onClick={() => onGuide(q.id, q.text)}
                title={liveSteering
                  ? 'Send into the running task now — steers the turn live'
                  : "Send now — this engine can't take mid-turn input, so it applies when the current engine turn finishes"}
                className="flex items-center gap-1 rounded-lg border border-line bg-card px-2 py-1 text-[13px] font-medium text-ink-2 transition-colors hover:border-line-strong hover:bg-card-2 hover:text-ink"
              >
                <Icon name="spark" size={12} className="text-accent" />Guide
              </button>
            )}
            {q.status === 'guiding' && (
              <span className="flex items-center gap-1 px-2 py-1 text-[13px] font-medium text-ink-3"><Icon name="refresh" size={12} className="animate-spin" />Sending</span>
            )}
            {q.status !== 'guiding' && (
              <button
                onClick={() => onRemove(q.id)}
                aria-label="Remove from queue"
                title="Remove"
                className="flex h-6 w-6 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink"
              >
                <Icon name="x" size={13} />
              </button>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}
