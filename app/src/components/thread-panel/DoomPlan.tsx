import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import clsx from 'clsx'
import { planProgress, type PlanStepLike } from '../../../shared/plan-progress'

// Bold progress rendering: block-glyph bar, big percentage, and per-status
// markers. Stays inside the mono skin — ink/blocks for structure, existing semantic
// tones (positive/danger) only where they already carry meaning.

const BAR_CELLS = 14

export function DoomPlan({ plan }: { plan: PlanStepLike[] }) {
  const progress = planProgress(plan)
  const pct = Math.round(progress.fraction * 100)
  const filled = Math.round(progress.fraction * BAR_CELLS)
  const remaining = progress.open - progress.blocked
  return (
    <div className="space-y-2.5">
      <div className="font-mono">
        <div className="flex items-baseline gap-2 text-[13px] leading-none">
          <span className="tracking-tighter">
            <span className="text-ink">{'█'.repeat(filled)}</span>
            <span className="text-ink-4">{'░'.repeat(Math.max(0, BAR_CELLS - filled))}</span>
          </span>
          <span className="text-[15px] font-bold tracking-tight text-ink">{pct}%</span>
        </div>
        <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] font-semibold tracking-wider text-ink-3">
          <span>{progress.done}/{progress.total} DONE</span>
          {progress.inProgress > 0 && <span className="text-ink">▸ {progress.inProgress} ACTIVE</span>}
          {progress.blocked > 0 && <span className="text-danger">■ {progress.blocked} BLOCKED</span>}
          {remaining - progress.inProgress > 0 && <span>{remaining - progress.inProgress} PENDING</span>}
        </div>
      </div>
      <div className="space-y-1 border-t border-line pt-2">
        {plan.map((step, index) => (
          <PlanRow key={index} step={step} />
        ))}
      </div>
    </div>
  )
}

function PlanRow({ step }: { step: PlanStepLike }) {
  const tx = useAuthoredText()
  const dropped = step.status === 'dropped'
  const blocked = step.status === 'blocked'
  const active = step.status === 'in_progress'
  const done = step.done || step.status === 'done'
  const marker = done ? '✓' : active ? '▸' : blocked ? '■' : dropped ? '×' : '○'
  return (
    <div className={clsx('flex items-start gap-2 font-mono text-[13px] leading-snug', step.depth && 'pl-4')}>
      {step.depth ? <span className="text-ink-4">└</span> : null}
      <span
        className={clsx(
          'w-3 shrink-0 text-center font-bold',
          done ? 'text-positive' : blocked ? 'text-danger' : dropped ? 'text-ink-4' : active ? 'text-ink' : 'text-ink-4',
        )}
      >
        {marker}
      </span>
      <span className="min-w-0">
        <span
          className={clsx(
            'block break-words',
            done ? 'text-ink-4 line-through' : dropped ? 'text-ink-4 line-through' : active ? 'font-semibold text-ink' : 'text-ink-2',
          )}
        >
          {step.depth ? <span className="text-ink-4">{'· '}</span> : null}
          {step.text}
        </span>
        {blocked && step.blocker ? <span className="mt-0.5 block break-words text-[12px] text-danger">{tx("waiting on ")}{step.blocker}</span> : null}
      </span>
    </div>
  )
}
