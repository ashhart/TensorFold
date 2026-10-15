import clsx from 'clsx'
import { Icon } from '../../lib/icons'
import { Menu } from '../ui'
import type { ApprovalMode } from '../../lib/types'

const APPROVAL: { value: ApprovalMode; label: string; desc: string; icon: 'shield' | 'spark' | 'file' }[] = [
  { value: 'suggest', label: 'Approve edits', desc: 'Reads freely · asks before writing or running', icon: 'shield' },
  { value: 'auto', label: 'Full access', desc: 'Runs and edits without asking · still asks before deletes, secrets and pushes', icon: 'shield' },
  { value: 'readonly', label: 'Read only', desc: 'Can inspect files but never modifies', icon: 'file' },
]

// A quiet pill; only Full access keeps a warning tint, because it runs commands unasked.
function approvalTriggerTone(mode: ApprovalMode, open: boolean): string {
  if (mode === 'auto') return open ? 'bg-warn/10 text-warn' : 'text-warn hover:bg-warn/10'
  return open ? 'bg-ink-5 text-ink' : 'text-ink-2 hover:bg-ink-5 hover:text-ink'
}

function approvalIconTone(mode: ApprovalMode): string {
  if (mode === 'auto') return 'text-warn'
  if (mode === 'suggest') return 'text-ink-3'
  return 'text-ink-4'
}

export interface ApprovalMenuProps {
  /** Currently-active approval mode (suggest | auto | readonly). */
  approvalMode: ApprovalMode
  /** Apply a new approval mode (persists into Settings). */
  onSetApproval: (mode: ApprovalMode) => void
  /** External provider selected: it only plans, so approvals don't apply. */
  planningOnly?: boolean
}

const PLANNING_ONLY_REASON = 'External models only plan: no tools, no file access. Pick a local model to choose approvals.'

export function ApprovalMenu({ approvalMode, onSetApproval, planningOnly }: ApprovalMenuProps) {
  const approval = APPROVAL.find((a) => a.value === approvalMode)!
  if (planningOnly) return (
    <button disabled aria-disabled title={PLANNING_ONLY_REASON} className="inline-flex h-8 cursor-default items-center gap-1.5 rounded-full px-2.5 text-[13px] text-ink-3">
      <Icon name="shield" size={16} className="text-ink-4" />
      <span className="whitespace-nowrap">Planning only</span>
    </button>
  )
  return (
    <Menu width={252} placement="top" trigger={(open) => (
      <button className={clsx('inline-flex h-8 items-center gap-1.5 rounded-full px-2.5 text-[13px] transition-colors', approvalTriggerTone(approval.value, open))} title="Permissions">
        <Icon name={approval.icon} size={16} className={approvalIconTone(approval.value)} />
        <span className="whitespace-nowrap">{approval.label}</span>
        <Icon name="chevronDown" size={14} className={clsx(approval.value === 'auto' ? 'text-warn/80' : 'text-ink-3')} />
      </button>
    )}>
      {(close) => (
        <div>
          {APPROVAL.map((a) => (
            <button key={a.value} onClick={() => { onSetApproval(a.value); close() }} className={clsx('flex w-full items-start gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors hover:bg-ink-5', a.value === approvalMode && 'bg-ink-5')}>
              <Icon name={a.icon} size={15} className={clsx('mt-0.5', approvalIconTone(a.value))} />
              <span className="flex-1">
                <span className={clsx('flex items-center gap-1.5 text-[13px] font-medium', a.value === 'auto' ? 'text-warn' : 'text-ink')}>{a.label}{a.value === approvalMode && <Icon name="check" size={13} className={a.value === 'auto' ? 'text-warn' : 'text-accent'} />}</span>
                <span className="mt-0.5 block text-[13px] leading-snug text-ink-3">{a.desc}</span>
              </span>
            </button>
          ))}
        </div>
      )}
    </Menu>
  )
}
