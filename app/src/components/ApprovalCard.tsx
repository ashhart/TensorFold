import { useState } from 'react'
import clsx from 'clsx'
import { Icon } from '../lib/icons'

// The one pending-approval prompt, docked above the composer: the question and buttons on one
// line, the command in a clamped monospace block, then why it asked.
export function ApprovalCard({ title, subject, mono, reason, canAllowScoped, onDeny, onAllowScoped, onAllow }: {
  title?: string; subject: string; mono: boolean; reason?: string; canAllowScoped: boolean
  onDeny: () => void; onAllowScoped: () => void; onAllow: () => void
}) {
  const [expanded, setExpanded] = useState(false)
  const long = subject.split('\n').length > 4 || subject.length > 320
  return (
    <div className="mb-2 overflow-hidden rounded-2xl border border-line-strong bg-card shadow-card fade-in">
      <div className="flex items-center gap-2 px-3 py-2">
        <Icon name="shield" size={14} className="shrink-0 text-ink-3" />
        <span className="min-w-0 flex-1 truncate text-[13px] text-ink">{approvalQuestion(title)}</span>
        <button onClick={onDeny} className="rounded-lg px-2.5 py-1 text-[13px] text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink">Deny</button>
        {canAllowScoped && <button onClick={onAllowScoped} title="Allow for this project or workspace, for up to 8 hours or 20 uses" className="rounded-lg px-2.5 py-1 text-[13px] text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink">Allow scoped</button>}
        <button onClick={onAllow} className="rounded-lg bg-accent px-3 py-1 text-[13px] font-medium text-accent-ink transition-colors hover:bg-accent-strong">Allow</button>
      </div>
      {subject && (
        <div className="border-t border-line bg-card-2/60 px-3 py-2">
          <div className={clsx('whitespace-pre-wrap break-words text-[12px] leading-[1.55] text-ink-2', mono && 'font-mono', !expanded && long && 'line-clamp-4')}>{subject}</div>
          {long && <button onClick={() => setExpanded((v) => !v)} className="mt-1 text-[12px] text-ink-3 hover:text-ink">{expanded ? 'Show less' : 'Show all'}</button>}
        </div>
      )}
      {reason && <div className="line-clamp-2 whitespace-pre-wrap border-t border-line px-3 py-1.5 text-[12px] text-ink-3">{reason}</div>}
    </div>
  )
}

function approvalQuestion(title?: string): string {
  const action = (title || '').trim()
  if (!action) return 'Allow this action?'
  if (/^run$/i.test(action)) return 'Run this command?'
  return `Allow ${action.charAt(0).toLowerCase()}${action.slice(1)}?`
}
