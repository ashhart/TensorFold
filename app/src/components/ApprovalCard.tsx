import { useState } from 'react'
import clsx from 'clsx'
import { Icon } from '../lib/icons'
import { useFlowText } from '../lib/preferences/useFlowText'

// Keep command text and arguments in their original form; only interface copy is translated.
export function ApprovalCard({ title, subject, mono, reason, canAllowScoped, onDeny, onAllowScoped, onAllow }: {
  title?: string; subject: string; mono: boolean; reason?: string; canAllowScoped: boolean
  onDeny: () => void; onAllowScoped: () => void; onAllow: () => void
}) {
  const { t, authored, known } = useFlowText()
  const [expanded, setExpanded] = useState(false)
  const long = subject.split('\n').length > 4 || subject.length > 320
  const action = (title || '').trim()
  const question = !action ? t('approval.generic') : /^run$/i.test(action) ? t('approval.command') : t('approval.named', { action: action.charAt(0).toLowerCase() + action.slice(1) })
  return <div className="mb-2 overflow-hidden rounded-2xl border border-line-strong bg-card shadow-card fade-in">
    <div className="flex items-center gap-2 px-3 py-2">
      <Icon name="shield" size={14} className="shrink-0 text-ink-3" />
      <span className="min-w-0 flex-1 truncate text-[13px] text-ink">{question}</span>
      <button onClick={onDeny} className="rounded-lg px-2.5 py-1 text-[13px] text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink">{t('approval.deny')}</button>
      {canAllowScoped && <button onClick={onAllowScoped} title={t('approval.scopeHelp')} className="rounded-lg px-2.5 py-1 text-[13px] text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink">{t('approval.scoped')}</button>}
      <button onClick={onAllow} className="rounded-lg bg-accent px-3 py-1 text-[13px] font-medium text-accent-ink transition-colors hover:bg-accent-strong">{t('approval.allow')}</button>
    </div>
    {subject && <div className="border-t border-line bg-card-2/60 px-3 py-2">
      <div dir="auto" className={clsx('whitespace-pre-wrap break-words text-[12px] leading-[1.55] text-ink-2', mono && 'font-mono', !expanded && long && 'line-clamp-4')}>{subject}</div>
      {long && <button onClick={() => setExpanded(value => !value)} className="mt-1 text-[12px] text-ink-3 hover:text-ink">{t(expanded ? 'approval.less' : 'approval.all')}</button>}
    </div>}
    {reason && <div className="line-clamp-2 whitespace-pre-wrap border-t border-line px-3 py-1.5 text-[12px] text-ink-3">{!known(reason) && <span>{t('common.diagnostic')}: </span>}{authored(reason)}</div>}
  </div>
}
