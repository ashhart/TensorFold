import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { authoredText } from '../../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../../lib/store'
import type { ReactNode } from 'react'
import clsx from 'clsx'
import { relativeTime } from '../../lib/format'
import { Button } from '../ui'
import { Icon, type IconName } from '../../lib/icons'
import type { ApprovalRow, ProviderSignal, Tone } from './model'
import { approvalRiskLabel } from './attention'
import { toneBadge, toneBg, toneDot, toneText } from './styles'

export function Panel({ title, icon, action, children }: { title: string; icon: IconName; action?: string; children: ReactNode }) {
  const tx = useAuthoredText()
  return (
    <section className="overflow-hidden rounded-lg border border-line bg-card shadow-card">
      <div className="flex items-center gap-2 border-b border-line bg-card-2 px-3 py-2.5">
        <Icon name={icon} size={14} className="text-ink-3" />
        <h2 className="min-w-0 flex-1 truncate text-[13px] font-semibold text-ink">{title}</h2>
        {action && <span className="shrink-0 rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] font-medium text-ink-3">{action}</span>}
      </div>
      {children}
    </section>
  )
}

export function DisclosurePanel({
  title,
  icon,
  badge,
  isOpen,
  onToggle,
  children,
}: {
  title: string
  icon: IconName
  badge?: string
  isOpen: boolean
  onToggle: () => void
  children: ReactNode
}) {
  const tx = useAuthoredText()
  return (
    <section className="overflow-hidden rounded-lg border border-line bg-card shadow-card">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={isOpen}
        className="flex w-full items-center gap-2 border-b border-line bg-card-2 px-3 py-2.5 text-left transition-colors hover:bg-ink-5"
      >
        <Icon name={icon} size={14} className="text-ink-3" />
        <h2 className="min-w-0 flex-1 truncate text-[13px] font-semibold text-ink">{title}</h2>
        {badge && <span className="shrink-0 rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] font-medium text-ink-3">{badge}</span>}
        <Icon name="chevronDown" size={14} className={clsx('shrink-0 text-ink-3 transition-transform', isOpen ? 'rotate-0' : '-rotate-90')} />
      </button>
      {isOpen && children}
    </section>
  )
}

export function CommandStatusSummary({
  summary,
  expanded,
  onToggle,
  children,
}: {
  summary: { maxTone: Tone; attentionCount: number; approvalsCount: number; jobsStatus: { queued: number; stalled: number; failed: number; deadLetter: number }; missionCount: number }
  expanded: boolean
  onToggle: () => void
  children: ReactNode
}) {
  const tx = useAuthoredText()
  const blocked = summary.jobsStatus.stalled + summary.jobsStatus.failed + summary.jobsStatus.deadLetter
  const parts = [
    `${summary.attentionCount} attention`,
    `${summary.approvalsCount} approvals`,
    `${summary.jobsStatus.queued} queued${blocked ? ` / ${blocked} blocked` : ''}`,
    `${summary.missionCount} missions`,
  ]
  const headline = summary.maxTone === 'critical'
    ? tx("Action needed")
    : summary.maxTone === 'high'
      ? tx("Needs attention")
      : tx("Systems operational")
  return (
    <div className="rounded-lg border border-line bg-card shadow-card">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
        className="flex w-full items-center gap-2.5 px-3 py-2.5 text-left"
      >
        <span className={clsx('h-2 w-2 shrink-0 rounded-full', toneDot(summary.maxTone))} />
        <span className={clsx('shrink-0 text-[13px] font-semibold', summary.maxTone === 'good' || summary.maxTone === 'low' || summary.maxTone === 'normal' ? 'text-ink' : toneText(summary.maxTone))}>{headline}</span>
        <span className="min-w-0 flex-1 truncate text-[13px] text-ink-3">{parts.join(' · ')}</span>
        <Icon name="chevronDown" size={14} className={clsx('shrink-0 text-ink-3 transition-transform', expanded ? 'rotate-0' : '-rotate-90')} />
      </button>
      {expanded && <div className="border-t border-line p-2">{children}</div>}
    </div>
  )
}

export function Metric({ icon, label, value, detail, tone }: { icon: IconName; label: string; value: string; detail: string; tone: Tone }) {
  const tx = useAuthoredText()
  return (
    <div className="rounded-lg border border-line bg-card px-3 py-2.5 shadow-card">
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-[12px] font-medium uppercase tracking-[0.05em] text-ink-3">{label}</span>
        <Icon name={icon} size={13} className={toneText(tone)} />
      </div>
      <div className="mt-1 flex min-h-6 items-baseline gap-2">
        <span className="truncate text-[18px] font-semibold leading-none text-ink">{value}</span>
      </div>
      <div className="mt-1 truncate text-[13px] text-ink-3">{detail}</div>
    </div>
  )
}

export function RunStat({ label, value, tone }: { label: string; value: number; tone: Tone }) {
  const tx = useAuthoredText()
  return (
    <div className="bg-card px-2 py-2 text-center">
      <div className={clsx('text-[17px] font-semibold leading-none', toneText(tone))}>{value}</div>
      <div className="mt-1 truncate text-[12px] font-medium uppercase tracking-[0.05em] text-ink-4">{label}</div>
    </div>
  )
}

export function RunFact({ icon, label, value, detail, tone }: { icon: IconName; label: string; value: string; detail: string; tone: Tone }) {
  const tx = useAuthoredText()
  return (
    <div className="min-w-0 rounded-lg bg-card-2 px-2.5 py-2">
      <div className="flex items-center gap-1.5 text-[12px] font-medium uppercase tracking-[0.05em] text-ink-4">
        <Icon name={icon} size={12} className={toneText(tone)} />
        <span className="truncate">{label}</span>
      </div>
      <div className="mt-1 truncate text-[13px] font-semibold text-ink">{value}</div>
      <div className="mt-0.5 truncate text-[12px] text-ink-3">{detail}</div>
    </div>
  )
}

export function ProviderSignalBlock({ signal }: { signal: ProviderSignal }) {
  const tx = useAuthoredText()
  return (
    <div className="min-w-0 rounded-md bg-card px-2 py-1.5">
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-[12px] font-medium uppercase tracking-[0.05em] text-ink-4">{signal.label}</span>
        <span className={clsx('h-1.5 w-1.5 shrink-0 rounded-full', toneDot(signal.tone))} />
      </div>
      <div className={clsx('mt-0.5 truncate text-[13px] font-semibold', signal.tone === 'low' ? 'text-ink-3' : toneText(signal.tone))}>{signal.value}</div>
      <div className="mt-0.5 truncate text-[12px] text-ink-4">{signal.detail}</div>
    </div>
  )
}

export function EmptyState({ icon, title, detail }: { icon: IconName; title: string; detail: string }) {
  const tx = useAuthoredText()
  return (
    <div className="px-3 py-5 text-center">
      <div className="mx-auto flex h-8 w-8 items-center justify-center rounded-lg bg-ink-5 text-ink-3">
        <Icon name={icon} size={15} />
      </div>
      <div className="mt-2 text-[13px] font-medium text-ink">{title}</div>
      <div className="mx-auto mt-0.5 max-w-[260px] text-[13px] leading-snug text-ink-3">{detail}</div>
    </div>
  )
}

export function ApprovalRowItem({ row }: { row: ApprovalRow }) {
  const tx = useAuthoredText()
  return (
    <div className="px-3 py-3">
      <div className="flex items-start gap-3">
        <StatusIcon icon={row.icon} tone={row.tone} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <div className="truncate text-[13px] font-semibold text-ink">{row.title}</div>
            <span className="ml-auto shrink-0 text-[12px] text-ink-4">{relativeTime(row.createdAt)}</span>
          </div>
          <div className="mt-0.5 flex min-w-0 flex-wrap items-center gap-1.5 text-[13px] text-ink-3">
            <span className="truncate">{row.meta}</span>
            <span className="rounded bg-ink-5 px-1.5 py-0.5 text-[11px] font-medium text-ink-4">{approvalRiskLabel(row.tone)}</span>
          </div>
          <div className="mt-1.5 line-clamp-2 text-[13px] leading-snug text-ink-2">{row.detail}</div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {row.secondaryLabel && <Button variant="ghost" size="xs" onClick={row.onSecondary}>{row.secondaryLabel}</Button>}
            <Button variant={row.tone === 'critical' ? 'primary' : 'default'} size="xs" onClick={row.onPrimary}>{row.primaryLabel}</Button>
          </div>
        </div>
      </div>
    </div>
  )
}

export function StatusIcon({ icon, tone, pulse }: { icon: IconName; tone: Tone; pulse?: boolean }) {
  const tx = useAuthoredText()
  return (
    <span className={clsx('relative mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg', toneBg(tone), toneText(tone))}>
      <Icon name={icon} size={15} />
      {pulse && <span className={clsx('absolute h-2 w-2 rounded-full', toneDot(tone), 'animate-ping opacity-20')} />}
    </span>
  )
}

export function ToneBadge({ tone }: { tone: Tone }) {
  const tx = useAuthoredText()
  if (tone === 'normal') return null
  const label: Record<Tone, string> = { get critical() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "critical") }, get high() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "high") }, get normal() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "normal") }, get low() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "low") }, get good() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "ready") } }
  return <span className={clsx('shrink-0 rounded-md px-1.5 py-0.5 text-[11px] font-medium', toneBadge(tone))}>{label[tone]}</span>
}
