import clsx from 'clsx'

export function UsagePill({ label, value, detail, tone = 'normal' }: { label: string; value: string; detail: string; tone?: 'good' | 'normal' | 'warn' | 'muted' }) {
  return (
    <div className={clsx(
      'rounded-lg border px-2.5 py-2',
      tone === 'good' ? 'border-positive/25 bg-positive/8'
        : tone === 'warn' ? 'border-warn/30 bg-warn/8'
          : tone === 'muted' ? 'border-line bg-card-2/70'
            : 'border-line bg-card-2',
    )}>
      <div className="text-[13px] text-ink-3">{label}</div>
      <div className={clsx('mt-0.5 truncate text-[13px] font-medium', tone === 'good' ? 'text-positive' : tone === 'warn' ? 'text-warn' : 'text-ink')}>{value}</div>
      <div className="mt-0.5 truncate text-[13px] text-ink-4">{detail}</div>
    </div>
  )
}
