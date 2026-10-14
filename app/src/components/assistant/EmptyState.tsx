import { Icon, type IconName } from '../../lib/icons'

export function EmptyState({ icon, title, detail }: { icon: IconName; title: string; detail: string }) {
  return (
    <div className="flex flex-col items-center justify-center px-5 py-12 text-center">
      <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-ink-5 text-ink-3">
        <Icon name={icon} size={18} />
      </span>
      <div className="mt-3 text-[13px] font-medium text-ink-2">{title}</div>
      <div className="mt-1 max-w-[320px] text-[13px] leading-relaxed text-ink-3">{detail}</div>
    </div>
  )
}
