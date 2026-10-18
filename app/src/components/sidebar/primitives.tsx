import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import clsx from 'clsx'
import { Icon, type IconName } from '../../lib/icons'
import { BrandIcon, connectorBrand } from '../../lib/brand'
import { ACTIVE_ROW, HOVER_ROW } from './constants'

export function SidebarSpinner({ size = 14 }: { size?: number }) {
  const tx = useAuthoredText()
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className="animate-spin" aria-label={tx("Running")}>
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2.5" strokeOpacity="0.2" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  )
}

// A 32 px navigation row: 16 px icon, 13 px label, optional count and shortcut hint.
export function NavRow({ icon, label, active, badge, hint, onClick, children }: { icon: IconName; label: string; active?: boolean; badge?: string; hint?: string; onClick: () => void; children?: ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        'no-drag relative flex h-8 w-full items-center gap-2 rounded-[10px] px-2.5 text-[13px] transition-colors',
        active ? ACTIVE_ROW : clsx('text-ink-2 hover:text-ink', HOVER_ROW),
      )}
    >
      <Icon name={icon} size={16} className="shrink-0 text-ink-3" />
      <span className="min-w-0 flex-1 truncate text-left">{label}</span>
      {children}
      {badge && <span className="shrink-0 text-[13px] tabular-nums text-ink-3">{badge}</span>}
      {hint && <kbd className="shrink-0 font-sans text-[13px] text-ink-4">{hint}</kbd>}
    </button>
  )
}

// A connector source under the Assistant row. It filters the Assistant inbox to that source.
export function AssistantSourceRow({ id, label, detail, active, onClick }: { id: string; label: string; detail: string; active?: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      title={detail}
      className={clsx('no-drag relative flex h-8 w-full items-center gap-2 rounded-[10px] pl-[34px] pr-2.5 text-left text-[13px] transition-colors', active ? ACTIVE_ROW : clsx('text-ink-2 hover:text-ink', HOVER_ROW))}
    >
      <BrandIcon brand={connectorBrand(id)} size={16} className="shrink-0" />
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {/^error/i.test(detail) && <span className="size-1.5 shrink-0 rounded-full bg-danger" />}
    </button>
  )
}

// Section label: sentence case, 12 px, quiet.
export function SectionTitle({ children, action, onClick, open }: { children: ReactNode; action?: ReactNode; onClick?: () => void; open?: boolean }) {
  const label = (
    <>
      <span className="truncate">{children}</span>
      {onClick && <Icon name={open ? 'chevronDown' : 'chevronRight'} size={12} className="shrink-0 opacity-0 transition-opacity group-hover/section:opacity-100" />}
    </>
  )
  return (
    <div className="group/section flex h-8 items-center gap-1 px-2.5 pt-3 text-[13px] leading-4 text-ink-3">
      {onClick
        ? <button onClick={onClick} aria-expanded={open} className="no-drag flex min-w-0 items-center gap-1 transition-colors hover:text-ink-2">{label}</button>
        : <span className="flex min-w-0 items-center gap-1">{label}</span>}
      <span className="flex-1" />
      {action}
    </div>
  )
}

// Double-click any sidebar label to rename it in place. Enter / blur commits,
// Escape cancels. The displayed text keeps the caller's styling via className.
export function InlineEditable({
  value,
  editing,
  onStartEdit,
  onCommit,
  onCancel,
  className,
  inputClassName,
}: {
  value: string
  editing: boolean
  onStartEdit: () => void
  onCommit: (next: string) => void
  onCancel: () => void
  className?: string
  inputClassName?: string
}) {
  const tx = useAuthoredText()
  const [draft, setDraft] = useState(value)
  const inputRef = useRef<HTMLInputElement>(null)
  useEffect(() => {
    if (editing) {
      setDraft(value)
      const id = requestAnimationFrame(() => { inputRef.current?.focus(); inputRef.current?.select() })
      return () => cancelAnimationFrame(id)
    }
  }, [editing, value])

  if (editing) {
    return (
      <input
        ref={inputRef}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onClick={(e) => e.stopPropagation()}
        onDoubleClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => {
          if (e.key === 'Enter') { e.preventDefault(); onCommit(draft) }
          else if (e.key === 'Escape') { e.preventDefault(); onCancel() }
        }}
        onBlur={() => onCommit(draft)}
        className={clsx('min-w-0 flex-1 rounded-md border border-line-strong bg-card px-1 py-0 text-[13px] leading-[18px] text-ink outline-none', inputClassName)}
        aria-label={tx("Rename")}
      />
    )
  }
  return (
    <span
      onDoubleClick={(e) => { e.stopPropagation(); onStartEdit() }}
      // Right-click is the discoverable, reliable rename gesture — double-click races
      // the row's single-click navigation. select-text re-enables user-select (the
      // body sets it to none) so the dblclick path can also fire.
      onContextMenu={(e) => { e.preventDefault(); e.stopPropagation(); onStartEdit() }}
      title={tx("Double-click or right-click to rename")}
      className={clsx('select-text', className)}
    >
      {value}
    </span>
  )
}
