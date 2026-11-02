import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import clsx from 'clsx'
import { Icon, type IconName } from '../lib/icons'

type BtnVariant = 'primary' | 'default' | 'ghost' | 'subtle' | 'danger'
type BtnSize = 'xs' | 'sm' | 'md'

export function Button({
  variant = 'default', size = 'md', icon, iconRight, className, children, ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: BtnVariant; size?: BtnSize; icon?: IconName; iconRight?: IconName }) {
  const { type = 'button', ...buttonProps } = rest
  const sizes: Record<BtnSize, string> = {
    xs: 'h-7 px-2.5 text-[13px] gap-1.5 rounded-md',
    sm: 'h-8 px-3 text-[13px] gap-1.5 rounded-lg',
    md: 'h-9 px-3.5 text-[13px] gap-2 rounded-lg',
  }
  const variants: Record<BtnVariant, string> = {
    primary: 'bg-accent text-accent-ink hover:bg-accent-strong shadow-[0_1px_2px_rgb(0_0_0/0.12)] active:scale-[0.985]',
    default: 'bg-card text-ink border border-line hover:border-line-strong hover:bg-card-2 active:scale-[0.985]',
    subtle: 'bg-ink-5 text-ink hover:bg-ink-4/60 active:scale-[0.985]',
    ghost: 'text-ink-2 hover:text-ink hover:bg-ink-5 active:scale-[0.985]',
    danger: 'text-danger hover:bg-danger/10 active:scale-[0.985]',
  }
  return (
    <button
      type={type}
      className={clsx('no-drag inline-flex select-none items-center justify-center font-medium transition-[transform,background-color,border-color,color] duration-100 disabled:pointer-events-none disabled:opacity-40', sizes[size], variants[variant], className)}
      {...buttonProps}
    >
      {icon && <Icon name={icon} size={size === 'md' ? 16 : 15} />}
      {children}
      {iconRight && <Icon name={iconRight} size={size === 'md' ? 16 : 15} />}
    </button>
  )
}

export function IconButton({ name, size = 16, label, active, tone = 'default', className, ...rest }: React.ButtonHTMLAttributes<HTMLButtonElement> & { name: IconName; size?: number; label?: string; active?: boolean; tone?: 'default' | 'danger' }) {
  const { type = 'button', ...buttonProps } = rest
  return (
    <button
      type={type}
      aria-label={label}
      title={label}
      className={clsx(
        'no-drag inline-flex h-8 w-8 items-center justify-center rounded-lg transition-colors duration-100',
        tone === 'danger' ? 'text-ink-3 hover:bg-danger/10 hover:text-danger' : 'text-ink-3 hover:bg-ink-5 hover:text-ink',
        active && 'bg-ink-5 text-ink',
        className,
      )}
      {...buttonProps}
    >
      <Icon name={name} size={size} />
    </button>
  )
}

export function Menu({
  trigger,
  children,
  align = 'start',
  width = 240,
  placement = 'auto',
}: {
  trigger: (open: boolean) => React.ReactNode
  children: (close: () => void) => React.ReactNode
  align?: 'start' | 'end'
  width?: number
  placement?: 'auto' | 'bottom' | 'top'
}) {
  const [open, setOpen] = useState(false)
  const [menuStyle, setMenuStyle] = useState<React.CSSProperties | null>(null)
  const ref = useRef<HTMLDivElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const focusLast = useRef(false)
  const focusable = () => Array.from(menuRef.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), a[href], [tabindex="0"]') ?? []).filter(element => element.getClientRects().length > 0)
  const close = () => {
    if (menuRef.current?.contains(document.activeElement)) ref.current?.querySelector<HTMLButtonElement>('button')?.focus()
    setOpen(false)
  }
  useEffect(() => {
    if (!open) return
    const focusFrame = window.requestAnimationFrame(() => {
      if (ref.current?.contains(document.activeElement)) {
        const choices = focusable()
        ;(focusLast.current ? choices.at(-1) : choices[0])?.focus()
      }
      focusLast.current = false
    })
    const onDown = (e: MouseEvent) => {
      const target = e.target as Node
      // The menu is portaled out of `ref`, so check both the trigger and the menu itself.
      if (ref.current?.contains(target) || menuRef.current?.contains(target)) return
      setOpen(false)
    }
    const onEsc = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || e.defaultPrevented) return
      e.preventDefault()
      setOpen(false)
      ref.current?.querySelector<HTMLButtonElement>('button')?.focus()
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onEsc)
    return () => { window.cancelAnimationFrame(focusFrame); document.removeEventListener('mousedown', onDown); document.removeEventListener('keydown', onEsc) }
  }, [open])
  // Position the menu in a portal with fixed (viewport) coordinates derived from the
  // trigger rect, so it is never clipped or scroll-trapped by an ancestor's overflow
  // (e.g. the sidebar's scrolling project list) — the cause of menus colliding with the
  // rows behind them. Reposition on scroll/resize so it tracks the trigger.
  useLayoutEffect(() => {
    if (!open) return
    const place = () => {
      const t = ref.current?.getBoundingClientRect()
      if (!t) return
      const edge = 12
      const gap = 6
      const below = Math.max(0, window.innerHeight - t.bottom - edge - gap)
      const above = Math.max(0, t.top - edge - gap)
      const menuH = menuRef.current?.getBoundingClientRect().height ?? 0
      const side = placement === 'top'
        ? 'top'
        : placement === 'bottom'
          ? 'bottom'
          : (menuH > below && above > below ? 'top' : 'bottom')
      const maxHeight = Math.max(96, Math.floor((side === 'top' ? above : below) || menuH))
      const menuWidth = Math.min(width, Math.max(0, window.innerWidth - edge * 2))
      const rawLeft = align === 'end' ? t.right - menuWidth : t.left
      const left = Math.max(edge, Math.min(rawLeft, window.innerWidth - menuWidth - edge))
      const style: React.CSSProperties = { position: 'fixed', width: menuWidth, maxHeight, left }
      if (side === 'bottom') style.top = t.bottom + gap
      else style.bottom = window.innerHeight - t.top + gap
      setMenuStyle(style)
    }
    place()
    const frame = window.requestAnimationFrame(place)
    window.addEventListener('resize', place)
    window.addEventListener('scroll', place, true)
    window.visualViewport?.addEventListener('resize', place)
    return () => {
      window.cancelAnimationFrame(frame)
      window.removeEventListener('resize', place)
      window.removeEventListener('scroll', place, true)
      window.visualViewport?.removeEventListener('resize', place)
    }
  }, [open, placement, width, align])
  return (
    <div ref={ref} className="relative no-drag">
      <div onClick={() => setOpen((o) => !o)} onKeyDown={event => {
        if (!open && !event.defaultPrevented && (event.key === 'ArrowDown' || event.key === 'ArrowUp')) {
          event.preventDefault()
          focusLast.current = event.key === 'ArrowUp'
          setOpen(true)
        }
      }}>{trigger(open)}</div>
      {open && createPortal(
        <div
          ref={menuRef}
          onKeyDown={event => {
            if (event.defaultPrevented) return
            const choices = focusable(), index = choices.indexOf(document.activeElement as HTMLElement)
            if (event.key === 'Tab' && (event.shiftKey ? index === 0 : index === choices.length - 1)) {
              if (event.shiftKey) event.preventDefault()
              close()
              return
            }
            if (!(event.target instanceof HTMLButtonElement) || !['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return
            event.preventDefault()
            const next = event.key === 'Home' ? 0 : event.key === 'End' ? choices.length - 1 : (index + (event.key === 'ArrowDown' ? 1 : -1) + choices.length) % choices.length
            choices[next]?.focus()
          }}
          className="overflow-y-auto overflow-x-hidden rounded-xl border border-line bg-card p-1 shadow-pop fade-in scroll-thin"
          style={{ ...(menuStyle ?? { position: 'fixed', width, left: -9999, top: 0 }), zIndex: 90, boxShadow: 'var(--shadow-pop)' }}
        >
          {children(close)}
        </div>,
        document.body,
      )}
    </div>
  )
}

export function MenuItem({ icon, children, onClick, active, danger }: { icon?: IconName; children: React.ReactNode; onClick?: () => void; active?: boolean; danger?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx('flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-[13px] transition-colors',
        danger ? 'text-danger hover:bg-danger/10' : 'text-ink hover:bg-ink-5', active && 'bg-ink-5')}
    >
      {icon && <Icon name={icon} size={15} className={danger ? '' : 'text-ink-3'} />}
      <span className="flex-1 truncate">{children}</span>
      {active && <Icon name="check" size={15} className="text-accent" />}
    </button>
  )
}

export function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={clsx('no-drag h-9 w-full rounded-lg border border-line bg-card px-3 text-[13px] text-ink outline-none transition-colors placeholder:text-ink-3 focus:border-accent focus:ring-2 focus:ring-accent/15', props.className)}
    />
  )
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <div className="mb-1.5 flex items-baseline justify-between">
        <span className="text-[13px] font-medium text-ink-2">{label}</span>
        {hint && <span className="text-[12px] text-ink-3">{hint}</span>}
      </div>
      {children}
    </label>
  )
}

export function Segmented<T extends string>({ value, onChange, options }: { value: T; onChange: (v: T) => void; options: { value: T; label: string; icon?: IconName }[] }) {
  return (
    <div className="no-drag inline-flex rounded-lg border border-line bg-card-2 p-0.5">
      {options.map((o) => (
        <button
          type="button"
          key={o.value}
          onClick={() => onChange(o.value)}
          className={clsx('inline-flex items-center gap-1.5 rounded-[7px] px-2.5 py-1.5 text-[13px] font-medium transition-all duration-100',
            value === o.value ? 'bg-card text-ink shadow-card' : 'text-ink-3 hover:text-ink-2')}
        >
          {o.icon && <Icon name={o.icon} size={14} />}
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Toggle({ checked, onChange, label, disabled }: { checked: boolean; onChange: (v: boolean) => void; label: string; disabled?: boolean }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={clsx('no-drag relative inline-flex h-[22px] w-[38px] shrink-0 items-center rounded-full transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-45', checked ? 'bg-positive' : 'bg-ink-4/50')}
    >
      <span className={clsx('inline-block h-[18px] w-[18px] rounded-full bg-white shadow-sm transition-transform duration-150', checked ? 'translate-x-[18px]' : 'translate-x-[2px]')} />
    </button>
  )
}

export function Pill({ icon, children, onClick, className }: { icon?: IconName; children: React.ReactNode; onClick?: () => void; className?: string }) {
  return (
    <button
      onClick={onClick}
      className={clsx('no-drag inline-flex items-center gap-1.5 rounded-full border border-line bg-card px-2.5 py-1 text-[13px] text-ink-2 transition-colors hover:border-line-strong hover:text-ink', className)}
    >
      {icon && <Icon name={icon} size={13} className="text-ink-3" />}
      {children}
    </button>
  )
}
