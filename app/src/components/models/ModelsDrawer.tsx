import { hasAccessibleDialogAbove } from '../navigation/useAccessibleDialog'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useRef, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { Button } from '../ui'

export interface ModelsDrawerProps {
  title: string
  labelId: string
  onClose: () => void
  children: ReactNode
  footer?: ReactNode
  closeDisabled?: boolean
  description?: string
}

/** One focus boundary for model workflows; callers own their dirty/apply close policy. */
export function ModelsDrawer({ title, labelId, onClose, children, footer, closeDisabled = false, description }: ModelsDrawerProps) {
  const tx = useAuthoredText()
  const panel = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLElement | null>(null)
  const close = useRef(onClose)
  const blocked = useRef(closeDisabled)
  close.current = onClose
  blocked.current = closeDisabled
  useEffect(() => {
    trigger.current = document.activeElement as HTMLElement | null
    const oldOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    panel.current?.focus()
    const focusable = () => [...(panel.current?.querySelectorAll<HTMLElement>('button:not(:disabled),select:not(:disabled),input:not(:disabled),textarea:not(:disabled),a[href],summary,[tabindex="0"]') ?? [])].filter((element) => element.getClientRects().length > 0 && !element.closest('[hidden],[inert]'))
    const key = (event: KeyboardEvent) => {
      if (hasAccessibleDialogAbove(panel.current)) return
      // The rest of the workspace must not navigate away from a mounted edit flow.
      // Keep browser/native text editing shortcuts available inside the drawer.
      if (event.metaKey || event.ctrlKey) {
        const editable = (event.target as HTMLElement | null)?.closest?.('input,textarea,[contenteditable="true"]')
        const textShortcut = ['a', 'c', 'v', 'x', 'z', 'y'].includes(event.key.toLowerCase()) || !!editable && ['arrowleft', 'arrowright', 'arrowup', 'arrowdown', 'home', 'end', 'backspace', 'delete'].includes(event.key.toLowerCase())
        if (!textShortcut) { event.preventDefault(); event.stopPropagation(); return }
        event.stopPropagation()
      }
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); if (!blocked.current) close.current(); return }
      if (event.key !== 'Tab') return
      const targets = focusable()
      if (!targets.length) { event.preventDefault(); panel.current?.focus(); return }
      const first = targets[0], last = targets.at(-1)!
      const atBoundary = document.activeElement === panel.current || !panel.current?.contains(document.activeElement)
      if (event.shiftKey && (document.activeElement === first || atBoundary)) { event.preventDefault(); last.focus() }
      else if (!event.shiftKey && (document.activeElement === last || atBoundary)) { event.preventDefault(); first.focus() }
    }
    const focus = (event: FocusEvent) => { if (hasAccessibleDialogAbove(panel.current)) return; if (!panel.current?.contains(event.target as Node)) panel.current?.focus() }
    window.addEventListener('keydown', key, true)
    document.addEventListener('focusin', focus, true)
    return () => {
      window.removeEventListener('keydown', key, true)
      document.removeEventListener('focusin', focus, true)
      document.body.style.overflow = oldOverflow
      if (trigger.current?.isConnected) trigger.current.focus()
    }
  }, [])
  return createPortal(<div className="fixed inset-0 z-[65] flex justify-end bg-black/40 backdrop-blur-sm" onMouseDown={(event) => { if (event.target === event.currentTarget && !blocked.current) close.current() }}>
    <div ref={panel} tabIndex={-1} role="dialog" aria-modal="true" aria-labelledby={labelId} aria-describedby={description ? `${labelId}-description` : undefined} aria-busy={closeDisabled} className="flex h-[100dvh] w-full min-w-0 max-w-[640px] flex-col border-l border-line bg-card text-ink shadow-pop outline-none">
      <header className="flex shrink-0 items-start justify-between gap-3 border-b border-line px-5 py-4"><div className="min-w-0"><h2 id={labelId} className="break-words text-[18px] font-semibold">{title}</h2>{description && <p id={`${labelId}-description`} className="mt-1 break-words text-[13px] leading-relaxed text-ink-2">{description}</p>}</div><Button size="sm" variant="ghost" icon="x" aria-label={tx("Close {value0}", { value0: title })} disabled={closeDisabled} onClick={() => { if (!blocked.current) close.current() }} /></header>
      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto overscroll-contain p-5">{children}</div>
      {footer && <footer className="shrink-0 border-t border-line px-5 py-3">{footer}</footer>}
    </div>
  </div>, document.body)
}
