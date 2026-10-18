import { useLayoutEffect, useRef, type RefObject } from 'react'

interface AccessibleDialogOptions {
  open: boolean
  onClose: () => void
  dialogRef: RefObject<HTMLElement>
  initialFocusRef?: RefObject<HTMLElement>
}

// A body portal keeps the active dialog outside the inert application root.
// Each body child retains its original inert value, including overlapping dialogs.
const backgrounds = new Map<HTMLElement, boolean>()
const dialogs: { owner: symbol; element: HTMLElement }[] = []
function updateBackground(): void {
  const top = dialogs.at(-1)?.element
  if (!top) {
    for (const [element, original] of backgrounds) element.inert = original
    backgrounds.clear()
    return
  }
  for (const child of [...document.body.children]) {
    if (!(child instanceof HTMLElement)) continue
    if (!backgrounds.has(child)) backgrounds.set(child, child.inert)
    child.inert = child.contains(top) ? backgrounds.get(child)! : true
  }
}

/** Existing drawer traps yield to a stacked security dialog without changing its decision. */
export function hasAccessibleDialogAbove(element: HTMLElement | null): boolean {
  return dialogs.length > 0 && dialogs.at(-1)?.element !== element
}
const focusable = 'button, input, select, textarea, a[href], [tabindex], [contenteditable="true"]'
function available(element: HTMLElement): boolean {
  return !element.closest('[inert], [hidden], [aria-hidden="true"]')
    && !element.matches(':disabled') && element.getClientRects().length > 0
}
function focusTargets(dialog: HTMLElement): HTMLElement[] {
  return [...dialog.querySelectorAll<HTMLElement>(focusable)].filter((element) => element.tabIndex >= 0 && available(element))
}

export function useAccessibleDialog({ open, onClose, dialogRef, initialFocusRef }: AccessibleDialogOptions): void {
  const close = useRef(onClose)
  close.current = onClose
  useLayoutEffect(() => {
    if (!open) return
    const dialog = dialogRef.current
    if (!dialog) return
    const owner = Symbol('dialog')
    dialogs.push({ owner, element: dialog })
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null
    updateBackground()
    const observer = typeof MutationObserver === 'undefined' ? null : new MutationObserver(updateBackground)
    observer?.observe(document.body, { childList: true })
    const isTop = () => dialogs.at(-1)?.owner === owner
    const focusFirst = () => {
      const initial = initialFocusRef?.current
      if (initial && dialog.contains(initial) && available(initial)) initial.focus()
      else (focusTargets(dialog)[0] ?? dialog).focus()
    }
    focusFirst()
    const onFocus = (event: FocusEvent) => {
      if (isTop() && !dialog.contains(event.target as Node)) focusFirst()
    }
    const onKey = (event: KeyboardEvent) => {
      if (!isTop()) return
      if (event.key === 'Escape') {
        event.preventDefault(); event.stopImmediatePropagation(); close.current(); return
      }
      if (event.key === 'Tab') {
        const targets = focusTargets(dialog)
        const index = targets.indexOf(document.activeElement as HTMLElement)
        // Handle each Tab explicitly so focus never enters an inert background.
        event.preventDefault(); event.stopImmediatePropagation()
        const next = event.shiftKey ? (index <= 0 ? targets.length - 1 : index - 1) : (index + 1) % Math.max(1, targets.length)
        ;(targets[next] ?? dialog).focus()
        return
      }
      // Keep native selection, copy/paste, undo and word navigation usable.
      // Other command shortcuts belong to the application behind this dialog.
      if ((event.metaKey || event.ctrlKey) && !['a', 'c', 'v', 'x', 'z', 'y', 'arrowleft', 'arrowright', 'arrowup', 'arrowdown', 'home', 'end', 'backspace', 'delete'].includes(event.key.toLowerCase())) {
        event.preventDefault(); event.stopImmediatePropagation()
      }
    }
    window.addEventListener('focusin', onFocus, true)
    window.addEventListener('keydown', onKey, true)
    return () => {
      observer?.disconnect()
      window.removeEventListener('focusin', onFocus, true)
      window.removeEventListener('keydown', onKey, true)
      const index = dialogs.findIndex((entry) => entry.owner === owner)
      if (index >= 0) dialogs.splice(index, 1)
      updateBackground()
      // A command may already have moved focus to the newly opened view.
      const active = document.activeElement
      if (previous?.isConnected && available(previous) && (active === document.body || active === null || dialog.contains(active))) previous.focus()
    }
  }, [open, dialogRef, initialFocusRef])
}
