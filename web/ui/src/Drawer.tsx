import { useEffect, useRef, type ReactNode } from 'react'

export default function Drawer({ title, close, children }: { title: string; close: () => void; children: ReactNode }) {
  const panel = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const before = document.activeElement as HTMLElement | null
    panel.current?.focus()
    const keys = (e: KeyboardEvent) => {
      if (!panel.current?.contains(document.activeElement)) return
      if (e.key === 'Escape') { e.preventDefault(); close() }
      if (e.key !== 'Tab') return
      const nodes = panel.current?.querySelectorAll<HTMLElement>('button:not([disabled]),a[href],input,select,textarea,[tabindex="0"]')
      if (!nodes?.length) return
      const first = nodes[0], last = nodes[nodes.length - 1]
      if (e.shiftKey && (document.activeElement === first || document.activeElement === panel.current)) { e.preventDefault(); last.focus() }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus() }
    }
    addEventListener('keydown', keys)
    return () => { removeEventListener('keydown', keys); before?.focus() }
  }, [close])
  return <div className="drawer-veil" onMouseDown={(e) => { if (e.target === e.currentTarget) close() }}><div className="drawer" role="dialog" aria-modal="true" aria-label={title} ref={panel} tabIndex={-1}>
    <div className="drawer-title"><h1>{title}</h1><button className="btn quiet" aria-label="Close compute and clusters" onClick={close}>✕</button></div>{children}
  </div></div>
}
