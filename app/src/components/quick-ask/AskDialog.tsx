import { useEffect, useId, useRef, type ReactNode } from 'react'
export function AskDialog({ title, onClose, children }: { title: string; onClose(): void; children: ReactNode }) {
  const ref = useRef<HTMLDialogElement>(null), id = useId()
  useEffect(() => { ref.current?.showModal(); return () => ref.current?.close() }, [])
  return <dialog ref={ref} aria-labelledby={id} onCancel={event => { event.preventDefault(); onClose() }} className="ask-dialog rounded-2xl border border-line bg-card p-0 text-ink shadow-pop">
    <div className="flex items-center justify-between border-b border-line px-5 py-4"><h2 id={id} className="text-base font-semibold">{title}</h2><button onClick={onClose} aria-label="Close dialog" className="ask-icon-button">×</button></div>
    <div className="max-h-[70vh] overflow-y-auto p-5">{children}</div>
  </dialog>
}
