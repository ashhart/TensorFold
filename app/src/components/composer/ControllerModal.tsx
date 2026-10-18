import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { Icon } from '../../lib/icons'
import { Button } from '../ui'
import type { ControllerSnapshot } from '../../lib/types'

export interface ControllerModalProps {
  /** Whether the modal is open. */
  controllerModal: boolean
  /** The captured foreground window snapshot, if any (null = permission/enabled flow). */
  controllerPreview: ControllerSnapshot | null
  /** True while a capture/enable request is in flight. */
  controllerBusy: boolean
  /** An error message to surface, or null. */
  controllerError: string | null
  /** Close the modal (also discards the preview). */
  onClose: () => void
  /** Enable Controller permission and capture (shown when there's no preview yet). */
  onEnable: () => void
  /** Insert the captured snapshot into the composer (shown when there's a preview). */
  onInsert: () => void
}

export function ControllerModal({
  controllerModal,
  controllerPreview,
  controllerBusy,
  controllerError,
  onClose,
  onEnable,
  onInsert,
}: ControllerModalProps) {
  const tx = useAuthoredText()
  if (!controllerModal) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/55 p-5">
      <div className="w-[min(480px,calc(100vw-40px))] overflow-hidden rounded-2xl border border-line bg-card p-5 shadow-pop">
        <div className="mb-8 flex items-start justify-between gap-4">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl border-4 border-line-strong bg-card-2 text-ink-2 shadow-card">
            <Icon name="monitor" size={30} strokeWidth={2.1} />
          </div>
          <button onClick={onClose} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink" aria-label={tx("Close Controller prompt")}>
            <Icon name="x" size={15} />
          </button>
        </div>
        <h2 className="text-[20px] font-semibold text-ink">{controllerPreview ? tx("Attach Controller Capture") : tx("Enable Controller")}</h2>
        {controllerPreview ? (
          <div className="mt-3 rounded-xl border border-line bg-card-2 p-3 text-[13px] leading-relaxed text-ink-2">
            <div className="font-medium text-ink">{controllerPreview.appName}</div>
            <div className="mt-1 text-ink-3">{controllerPreview.windowTitle}</div>
            <div className="mt-2 text-ink-3">{controllerPreview.text ? controllerPreview.truncated ? tx("Captured readable UI characters: {count} (truncated).", { count: controllerPreview.textCharCount }) : tx("Captured readable UI characters: {count}.", { count: controllerPreview.textCharCount }) : tx("Metadata only.")}</div>
            {controllerPreview.text && <pre className="scroll-thin mt-2 max-h-36 overflow-auto whitespace-pre-wrap rounded-lg border border-line bg-card px-2.5 py-2 font-mono text-[12px] text-ink-2">{controllerPreview.text.slice(0, 1200)}</pre>}
            {controllerPreview.warning && <div className="mt-2 text-warn">{controllerPreview.warning}</div>}
          </div>
        ) : (
          <p className="mt-2 text-[14px] leading-relaxed text-ink-3">
            {tx("Controller attaches the foreground window only after you ask. You can review the captured app, title, and readable text before inserting it as context.")}</p>
        )}
        {controllerError && <div className="mt-3 rounded-lg border border-warn/30 bg-warn/5 px-3 py-2 text-[13px] text-warn">{controllerError}</div>}
        <div className="mt-5 flex justify-end gap-2">
          <Button variant="ghost" onClick={onClose}>{tx("Cancel")}</Button>
          {controllerPreview
            ? <Button variant="primary" onClick={onInsert} disabled={controllerBusy}>{tx("Insert capture")}</Button>
            : <Button variant="primary" onClick={onEnable} disabled={controllerBusy}>{controllerBusy ? tx("Enabling...") : tx("Enable")}</Button>}
        </div>
      </div>
    </div>
  )
}
