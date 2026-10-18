import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { Icon } from '../../lib/icons'
import { Button } from '../ui'
import { QrCode } from '../settings/QrCode'

// WhatsApp QR pairing surfaced inline: scan with your phone to pair.
export function WhatsappQrBanner({ qr }: { qr: string }) {
  const tx = useAuthoredText()
  return (
    <div className="mx-8 mt-4 flex flex-wrap items-center gap-4 rounded-xl border border-line bg-card px-4 py-3 shadow-card">
      <div className="rounded-lg bg-white p-2"><QrCode value={qr} size={132} /></div>
      <div className="min-w-0 flex-1">
        <div className="text-[13px] font-medium text-ink">{tx("Pair WhatsApp")}</div>
        <div className="mt-1 max-w-[420px] text-[13px] leading-snug text-ink-3">{tx("Open WhatsApp on your phone → Settings → Linked Devices → Link a Device, then scan this code. The QR refreshes automatically until paired.")}</div>
      </div>
    </div>
  )
}

// iMessage Full Disk Access popup: chat.db can't be read until the user
// grants TensorFold FDA. Offer to open the pane directly (Bug 3).
export function ImessageFdaBanner({ onOpen, onDismiss }: { onOpen: () => void; onDismiss: () => void }) {
  const tx = useAuthoredText()
  return (
    <div className="mx-8 mt-4 flex flex-wrap items-center gap-3 rounded-xl border border-warn/30 bg-warn/10 px-4 py-3 shadow-card fade-in">
      <Icon name="shield" size={18} className="shrink-0 text-warn" />
      <div className="min-w-0 flex-1">
        <div className="text-[13px] font-medium text-ink">{tx("iMessage needs Full Disk Access")}</div>
        <div className="mt-0.5 text-[13px] leading-snug text-ink-2">{tx("TensorFold can't read your iMessage database until you grant Full Disk Access in System Settings.")}</div>
      </div>
      <Button variant="primary" size="sm" icon="shield" onClick={onOpen}>{tx("Open System Settings")}</Button>
      <button onClick={onDismiss} className="text-[13px] text-ink-3 transition-colors hover:text-ink">{tx("Dismiss")}</button>
    </div>
  )
}
