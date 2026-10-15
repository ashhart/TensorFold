import { useEffect, useRef } from 'react'
import QRCode from 'qrcode'

// Renders a scannable QR code to a canvas. Used by the WhatsApp bridge pairing
// flow: Baileys emits a QR string, we draw it so the user can scan it with their
// phone (WhatsApp → Linked Devices → Link a Device).
export function QrCode({ value, size = 200 }: { value: string; size?: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    if (!ref.current) return
    QRCode.toCanvas(ref.current, value, { width: size, margin: 1, color: { dark: '#000000', light: '#ffffff' } }, () => {})
  }, [value, size])
  return <canvas ref={ref} width={size} height={size} className="block" />
}
