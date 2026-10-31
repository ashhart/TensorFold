import qrcode from 'qrcode-generator'
import { useEffect, useState } from 'react'
import { api } from '../api'
import { copyText } from '../shareCard'
import { useWorking } from '../working'

/** A one-time link, and its QR code, that opens this TensorFold Web signed in on another device, without the token ever
 *  being typed or shown. Good once, for five minutes. */
export default function SettingsAccess({ toast }: { toast: (t: string, k?: 'ok' | 'bad') => void }) {
  const [link, setLink] = useState<{ url: string; until: number } | null>(null)
  const [now, setNow] = useState(Date.now())
  useEffect(() => { if (!link) return; const t = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(t) }, [link])
  const local = ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname)
  const { run, is } = useWorking()
  const make = () => run('make', async () => {
    try {
      const r = await api.accessLink()
      setLink({ url: `${location.origin}/?code=${r.code}`, until: Date.now() + r.expires_in * 1000 })
    } catch (e) { toast(String((e as Error).message ?? e), 'bad') }
  })
  const left = link ? Math.max(0, Math.round((link.until - now) / 1000)) : 0
  const svg = (() => {
    if (!link || !left) return ''
    const q = qrcode(0, 'M'); q.addData(link.url); q.make()
    return q.createSvgTag({ cellSize: 4, margin: 2, scalable: true })
  })()
  return (
    <div className="card">
      <b>Open this TensorFold Web in another browser</b>
      <p className="small muted">A link that opens this TensorFold Web signed in, without typing its token: good once, for five minutes. Copy it to the other computer (a message to yourself, for instance), or show it as a QR code to open this TensorFold Web on a phone. Either one reaches TensorFold Web's machine directly, the way this browser does, so it must be able to reach this address too (on this network, Tailscale on the phone).</p>
      {local && <p className="small bad">This page is open at {location.hostname}: a link made here opens on this machine only. Open TensorFold Web at its network name to make a link for other devices.</p>}
      <div className="actions"><button className="btn" aria-busy={is('make')} onClick={make}>{link && left ? 'Make another link' : 'Make a one-time link'}</button></div>
      {link && left > 0 && <>
        <div className="row" style={{ alignItems: 'center', gap: 8 }}>
          <input className="input" readOnly value={link.url} aria-label="One-time link" onFocus={(e) => e.currentTarget.select()} style={{ flex: 1 }} />
          <button className="btn sm" onClick={() => copyText(link.url).then(() => toast('Link copied'), () => toast('The browser refused the clipboard', 'bad'))}>Copy the link</button>
        </div>
        <p className="small muted">Good for {Math.floor(left / 60)} min {String(left % 60).padStart(2, '0')} s, and only once.</p>
        <details>
          <summary className="small">On a phone: scan a QR code</summary>
          <div className="row" style={{ alignItems: 'center', gap: 16, marginTop: 8 }}>
            <div className="access-qr" role="img" aria-label="QR code of the one-time link" dangerouslySetInnerHTML={{ __html: svg }} />
            <p className="small muted" style={{ maxWidth: 320 }}>Point the phone's camera at it and open the link it offers: this TensorFold Web opens in the phone's browser, signed in (no app needed; "Add to Home Screen" makes it one). The phone reaches this TensorFold Web's machine itself, not this computer: with Tailscale on the phone, as here.</p>
          </div>
        </details>
      </>}
      {link && left === 0 && <p className="small muted">The link has expired; make another.</p>}
      <p className="small muted" style={{ marginBottom: 0 }}>From another TensorFold Web, Fleet's "Open its TensorFold Web" does the same in one click. On this TensorFold Web's machine, <code>tensorfold-web token</code> shows the token itself (<code>--new</code> makes a new one).</p>
    </div>
  )
}
