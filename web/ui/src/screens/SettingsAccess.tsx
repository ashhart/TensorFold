import qrcode from 'qrcode-generator'
import { useEffect, useState } from 'react'
import { api, type Access } from '../api'
import { copyText } from '../shareCard'
import { useWorking } from '../working'



export default function SettingsAccess({ toast }: { toast: (t: string, k?: 'ok' | 'bad') => void }) {
  const [access, setAccess] = useState<Access | null>(null)
  useEffect(() => { api.access().then(setAccess).catch(() => {}) }, [])
  const reach = !access ? null : access.engine.answers_without_key ? 'The engine is reachable without a key.' : access.posture === 'warn' ? 'Review access: Web or its engine is reachable from other devices.' : 'Access is limited to this machine.'
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
      <h2>Access</h2>
      {reach && <p className={`small ${access?.posture === 'warn' || access?.engine.answers_without_key ? 'bad' : 'muted'}`}>{reach}</p>}
      <b>Open this TensorFold Web in another browser</b>
      <p className="small muted">Create a link valid once for five minutes; the other device must be able to reach this Web address.</p>
      {local && <p className="small bad">Open Web at its network name to share with other devices; {location.hostname} is local only.</p>}
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
            <p className="small muted" style={{ maxWidth: 320 }}>Scan with a phone that can reach this Web address to open it signed in.</p>
          </div>
        </details>
      </>}
      {link && left === 0 && <p className="small muted">The link has expired; make another.</p>}
      <p className="small muted" style={{ marginBottom: 0 }}>Use <code>tensorfold-web token</code> to read the token, or add <code>--new</code> to replace it.</p>
    </div>
  )
}
