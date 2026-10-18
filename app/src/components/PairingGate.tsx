import { OriginalDiagnostic } from './OriginalDiagnostic'
import { useAuthoredText } from '../lib/preferences/useAuthoredText'
// Pairing screen (VST-035): shown when this page is served by a TensorFold host but the
// browser holds no working device token. Redeeming a one-time credential (from the host's
// Settings → Orchestrator → Paired devices) mints this device's own revocable session.
import { useState } from 'react'
import { redeemPairingCredential } from '../lib/remote-pairing'

export function PairingGate() {
  const tx = useAuthoredText()
  const [code, setCode] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const pair = async () => {
    const raw = code.trim()
    if (!raw || busy) return
    // Accept a bare credential or a full fragment-based link. Query links remain accepted only
    // for backward compatibility and are scrubbed by the bootstrap before this screen renders.
    let credential = raw
    try {
      const url = new URL(raw)
      const fragment = /(?:^|[#&])pair=([^&]+)/.exec(url.hash)?.[1]
      credential = url.searchParams.get('pair') || (fragment ? decodeURIComponent(fragment) : null) || raw
    } catch { /* not a URL — treat as the bare credential */ }
    setBusy(true)
    setError(null)
    const result = await redeemPairingCredential(credential)
    if (result.ok) {
      window.location.reload()
      return
    }
    setError(result.error ?? tx("Pairing failed."))
    setBusy(false)
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-bg p-6">
      <div className="w-[min(420px,calc(100vw-48px))] rounded-2xl border border-line bg-card p-6 shadow-pop">
        <div className="mb-1 text-[15px] font-semibold text-ink">{tx("Pair this browser")}</div>
        <p className="mb-4 text-[13px] leading-relaxed text-ink-3">
          {tx("This page is served by a TensorFold host. Paste a pairing link or code from the host&rsquo;s ")}<span className="text-ink-2">{tx("Settings → Orchestrator → Paired devices")}</span> {tx("to drive it from here. Each pairing is a separate device the host can revoke.")}</p>
        <input
          autoFocus
          value={code}
          onChange={(e) => setCode(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') void pair() }}
          placeholder={tx("Pairing link or code")}
          className="mb-2 w-full rounded-lg border border-line bg-card-2 px-3 py-2 text-[13px] text-ink placeholder:text-ink-4 focus:border-ink-3 focus:outline-none"
        />
        {error && <p className="mb-2 text-[13px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
        <button
          onClick={() => void pair()}
          disabled={!code.trim() || busy}
          className="w-full rounded-lg bg-ink px-3 py-2 text-[13px] font-semibold text-bg transition-opacity hover:opacity-90 disabled:opacity-40"
        >
          {busy ? tx("Pairing…") : tx("Pair")}
        </button>
      </div>
    </div>
  )
}
