import { currentIntlLocale } from '../../lib/preferences/current-locale'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useState } from 'react'
import type { MobileBridgePairing, MobileBridgeStatus } from '../../../electron/agent/types/mobile-bridge'
import type { RemoteControlSettings } from '../../lib/types'
import { useStore } from '../../lib/store'
import { api } from '../../lib/ipc'
import { Button, Field, Input } from '../ui'
import { QrCode } from './QrCode'

export function MobileConnectionPanel({ onEnabled }: { onEnabled?: (settings: RemoteControlSettings) => void } = {}) {
  const tx = useAuthoredText()
  const [status, setStatus] = useState<MobileBridgeStatus | null>(null)
  const [host, setHost] = useState(''), [port, setPort] = useState('8766'), [label, setLabel] = useState('My phone')
  const [pairing, setPairing] = useState<MobileBridgePairing | null>(null)
  const [devices, setDevices] = useState<Awaited<ReturnType<typeof api.listDeviceSessions>>>([])
  const [deviceBusy, setDeviceBusy] = useState<string | null>(null)
  const [error, setError] = useState(''), [busy, setBusy] = useState(false), [copied, setCopied] = useState(false)
  async function refresh() {
    const [next, sessions] = await Promise.all([api.getMobileBridgeStatus(), api.listDeviceSessions()])
    setDevices(sessions.filter(session => !session.revoked))
    setStatus(next)
    setHost(current => current || next.host || next.interfaces[0]?.address || '')
    if (next.port) setPort(String(next.port))
  }
  useEffect(() => { void refresh().catch(() => setError(tx("The private mobile connection is available in the desktop app."))) }, [])
  async function configure(enabled: boolean) {
    if (busy) return
    setBusy(true); setError(''); setPairing(null)
    try {
      setStatus(await api.configureMobileBridge({ enabled, host, port: Number(port) }))
      if (enabled) {
        const settings = await api.getSettings()
        useStore.setState(current => ({ settings: current.settings ? { ...current.settings, remoteControl: settings.remoteControl } : settings }))
        onEnabled?.(settings.remoteControl)
      }
    } catch (failure) { setError(failure instanceof Error ? failure.message : tx("The mobile connection could not start.")); await refresh().catch(() => {}) }
    finally { setBusy(false) }
  }
  async function setWatchSpeech(sessionId: string, enabled: boolean) {
    if (deviceBusy) return
    setDeviceBusy(sessionId); setError('')
    try { await api.setDeviceWatchSpeech({sessionId, enabled}); await refresh() }
    catch (failure) { setError(failure instanceof Error ? failure.message : tx("The watch permission could not be saved.")); await refresh().catch(() => {}) }
    finally { setDeviceBusy(null) }
  }
  async function pair() {
    if (busy) return
    setBusy(true); setError(''); setPairing(null)
    try { setPairing(await api.issueMobilePairing(label.trim() || 'Paired phone')) }
    catch (failure) { setError(failure instanceof Error ? failure.message : tx("The pairing code could not be created.")) }
    finally { setBusy(false) }
  }
  return <div className="rounded-xl border border-line bg-card p-3.5 shadow-card">
    <div className="text-[13px] font-medium text-ink">{tx("Connect your phone")}</div>
    <p className="mt-1 max-w-[65ch] text-[13px] leading-relaxed text-ink-3">{tx("Control tasks from the TensorFold mobile app while your Mac stays awake on the same private network.")}</p>
    <div className="mt-3 grid gap-2.5 sm:grid-cols-2">
      <Field label={tx("Private network interface")}><select aria-label={tx("Private network interface")} value={host} disabled={busy || !!status?.listening} onChange={event => setHost(event.target.value)} className="h-8 w-full rounded-md border border-line bg-card px-2 text-[13px] text-ink outline-none">
        <option value="">{tx("Select an interface")}</option>{status?.interfaces.map(item => <option key={`${item.name}:${item.address}`} value={item.address}>{item.name} · {item.address}</option>)}
      </select></Field>
      <Field label={tx("Port")}><Input aria-label={tx("Mobile connection port")} type="number" min={1024} max={65535} value={port} disabled={busy || !!status?.listening} onChange={event => setPort(event.target.value)} /></Field>
    </div>
    <div className="mt-3 flex flex-wrap items-center gap-2">
      <Button size="sm" variant={status?.enabled ? 'default' : 'primary'} disabled={busy || !status || !status.enabled && (!host || !Number.isInteger(Number(port)) || Number(port) < 1024 || Number(port) > 65535)} onClick={() => void configure(!status?.enabled)}>{busy ? tx("Working") : status?.enabled ? tx("Disable mobile connection") : tx("Enable mobile connection")}</Button>
      <span className="text-[12px] text-ink-3" role="status">{status?.listening ? tx("Private HTTPS is listening") : tx("Mobile connection is off")}</span>
    </div>
    {status && status.interfaces.length === 0 && <p className="mt-2 text-[12px] leading-relaxed text-ink-3">{tx("Join a private Wi-Fi or Ethernet network to choose an interface.")}</p>}
    {status?.listening && <>
      <p className="mt-3 break-all font-mono text-[12px] text-ink-2">{status.url}</p>
      <p className="mt-1 text-[12px] leading-relaxed text-ink-3">{tx("Pairing gives this phone a revocable device session for task status and start, guide and stop actions; transcripts, file diffs and approvals remain on the Mac.")}</p>
      <div className="mt-3 flex flex-wrap items-center gap-2"><Input className="min-w-0 flex-1" value={label} maxLength={120} onChange={event => setLabel(event.target.value)} aria-label={tx("Phone name")} placeholder={tx("Phone name")} /><Button className="shrink-0 whitespace-nowrap" size="sm" variant="default" disabled={busy} onClick={() => void pair()}>{tx("Create pairing code")}</Button></div>
    </>}
    {pairing && status?.listening && <div className="mt-3 flex flex-wrap items-start gap-3 rounded-lg border border-line bg-card-2 p-3">
      <div className="shrink-0 rounded-md bg-white p-1"><QrCode value={pairing.link} size={132} /></div>
      <div className="min-w-0 flex-1"><p className="text-[13px] text-ink-2">{tx("Scan this code in the mobile app.")}</p><p className="mt-1 text-[12px] leading-relaxed text-ink-3">{tx("Use once before ")}{new Date(pairing.expiresAt).toLocaleTimeString(currentIntlLocale())}{tx("; the app also verifies this Mac's certificate.")}</p><Button size="xs" variant="ghost" icon={copied ? 'check' : 'copy'} onClick={() => { void api.writeClipboard(pairing.link).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500) }) }}>{copied ? tx("Copied") : tx("Copy pairing link")}</Button></div>
    </div>}
    {devices.length > 0 && <div className="mt-3 space-y-2 rounded-lg border border-line bg-card-2 p-3"><div className="text-[13px] font-medium text-ink">{tx("Kokoro replies on Apple Watch")}</div><p className="text-[12px] leading-relaxed text-ink-3">{tx("Allow this named phone and its paired watch to receive audio of task replies from your Mac. Kokoro must be installed in Voice settings; private chats, tool output, file diffs and approvals stay on the Mac.")}</p>{devices.map(device => <label key={device.id} className="flex min-h-10 items-center gap-2 text-[13px] text-ink-2"><input type="checkbox" checked={device.watchSpeech === true} disabled={!!deviceBusy} onChange={event => void setWatchSpeech(device.id, event.target.checked)} aria-label={tx("Allow Kokoro watch replies for {value0}", { value0: device.label })} /><span className="min-w-0 flex-1 truncate">{device.label}</span><span className="shrink-0 text-[12px] text-ink-3">{deviceBusy === device.id ? tx("Saving…") : device.watchSpeech ? tx("Audio allowed") : tx("Audio off")}</span></label>)}</div>}
    {(error || status?.lastError) && <p className="mt-2 text-[13px] leading-relaxed text-danger" role="alert">{error || status?.lastError}</p>}
    <p className="mt-3 text-[12px] leading-relaxed text-ink-3">{tx("Disable the connection to close network access; revoke an existing phone in the Paired devices list.")}</p>
  </div>
}
