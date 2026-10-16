import { useId, useState } from 'react'
import { api } from '../../lib/ipc'
import { Button, Field, Input } from '../ui'
import { phoneError } from './phone-format'

export interface PhoneSetupProps {
  busy?: boolean
  adbAvailable?: boolean
  onRefresh: () => Promise<unknown>
  onPair: (address: string, code: string) => Promise<unknown>
  onConnectWifi: (address: string) => Promise<unknown>
  onAddIos: (endpoint: string, transport: 'usb' | 'wifi') => Promise<unknown>
  compact?: boolean
  onCheckMirroring?: () => Promise<unknown>
}

export function PhoneSetup({ busy = false, adbAvailable = true, onRefresh, onPair, onConnectWifi, onAddIos, onCheckMirroring, compact = false }: PhoneSetupProps) {
  const setupId = useId().replace(/:/g, '')
  const [tab, setTab] = useState<'usb' | 'wifi' | 'iphone'>('usb')
  const [pairAddress, setPairAddress] = useState('')
  const [pairCode, setPairCode] = useState('')
  const [wifiAddress, setWifiAddress] = useState('')
  const [iosEndpoint, setIosEndpoint] = useState('http://127.0.0.1:8100')
  const [iosTransport, setIosTransport] = useState<'usb' | 'wifi'>('usb')
  const [operation, setOperation] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const pending = busy || !!operation
  const run = async (name: string, task: () => Promise<unknown>, message: string) => {
    setOperation(name); setError(''); setNotice('')
    try { await task(); setNotice(message) }
    catch (error) { setError(phoneError(error)) }
    finally { setPairCode(''); setOperation(null) }
  }
  return <div className={compact ? 'mt-3 border-t border-line pt-3' : ''}>
    <div role="tablist" aria-label="Phone connection method" className="flex flex-wrap gap-1 rounded-xl border border-line bg-card-2 p-1">
      {([{ key: 'usb', label: 'Android · USB' }, { key: 'wifi', label: 'Android · Wi-Fi' }, { key: 'iphone', label: 'iPhone' }] as const).map(({ key, label }) => <button key={key} type="button" role="tab" aria-selected={tab === key} aria-controls={`${setupId}-phone-setup-${key}`} id={`${setupId}-phone-setup-tab-${key}`} tabIndex={tab === key ? 0 : -1} onKeyDown={(event) => { const keys = ['usb', 'wifi', 'iphone'] as const; const index = keys.indexOf(key); const target = event.key === 'ArrowRight' ? keys[(index + 1) % 3] : event.key === 'ArrowLeft' ? keys[(index + 2) % 3] : event.key === 'Home' ? keys[0] : event.key === 'End' ? keys[2] : null; if (target) { event.preventDefault(); setTab(target); setPairCode(''); setError(''); setNotice(''); document.getElementById(`${setupId}-phone-setup-tab-${target}`)?.focus() } }} aria-label={label} onClick={() => { setTab(key); setPairCode(''); setError(''); setNotice('') }} className={`min-h-9 min-w-0 flex-1 rounded-lg px-3 text-[13px] font-medium focus-visible:outline-2 focus-visible:outline-accent ${tab === key ? 'bg-card text-ink shadow-card' : 'text-ink-2 hover:bg-ink-5'}`}>{key === 'iphone' ? label : <><span className="hidden sm:inline">Android · </span>{key === 'usb' ? 'USB' : 'Wi-Fi'}</>}</button>)}
    </div>
    <div id={`${setupId}-phone-setup-${tab}`} role="tabpanel" aria-labelledby={`${setupId}-phone-setup-tab-${tab}`} className="mt-3 space-y-3 text-[13px] leading-relaxed text-ink-2">
      {tab === 'usb' && <>
        <p>Connect your Android phone with a data-capable USB cable, unlock it, then allow USB debugging on the phone.</p>
        <ol className="list-decimal space-y-1.5 pl-5"><li>Enable Developer options in Android Settings, then turn on USB debugging.</li><li>When the phone asks, allow this computer and keep the phone unlocked.</li><li>Refresh devices, select your phone, then allow the current chat to control it.</li></ol>
        {!adbAvailable && <p className="rounded-lg border border-line bg-card p-3 text-ink">Android Platform Tools are required on this Mac; set the ADB path in Settings → Phone control.</p>}
        <Button size="sm" icon="refresh" disabled={pending} onClick={() => void run('refresh', onRefresh, 'Device list refreshed.')}>{operation === 'refresh' ? 'Checking…' : 'Refresh devices'}</Button>
      </>}
      {tab === 'wifi' && <>
        <p>On Android 11 or later, connect the phone and this Mac to the same trusted network, then open Developer options → Wireless debugging.</p>
        <div className="rounded-xl border border-line bg-card p-3"><p className="mb-3 font-medium text-ink">1. Pair this computer</p><p className="mb-3">Choose “Pair device with pairing code” on the phone and enter the address and six-digit code shown there.</p><div className="grid gap-3 sm:grid-cols-[1fr_130px]"><Field label="Pairing IP and port"><Input placeholder="192.168.1.20:37123" value={pairAddress} onChange={(event) => setPairAddress(event.target.value)} autoCapitalize="off" autoCorrect="off" spellCheck={false} disabled={pending} /></Field><Field label="Pairing code"><Input type="password" inputMode="numeric" autoComplete="off" maxLength={6} placeholder="6 digits" value={pairCode} onChange={(event) => setPairCode(event.target.value.replace(/\D/g, '').slice(0, 6))} disabled={pending} /></Field></div><Button size="sm" className="mt-3" disabled={pending || !pairAddress.trim() || pairCode.length !== 6 || !adbAvailable} onClick={() => void run('pair', () => onPair(pairAddress.trim(), pairCode), 'Paired; enter the connection address below.')}>{operation === 'pair' ? 'Pairing…' : 'Pair phone'}</Button></div>
        <div className="rounded-xl border border-line bg-card p-3"><p className="mb-3 font-medium text-ink">2. Connect over Wi-Fi</p><p className="mb-3">Use “IP address & port” on the main Wireless debugging screen; this usually differs from the pairing port.</p><Field label="Connection IP and port"><Input placeholder="192.168.1.20:41807" value={wifiAddress} onChange={(event) => setWifiAddress(event.target.value)} autoCapitalize="off" autoCorrect="off" spellCheck={false} disabled={pending} /></Field><Button size="sm" className="mt-3" disabled={pending || !wifiAddress.trim() || !adbAvailable} onClick={() => void run('wifi', () => onConnectWifi(wifiAddress.trim()), 'Connected; select the phone to grant this chat access.')}>{operation === 'wifi' ? 'Connecting…' : 'Connect phone'}</Button></div>
        <p className="text-[12px]">The pairing code is cleared after each attempt and is never saved in TensorFold settings; Android keeps the computer pairing until you revoke it on the phone.</p>
      </>}
      {tab === 'iphone' && <>
        <div className="rounded-xl border border-line bg-card p-3"><p className="font-medium text-ink">iPhone Mirroring over Wi-Fi</p><p className="mt-1">Open Apple’s iPhone Mirroring app on this Mac and finish its pairing first; then refresh devices and select iPhone Mirroring in TensorFold.</p><p className="mt-2">Keep the iPhone nearby and locked, with Wi-Fi and Bluetooth on; Apple requires compatible devices using the same Apple Account.</p><div className="mt-3 flex flex-wrap gap-2"><Button size="sm" icon="refresh" disabled={pending} onClick={() => void run('refresh', onRefresh, 'Device list refreshed.')}>Refresh devices</Button>{onCheckMirroring && <Button size="sm" disabled={pending} onClick={() => void run('permissions', onCheckMirroring, 'Permissions checked; complete any macOS prompts, then refresh devices.')}>Check permissions</Button>}</div></div>
        <details className="rounded-xl border border-line bg-card"><summary className="cursor-pointer rounded-xl px-3 py-3 font-medium text-ink focus-visible:outline-2 focus-visible:outline-accent">USB or developer Wi-Fi with WebDriverAgent</summary><div className="rounded-xl border border-line bg-card p-3"><p className="font-medium text-ink">iPhone developer connection</p><p className="mt-1">A signed WebDriverAgent must already be running on your iPhone through compatible Xcode; enable Developer Mode and trust this Mac first.</p><p className="mt-2">For USB, use your existing local port forward; for Wi-Fi, enter the phone’s private IP address on the same trusted network.</p><div className="mt-3 grid gap-3 sm:grid-cols-[120px_1fr]"><Field label="Connection"><select aria-label="iPhone connection" value={iosTransport} onChange={(event) => { const transport = event.target.value as 'usb' | 'wifi'; setIosTransport(transport); setIosEndpoint(transport === 'usb' ? 'http://127.0.0.1:8100' : '') }} disabled={pending} className="h-9 w-full rounded-lg border border-line bg-card px-3 text-[13px] text-ink outline-none focus:border-accent"><option value="usb">USB</option><option value="wifi">Wi-Fi</option></select></Field><Field label="WebDriverAgent endpoint"><Input value={iosEndpoint} onChange={(event) => setIosEndpoint(event.target.value)} autoCapitalize="off" autoCorrect="off" spellCheck={false} disabled={pending} /></Field></div><Button size="sm" className="mt-3" disabled={pending || !iosEndpoint.trim()} onClick={() => void run('iphone', () => onAddIos(iosEndpoint.trim(), iosTransport), 'iPhone connection checked; select it to grant this chat access.')}>{operation === 'iphone' ? 'Checking…' : 'Connect iPhone'}</Button></div></details>
        <p>TensorFold does not install WebDriverAgent or unlock the phone; protected screens and apps can block capture or control.</p>
        <button type="button" onClick={() => void api.openExternal('https://appium.github.io/appium-xcuitest-driver/latest/guides/run-preinstalled-wda/')} className="rounded text-left text-accent underline underline-offset-4 focus-visible:outline-2 focus-visible:outline-accent">WebDriverAgent setup guide</button>
      </>}
      {error && <p role="alert" className="rounded-lg bg-danger/10 p-3 text-danger">{error}</p>}
      {notice && <p role="status" className="text-positive">{notice}</p>}
    </div>
  </div>
}
