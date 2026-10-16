import { useEffect, useRef, useState } from 'react'
import { api } from '../../lib/ipc'
import { runPhoneOperation, usePhone } from '../../lib/phone'
import type { PhoneAction, PhoneObservation } from '../../lib/types'
import { Icon } from '../../lib/icons'
import { Button, IconButton, Input } from '../ui'
import { PhoneGlyph } from './PhoneGlyph'
import { PhoneSetup } from './PhoneSetup'
import { phoneChatGranted, phoneObservationAge, phonePreviewPoint } from './phone-format'

export interface PhoneChatPanelProps {
  threadId?: string | null
  providerId?: string | null
  localAllowed?: boolean
  onClose: () => void
}
const transportLabel = { usb: 'USB', wifi: 'Wi-Fi', emulator: 'Emulator', mirroring: 'iPhone Mirroring' }
const stateLabel = { ready: 'Ready', unauthorized: 'Allow USB debugging on the phone', offline: 'Offline', 'setup-required': 'Setup required' }

export function PhoneChatPanel({ threadId, localAllowed = false, onClose }: PhoneChatPanelProps) {
  const { snapshot, error, pending } = usePhone()
  const [selectedId, setSelectedId] = useState('')
  const [setup, setSetup] = useState(false)
  const [tab, setTab] = useState<'screen' | 'elements'>('screen')
  const [tapArmed, setTapArmed] = useState(false)
  const [text, setText] = useState('')
  const [notice, setNotice] = useState('')
  const [now, setNow] = useState(Date.now())
  const [stopPending, setStopPending] = useState(false)
  const panelRef = useRef<HTMLElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const session = snapshot?.session
  const anotherChat = !!session?.threadId && session.threadId !== threadId
  const observation = anotherChat ? null : snapshot?.observation
  const bound = phoneChatGranted(session, threadId, now)
  const expiresIn = bound && session?.grantExpiresAt ? Math.max(1, Math.ceil((session.grantExpiresAt - now) / 60_000)) : null
  const busy = pending || !!snapshot?.busy
  const availableDevices = snapshot?.devices ?? []
  const deviceId = selectedId || session?.deviceId || availableDevices.find((device) => device.state === 'ready')?.id || availableDevices[0]?.id || ''
  const device = availableDevices.find((device) => device.id === deviceId)
  const stale = !observation || now - observation.capturedAt > 30_000
  const canAct = !!session && !!observation && !busy && !stale && localAllowed && !anotherChat
  const problem = error || snapshot?.error
  const task = async (operation: () => ReturnType<typeof api.phoneSnapshot>, message?: string) => {
    setNotice('')
    try { await runPhoneOperation(operation); if (message) setNotice(message) } catch { /* The shared phone view displays the service error. */ }
  }
  useEffect(() => { panelRef.current?.focus({ preventScroll: true }) }, [])
  useEffect(() => { setTapArmed(false); setText('') }, [observation?.id, session?.id, threadId])
  useEffect(() => { if (selectedId && !availableDevices.some((entry) => entry.id === selectedId)) setSelectedId('') }, [selectedId, availableDevices])
  useEffect(() => {
    if (!observation && !session?.grantExpiresAt) return
    setNow(Date.now())
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [observation?.id, session?.grantExpiresAt])
  const action = async (next: PhoneAction) => {
    if (!canAct || !session || !observation) return
    setTapArmed(false)
    await task(() => api.phoneAction({ ...next, sessionId: session.id, observationId: observation.id }), 'Action completed; the preview shows the updated screen.')
  }
  const stop = async () => {
    setStopPending(true); setTapArmed(false); setText('')
    try { await task(() => api.phoneStop(), 'Phone control stopped; this chat no longer has access.') }
    finally { setStopPending(false) }
  }
  const tapPreview = (event: React.MouseEvent<HTMLImageElement>) => {
    if (!tapArmed || !observation || !imageRef.current) return
    const point = phonePreviewPoint(event.clientX, event.clientY, imageRef.current.getBoundingClientRect(), observation.width, observation.height)
    if (point) void action({ action: 'tap', ...point })
  }
  const swipe = (direction: 'up' | 'down') => {
    if (!observation) return
    const { width, height } = observation
    const x = Math.floor(width / 2)
    void action({ action: 'swipe', x, toX: x, y: Math.floor(height * (direction === 'up' ? 0.75 : 0.25)), toY: Math.floor(height * (direction === 'up' ? 0.25 : 0.75)), durationMs: 400 })
  }
  return <section id="phone-chat-panel" aria-label="Phone control" tabIndex={-1} ref={panelRef} className="@container mb-3 max-h-[min(700px,70vh)] overflow-y-auto scroll-thin rounded-2xl border border-line bg-card outline-none focus-visible:ring-2 focus-visible:ring-accent/30">
    <div className="sticky top-0 z-10 flex items-center gap-2.5 border-b border-line bg-card px-4 py-3"><span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-ink-5 text-ink-2"><PhoneGlyph size={18} /></span><div className="min-w-0 flex-1"><h2 className="truncate text-[14px] font-medium text-ink">{session?.name || 'Connect your phone'}</h2><p className="text-[12px] text-ink-2">{session ? `${transportLabel[session.transport]} · ${bound ? 'This chat has control' : session.threadId ? 'Another chat has control' : 'Connected; chat access is off'}` : 'USB or Wi-Fi · Android and iPhone'}</p></div>{session && <Button size="xs" variant="danger" icon="stop" disabled={stopPending} onClick={() => void stop()}>{stopPending ? 'Stopping…' : 'Stop'}</Button>}<IconButton name="x" label="Close phone panel" onClick={onClose} /></div>
    <div className="p-4">
      {api.platform === 'browser' && <p className="mb-3 rounded-lg bg-ink-5 px-3 py-2 text-[12px] text-ink-2">Browser preview uses a simulated phone; it cannot connect to a real device.</p>}
      {problem && <div role="alert" className="mb-3 rounded-xl bg-danger/10 px-3 py-2.5 text-[13px] text-danger">{problem}<button type="button" className="ml-2 rounded font-medium underline focus-visible:outline-2 focus-visible:outline-danger" disabled={busy} onClick={() => void task(() => api.phoneRefresh())}>Retry</button></div>}
      {!snapshot && !problem && <p role="status" className="py-4 text-[13px] text-ink-2">Checking phone controls…</p>}
      {snapshot && !snapshot.supported && <p className="mb-3 text-[13px] text-ink-2">Phone control runs in the desktop app on this Mac.</p>}
      {snapshot?.supported && <>
        <div className="flex flex-wrap items-center gap-2"><label className="min-w-0 flex-1 basis-40"><span className="sr-only">Phone device</span><select value={deviceId} onChange={(event) => { setSelectedId(event.target.value); setTapArmed(false) }} disabled={busy} className="h-9 w-full min-w-0 rounded-lg border border-line bg-card px-3 text-[13px] text-ink outline-none focus:border-accent"><option value="" disabled>{availableDevices.length ? 'Select a phone' : 'Refresh to find phones'}</option>{availableDevices.map((entry) => <option key={entry.id} value={entry.id}>{entry.name} · {transportLabel[entry.transport]}{entry.state === 'ready' ? '' : ` · ${stateLabel[entry.state]}`}</option>)}</select></label><Button size="sm" disabled={busy || !device || device.state !== 'ready' || !!session} onClick={() => void task(() => api.phoneConnect({ deviceId }))}>{session && device?.id === session.deviceId ? 'Connected' : session ? 'Disconnect first' : 'Connect'}</Button><IconButton name="refresh" label="Refresh phone devices" disabled={busy} onClick={() => void task(() => api.phoneRefresh())} /></div>
        {device?.detail && <p className="mt-2 break-words text-[12px] leading-relaxed text-ink-2">{device.detail}</p>}
        {!localAllowed && <p className="mt-3 rounded-xl border border-line bg-card-2 p-3 text-[13px] leading-relaxed text-ink-2">Choose a local model in a chat on this Mac to enable phone control; phone screens are not sent to cloud providers or remote workspaces.</p>}
        {session && <>
          <div className="mt-3 grid gap-4 @lg:grid-cols-[minmax(180px,0.8fr)_minmax(0,1fr)]"><div className="min-w-0">
          <div className="flex flex-wrap items-center justify-between gap-2"><div role="tablist" aria-label="Phone preview" className="inline-flex gap-1 rounded-lg bg-ink-5 p-0.5">{([{ value: 'screen', label: 'Screen' }, { value: 'elements', label: `Elements${observation ? ` · ${observation.nodes.length}` : ''}` }] as const).map((item) => <button key={item.value} type="button" role="tab" aria-selected={tab === item.value} aria-controls={`phone-preview-${item.value}`} id={`phone-preview-tab-${item.value}`} tabIndex={tab === item.value ? 0 : -1} onKeyDown={(event) => { if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) { event.preventDefault(); const target = event.key === 'Home' ? 'screen' : event.key === 'End' ? 'elements' : tab === 'screen' ? 'elements' : 'screen'; setTab(target); document.getElementById(`phone-preview-tab-${target}`)?.focus() } }} onClick={() => setTab(item.value)} className={`min-h-8 rounded-md px-2.5 text-[12px] focus-visible:outline-2 focus-visible:outline-accent ${tab === item.value ? 'bg-card text-ink shadow-card' : 'text-ink-2'}`}>{item.label}</button>)}</div><Button size="sm" icon="monitor" disabled={busy || !localAllowed || anotherChat} onClick={() => { setTapArmed(false); void task(() => api.phoneObserve()) }}>{busy ? 'Working…' : observation ? 'Refresh screen' : 'Capture screen'}</Button></div>
          {tab === 'screen' ? <div id="phone-preview-screen" role="tabpanel" aria-labelledby="phone-preview-tab-screen" className="mt-3 flex min-h-44 items-center justify-center overflow-hidden rounded-xl border border-line bg-canvas px-3 py-4">{observation?.image ? <img ref={imageRef} src={observation.image} alt={`Phone screen captured from ${session.name}; use Elements for keyboard controls`} draggable={false} onClick={tapPreview} className={`h-[min(260px,32vh)] max-w-full object-contain ${tapArmed && canAct ? 'cursor-crosshair ring-2 ring-accent' : ''}`} /> : <div className="max-w-64 py-4 text-center"><PhoneGlyph size={28} className="mx-auto mb-3 text-ink-3" /><p className="text-[13px] font-medium text-ink">{observation ? 'Screen image unavailable' : 'Preview is off'}</p><p className="mt-1 text-[12px] leading-relaxed text-ink-2">{observation ? 'Use the available screen elements; protected content may hide the image.' : 'Capture a screen when you need it; TensorFold does not record in the background.'}</p></div>}</div> : <PhoneElements observation={observation ?? null} disabled={!canAct} onTap={(x, y) => void action({ action: 'tap', x, y })} />}
          {observation && <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-[12px] text-ink-2"><span role="status">{phoneObservationAge(observation.capturedAt, now)}{stale ? ' · Refresh before acting' : ''}</span>{observation.app && <span className="max-w-full truncate" title={observation.app}>{observation.app}</span>}</div>}
          {observation?.warnings.map((warning, index) => <p key={index} className="mt-2 text-[12px] leading-relaxed text-ink-2">{warning}</p>)}
          </div><div className="min-w-0">
          <div className="rounded-xl border border-line bg-card-2 p-3"><div className="flex flex-wrap items-center justify-between gap-3"><div className="min-w-0 flex-1 basis-40"><p className="text-[13px] font-medium text-ink">{bound ? 'Phone access is on for this chat' : 'Give this chat phone access'}</p><p className="mt-1 text-[12px] leading-relaxed text-ink-2">{bound ? 'Ask TensorFold to inspect or use the phone; access lasts up to 15 minutes, and Stop cancels the running task.' : anotherChat ? 'Press Stop to end the other chat’s access, then allow this chat.' : 'Your local model can read the screen and request actions for up to 15 minutes; this conversation stays local after access ends. Start a separate chat for cloud models.'}</p></div>{!bound && <Button size="sm" variant="primary" disabled={busy || !threadId || !localAllowed || anotherChat} onClick={() => threadId && void task(() => api.phoneGrantChat(threadId), 'This chat now has phone access.')}>Allow this chat</Button>}{bound && <span className="inline-flex items-center gap-1.5 text-[12px] font-medium text-positive"><span className="h-1.5 w-1.5 rounded-full bg-positive" />Control enabled{expiresIn != null ? ` · ${expiresIn}m left` : ''}</span>}</div>{!threadId && <p className="mt-2 text-[12px] text-ink-2">Start a chat first, then allow that chat to control this phone.</p>}</div>
          <div className="mt-3 flex flex-wrap gap-2"><Button size="xs" disabled={!canAct || !observation?.image || tab !== 'screen'} aria-pressed={tapArmed} variant={tapArmed ? 'primary' : 'default'} onClick={() => setTapArmed((armed) => !armed)}>{tapArmed ? 'Tap a point on the screen' : 'Tap on preview'}</Button><Button size="xs" disabled={!canAct} icon="back" onClick={() => void action({ action: 'back' })}>Back</Button><Button size="xs" disabled={!canAct} onClick={() => void action({ action: 'home' })}>Home</Button><Button size="xs" disabled={!canAct} onClick={() => swipe('up')}>Swipe up</Button><Button size="xs" disabled={!canAct} onClick={() => swipe('down')}>Swipe down</Button></div>
          <form onSubmit={(event) => { event.preventDefault(); if (text.trim()) void action({ action: 'type', text }) }} className="mt-3 flex flex-wrap gap-2"><label className="min-w-0 flex-1 basis-40"><span className="sr-only">Text to type on the phone</span><Input aria-label="Text to type on the phone" placeholder="Type in the focused field" value={text} disabled={!canAct} onChange={(event) => setText(event.target.value)} autoComplete="off" /></label><Button type="submit" size="sm" disabled={!canAct || !text.trim()}>Type text</Button></form>
          <p className="mt-2 text-[12px] leading-relaxed text-ink-2">Typing does not press Enter; review the phone before sending messages, making purchases or confirming changes.</p>
          </div></div>
          <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-3"><span className="inline-flex items-center gap-1.5 text-[12px] text-ink-2"><Icon name="shield" size={13} />Screen previews stay on this Mac</span><Button size="xs" variant="ghost" disabled={stopPending} onClick={() => { setTapArmed(false); void task(() => api.phoneDisconnect(), 'Phone disconnected.') }}>Disconnect</Button></div>
        </>}
        <button type="button" onClick={() => setSetup((open) => !open)} aria-expanded={setup || !availableDevices.length} className="mt-3 inline-flex min-h-8 items-center gap-1.5 rounded px-1 text-[12px] font-medium text-ink-2 hover:text-ink focus-visible:outline-2 focus-visible:outline-accent"><Icon name="chevronRight" size={13} className={setup || !availableDevices.length ? 'rotate-90' : ''} />{session ? 'Connect another phone' : 'Connection help'}</button>
        {(setup || !availableDevices.length) && <PhoneSetup onCheckMirroring={() => runPhoneOperation(() => api.phoneMirroringPermissions())} compact busy={busy} adbAvailable={snapshot.adb.available} onRefresh={() => runPhoneOperation(() => api.phoneRefresh())} onPair={(endpoint, code) => runPhoneOperation(() => api.phonePairWifi({ endpoint, code }))} onConnectWifi={(endpoint) => runPhoneOperation(() => api.phoneConnectWifi(endpoint))} onAddIos={(endpoint, transport) => runPhoneOperation(() => api.phoneConnectDeveloper({ endpoint, transport }))} />}
      </>}
      {notice && <p role="status" className="mt-3 text-[12px] text-positive">{notice}</p>}
    </div>
  </section>
}

function PhoneElements({ observation, disabled, onTap }: { observation: PhoneObservation | null; disabled: boolean; onTap: (x: number, y: number) => void }) {
  const [search, setSearch] = useState('')
  const nodes = observation?.nodes.filter((node) => !node.secure && `${node.text} ${node.description} ${node.role}`.toLowerCase().includes(search.toLowerCase())) ?? []
  return <div id="phone-preview-elements" role="tabpanel" aria-labelledby="phone-preview-tab-elements" className="mt-3 rounded-xl border border-line bg-card-2 p-3"><label><span className="sr-only">Search phone elements</span><Input placeholder="Find a screen element" value={search} onChange={(event) => setSearch(event.target.value)} /></label><div className="mt-2 max-h-64 space-y-1 overflow-y-auto">{nodes.map((node) => <button key={node.id} type="button" disabled={disabled || node.bounds.width <= 0 || node.bounds.height <= 0} onClick={() => onTap(Math.floor(node.bounds.x + node.bounds.width / 2), Math.floor(node.bounds.y + node.bounds.height / 2))} className="flex min-h-10 w-full items-center gap-3 rounded-lg px-2.5 py-2 text-left text-[12px] text-ink hover:bg-ink-5 focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-45"><span className="min-w-0 flex-1 break-words">{node.text || node.description || node.role || 'Unnamed element'}{node.role && <span className="mt-0.5 block text-[11px] text-ink-2">{node.role}{node.editable ? ' · Text field' : ''}</span>}</span><Icon name="arrowRight" size={13} className="shrink-0 text-ink-3" /></button>)}{!nodes.length && <p className="px-2 py-4 text-[12px] text-ink-2">{observation ? 'No accessible elements match; a protected or image-only screen may hide its controls.' : 'Capture the phone screen to inspect its elements.'}</p>}</div></div>
}
