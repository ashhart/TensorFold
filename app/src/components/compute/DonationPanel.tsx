import { useEffect, useMemo, useState } from 'react'
import { Button, Input } from '../ui'
import { api } from '../../lib/ipc'
import { makeDonationCard, type DonationCard, type DonationHostStatus, type DonationSnapshot } from '../../../shared/donation'

/**
 * POC: lend a machine for a window, borrow somebody else's. No payment, no proof of computation.
 * The donor side is a separate process (`donation-node.js --config donation.json`); this panel
 * builds its config and the card to hand over, and imports cards on the borrowing side.
 */

const HOUR = 60 * 60 * 1000
const randomHex = (bytes: number) => Array.from(crypto.getRandomValues(new Uint8Array(bytes))).map((value) => value.toString(16).padStart(2, '0')).join('')

function copy(text: string): Promise<void> {
  return navigator.clipboard?.writeText(text) ?? Promise.reject(new Error('The clipboard is unavailable.'))
}

function remaining(ms: number): string {
  if (ms <= 0) return 'closed'
  const minutes = Math.round(ms / 60_000)
  if (minutes < 60) return `${minutes} min left`
  const hours = Math.floor(minutes / 60)
  return `${hours} h ${minutes % 60} min left`
}

export function DonationPanel() {
  const [snapshot, setSnapshot] = useState<DonationSnapshot | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [cardText, setCardText] = useState('')
  const [made, setMade] = useState<{ card: DonationCard; config: string } | null>(null)

  const [label, setLabel] = useState('My machine')
  const [endpoint, setEndpoint] = useState('')
  const [upstream, setUpstream] = useState('http://127.0.0.1:8000')
  const [model, setModel] = useState('')
  const [hours, setHours] = useState('4')
  const [maxConcurrent, setMaxConcurrent] = useState('1')
  const [maxTokens, setMaxTokens] = useState('1024')
  const [maxBytes, setMaxBytes] = useState('1048576')
  const [hardware, setHardware] = useState('')
  const [parameters, setParameters] = useState('')
  const [quantization, setQuantization] = useState('')
  const [contextTokens, setContextTokens] = useState('')
  const [tokensPerSecond, setTokensPerSecond] = useState('')
  const [firstTokenMs, setFirstTokenMs] = useState('')
  const [listenPort, setListenPort] = useState('8010')
  const [host, setHost] = useState<DonationHostStatus>({ state: 'idle' })

  useEffect(() => {
    let live = true
    api.donationList().then((value) => { if (live) setSnapshot(value) }).catch((failure) => { if (live) setError(String(failure?.message ?? failure)) })
    api.donationHostStatus().then((value) => { if (live) setHost(value) }).catch(() => {})
    return () => { live = false }
  }, [])
  useEffect(() => {
    if (host.state !== 'running' && host.state !== 'starting') return
    const timer = setInterval(() => { api.donationHostStatus().then(setHost).catch(() => {}) }, 2_000)
    return () => clearInterval(timer)
  }, [host.state])

  const cards = snapshot?.cards ?? []
  const usable = useMemo(() => cards.filter((card) => card.phase !== 'after'), [cards])

  const specNow = () => (hardware.trim() || parameters.trim() || quantization.trim() || contextTokens.trim() || tokensPerSecond.trim() || firstTokenMs.trim() ? {
    ...(hardware.trim() ? { hardware: hardware.trim() } : {}),
    ...(parameters.trim() ? { parameters: parameters.trim() } : {}),
    ...(quantization.trim() ? { quantization: quantization.trim() } : {}),
    ...(contextTokens.trim() ? { contextTokens: Number(contextTokens) } : {}),
    ...(tokensPerSecond.trim() ? { tokensPerSecond: Number(tokensPerSecond) } : {}),
    ...(firstTokenMs.trim() ? { firstTokenMs: Number(firstTokenMs) } : {}),
  } : undefined)

  const startHost = async () => {
    setBusy('host'); setError(null); setNotice(null)
    try {
      const status = await api.donationHostStart({
        label, upstream, model, listenPort: Number(listenPort), hours: Number(hours),
        caps: { maxConcurrent: Number(maxConcurrent), maxTokensPerRequest: Number(maxTokens), maxRequestBytes: Number(maxBytes) },
        ...(specNow() ? { spec: specNow() } : {}),
      })
      setHost(status)
      setNotice('This machine is lending. Hand the card to whoever should use it.')
    } catch (failure) {
      setError(String((failure as Error)?.message ?? failure))
    } finally { setBusy(null) }
  }
  const stopHost = async () => {
    setBusy('host'); setError(null); setNotice(null)
    try { setHost(await api.donationHostStop()); setNotice('Lending stopped.') }
    catch (failure) { setError(String((failure as Error)?.message ?? failure)) }
    finally { setBusy(null) }
  }

  const create = () => {
    setError(null); setNotice(null); setMade(null)
    const length = Number(hours)
    if (!Number.isFinite(length) || length <= 0 || length > 24 * 30) { setError('Choose a window between 1 hour and 30 days.'); return }
    const now = Date.now()
    const result = makeDonationCard({
      id: `donate-${randomHex(4)}`,
      label,
      endpoint,
      token: randomHex(24),
      model,
      window: { from: now, until: now + Math.round(length * HOUR) },
      caps: { maxConcurrent: Number(maxConcurrent), maxTokensPerRequest: Number(maxTokens), maxRequestBytes: Number(maxBytes) },
      ...(specNow() ? { spec: specNow() } : {}),
      security: 'poc-insecure-lan',
      createdAt: now,
    })
    if ('error' in result) { setError(result.error); return }
    const card = result.card
    const port = new URL(card.endpoint).port || '8010'
    const config = JSON.stringify({ upstream, listen: { host: '0.0.0.0', port: Number(port) }, card }, null, 2)
    setMade({ card, config })
  }

  const borrow = async () => {
    setBusy('import'); setError(null); setNotice(null)
    try {
      const value = await api.donationImport({ text: cardText })
      setSnapshot(value); setCardText(''); setNotice('Card imported. The machine is now a provider in this workspace.')
    } catch (failure) {
      setError(String((failure as Error)?.message ?? failure))
    } finally { setBusy(null) }
  }

  const check = async (id: string) => {
    setBusy(id); setError(null); setNotice(null)
    try {
      const value = await api.donationProbe({ id })
      setNotice(value.ok ? `${value.label} answers with ${value.models?.join(', ') ?? value.model} (${value.phase}).` : `${value.label}: ${value.error}`)
    } catch (failure) {
      setError(String((failure as Error)?.message ?? failure))
    } finally { setBusy(null) }
  }

  const remove = async (id: string) => {
    setBusy(id); setError(null); setNotice(null)
    try {
      setSnapshot(await api.donationRemove({ id }))
      setNotice('Borrowed machine removed; its provider entry is gone.')
    } catch (failure) {
      setError(String((failure as Error)?.message ?? failure))
    } finally { setBusy(null) }
  }

  return <div className="space-y-6">
    <section className="rounded-xl border border-warn/40 bg-warn/5 p-4 text-[12px] leading-relaxed text-ink-2">
      <p className="font-medium text-ink">POC — no money, no proof of computation.</p>
      <p className="mt-1">A lent machine is reachable over plain HTTP on your network. Anyone with the card token can use it until the window closes. Borrowed models are classified as external, so agent runs plan with them and keep tools local.</p>
    </section>

    {(error || notice) && <p role="status" className={error ? 'rounded-lg border border-danger/25 bg-danger/10 px-3 py-2 text-[13px] text-danger' : 'rounded-lg border border-line bg-card px-3 py-2 text-[13px] text-ink-2'}>{error ?? notice}</p>}

    <section className="rounded-xl border border-line bg-card-2 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-[14px] font-medium text-ink">Donate this machine</h3>
          <p className="mt-1 text-[12px] leading-relaxed text-ink-2">Start a donor the app supervises: it listens on loopback, enforces the window and caps below, and stops itself when the window ends or the app quits. Renters reach it through an SSH tunnel (<code>ssh -N -L {listenPort}:127.0.0.1:{listenPort} you@this-host</code>).</p>
        </div>
        <span className={host.state === 'running' ? 'rounded-md bg-accent/10 px-2 py-0.5 text-[11px] text-accent' : 'rounded-md bg-surface px-2 py-0.5 text-[11px] text-ink-3'}>{host.state === 'running' ? `lending · ${host.receipts ?? 0} receipts` : host.state}</span>
      </div>
      <div className="mt-3 grid gap-2 sm:grid-cols-2">
        <label className="text-[12px] text-ink-2">Listen port<Input value={listenPort} onChange={(event) => setListenPort(event.target.value)} inputMode="numeric" placeholder="8010" /></label>
      </div>
      {host.error && <p className="mt-2 text-[12px] text-danger">{host.error}</p>}
      {host.card && host.state !== 'idle' && <div className="mt-3">
        <div className="flex items-center justify-between gap-2"><span className="text-[12px] font-medium text-ink">Card to hand over{host.until ? ` · until ${new Date(host.until).toLocaleTimeString()}` : ''}</span><Button size="xs" variant="ghost" onClick={() => void copy(JSON.stringify(host.card, null, 2)).then(() => setNotice('Card copied.'), (failure) => setError(String(failure.message ?? failure)))}>Copy</Button></div>
        <pre className="scroll-thin mt-1 max-h-40 overflow-auto rounded-lg border border-line bg-card p-3 text-[11px] text-ink-2">{JSON.stringify(host.card, null, 2)}</pre>
      </div>}
      <div className="mt-3 flex gap-2">
        {host.state === 'running' || host.state === 'starting' ? <Button size="sm" variant="default" disabled={busy === 'host'} onClick={() => void stopHost()}>Stop donating</Button> : <Button size="sm" variant="primary" disabled={busy === 'host'} onClick={() => void startHost()}>{busy === 'host' ? 'Starting…' : 'Start donating'}</Button>}
      </div>
    </section>

    <section>
      <h3 className="text-[14px] font-medium text-ink">Lend this machine</h3>
      <p className="mt-1 text-[12px] text-ink-2">Builds a card to hand over and the config for the donor process. The app does not start the donor process itself.</p>
      <div className="mt-3 grid gap-2 sm:grid-cols-2">
        <label className="text-[12px] text-ink-2">Label<Input value={label} onChange={(event) => setLabel(event.target.value)} maxLength={120} /></label>
        <label className="text-[12px] text-ink-2">Address renters use (https, or 127.0.0.1 through a tunnel)<Input value={endpoint} onChange={(event) => setEndpoint(event.target.value)} placeholder="http://127.0.0.1:8010" spellCheck={false} /></label>
        <label className="text-[12px] text-ink-2">Local engine<Input value={upstream} onChange={(event) => setUpstream(event.target.value)} placeholder="http://127.0.0.1:8000" spellCheck={false} /></label>
        <label className="text-[12px] text-ink-2">Model id<Input value={model} onChange={(event) => setModel(event.target.value)} placeholder="qwen3.6" spellCheck={false} /></label>
        <label className="text-[12px] text-ink-2">Window (hours)<Input value={hours} onChange={(event) => setHours(event.target.value)} inputMode="numeric" /></label>
        <label className="text-[12px] text-ink-2">Concurrent requests<Input value={maxConcurrent} onChange={(event) => setMaxConcurrent(event.target.value)} inputMode="numeric" /></label>
        <label className="text-[12px] text-ink-2">Max output tokens<Input value={maxTokens} onChange={(event) => setMaxTokens(event.target.value)} inputMode="numeric" /></label>
        <label className="text-[12px] text-ink-2">Max request bytes<Input value={maxBytes} onChange={(event) => setMaxBytes(event.target.value)} inputMode="numeric" /></label>
        <label className="text-[12px] text-ink-2">Hardware<Input value={hardware} onChange={(event) => setHardware(event.target.value)} placeholder="NVIDIA GB10 · 128 GB" /></label>
        <label className="text-[12px] text-ink-2">Parameters<Input value={parameters} onChange={(event) => setParameters(event.target.value)} placeholder="27B" /></label>
        <label className="text-[12px] text-ink-2">Quantization<Input value={quantization} onChange={(event) => setQuantization(event.target.value)} placeholder="EXL3 3.5 bpw" /></label>
        <label className="text-[12px] text-ink-2">Context tokens<Input value={contextTokens} onChange={(event) => setContextTokens(event.target.value)} inputMode="numeric" placeholder="131072" /></label>
        <label className="text-[12px] text-ink-2">Tokens/s measured<Input value={tokensPerSecond} onChange={(event) => setTokensPerSecond(event.target.value)} inputMode="decimal" placeholder="32" /></label>
        <label className="text-[12px] text-ink-2">First token (ms)<Input value={firstTokenMs} onChange={(event) => setFirstTokenMs(event.target.value)} inputMode="numeric" placeholder="480" /></label>
      </div>
      <div className="mt-3"><Button size="sm" variant="primary" onClick={create}>Create card</Button></div>
      {made && <div className="mt-3 space-y-3">
        <div>
          <div className="flex items-center justify-between gap-2"><span className="text-[12px] font-medium text-ink">Card to hand over</span><Button size="xs" variant="ghost" onClick={() => void copy(JSON.stringify(made.card, null, 2)).then(() => setNotice('Card copied.'), (failure) => setError(String(failure.message ?? failure)))}>Copy</Button></div>
          <pre className="scroll-thin mt-1 max-h-48 overflow-auto rounded-lg border border-line bg-card p-3 text-[11px] text-ink-2">{JSON.stringify(made.card, null, 2)}</pre>
        </div>
        <div>
          <div className="flex items-center justify-between gap-2"><span className="text-[12px] font-medium text-ink">Donor config</span><Button size="xs" variant="ghost" onClick={() => void copy(made.config).then(() => setNotice('Config copied.'), (failure) => setError(String(failure.message ?? failure)))}>Copy</Button></div>
          <pre className="scroll-thin mt-1 max-h-48 overflow-auto rounded-lg border border-line bg-card p-3 text-[11px] text-ink-2">{made.config}</pre>
          <p className="mt-1 text-[11px] text-ink-3">Run on the donor: <code>node donation-node.js --config donation.json</code></p>
        </div>
      </div>}
    </section>

    <section>
      <h3 className="text-[14px] font-medium text-ink">Borrow a machine</h3>
      <p className="mt-1 text-[12px] text-ink-2">Paste a card. The app checks the donor answers with its token and model before adding it as a provider.</p>
      <textarea value={cardText} onChange={(event) => setCardText(event.target.value)} rows={5} spellCheck={false} className="mt-2 w-full rounded-lg border border-line bg-card px-3 py-2 font-mono text-[11px] text-ink outline-none focus:border-accent" placeholder={'{\n  "schema": "tensorfold-donation/1",\n  ...\n}'} />
      <div className="mt-2"><Button size="sm" variant="primary" disabled={!cardText.trim() || busy === 'import'} onClick={() => void borrow()}>{busy === 'import' ? 'Checking…' : 'Import card'}</Button></div>
    </section>

    <section>
      <h3 className="text-[14px] font-medium text-ink">Borrowed machines</h3>
      {cards.length === 0 ? <p className="mt-2 text-[13px] text-ink-2">None yet.</p> : <ul className="mt-2 space-y-2">
        {cards.map((card) => <li key={card.id} className="rounded-lg border border-line bg-card p-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="min-w-0">
              <p className="truncate text-[13px] font-medium text-ink">{card.label} <span className="font-normal text-ink-3">· {card.model}</span></p>
              <p className="mt-0.5 break-all text-[11px] text-ink-3">{card.endpoint} · {card.phase === 'open' ? remaining(card.remainingMs) : card.phase === 'before' ? 'opens later' : 'closed'}</p>
            </div>
            <div className="flex gap-1">
              <Button size="xs" variant="ghost" disabled={busy === card.id} onClick={() => void check(card.id)}>Check</Button>
              <Button size="xs" variant="ghost" disabled={busy === card.id} onClick={() => void remove(card.id)}>Remove</Button>
            </div>
          </div>
        </li>)}
      </ul>}
      {usable.length === 0 && cards.length > 0 && <p className="mt-2 text-[12px] text-ink-3">Every card here has expired; remove it before relying on the provider list.</p>}
      {snapshot?.errors.map((message) => <p key={message} className="mt-2 text-[12px] text-warn">{message}</p>)}
    </section>
  </div>
}
