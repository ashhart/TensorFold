import { useEffect, useState } from 'react'
import { Button } from '../ui'
import { api } from '../../lib/ipc'
import { useStore } from '../../lib/store'
import { formatBytes } from '../../lib/engine'
import type { DonationCardView } from '../../../shared/donation'

/**
 * The community catalogue: models other people host through TensorFold, with the specs their
 * card declares. Picking one opens a chat against that host; the host enforces its own window
 * and caps. POC: no payment and no proof of computation.
 */

function left(ms: number): string {
  if (ms <= 0) return 'closed'
  const minutes = Math.round(ms / 60_000)
  return minutes < 60 ? `${minutes} min left` : `${Math.floor(minutes / 60)} h ${minutes % 60} min left`
}

function Row({ label, value }: { label: string; value: string }) {
  return <div className="flex items-baseline justify-between gap-3 text-[12px]"><span className="text-ink-3">{label}</span><span className="text-right text-ink-2">{value}</span></div>
}

export function CommunityModels({ onOpen, onPicked, heading = 'TensorFold Community', intro = 'Models hosted by other TensorFold users. Each card states the hardware, model and measured speed; the host enforces the window and caps. POC — free, with no proof of computation.' }: { onOpen?: () => void; onPicked?: () => void; heading?: string; intro?: string }) {
  const [cards, setCards] = useState<DonationCardView[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  const load = () => {
    api.donationList().then((value) => { setCards(value.cards); setError(value.errors[0] ?? null) }, (failure) => setError(String((failure as Error)?.message ?? failure)))
  }
  useEffect(() => {
    load()
    // Importing or removing a card broadcasts a settings change; window badges also age, so
    // refresh on both rather than only on mount.
    const off = api.onDesktopModeChanged?.(() => load())
    const timer = setInterval(load, 30_000)
    return () => { off?.(); clearInterval(timer) }
  }, [])

  const use = async (card: DonationCardView) => {
    setBusy(card.id); setError(null); setNotice(null)
    try {
      if (card.phase === 'after') throw new Error('That host\'s window has closed; ask them for a new card.')
      const before = new Set(useStore.getState().threads.map((row) => row.id))
      await useStore.getState().newChat(null)
      const thread = useStore.getState().threads.find((row) => !before.has(row.id))
      if (!thread) throw new Error('The new chat did not open; try again.')
      await useStore.getState().setThreadProvider(thread.id, card.providerId, card.model, null)
      setNotice(`New chat on ${card.label}. The host enforces its window and caps.`)
      onPicked?.()
    } catch (failure) {
      setError(String((failure as Error)?.message ?? failure))
    } finally { setBusy(null) }
  }

  const check = async (card: DonationCardView) => {
    setBusy(card.id); setError(null); setNotice(null)
    try {
      const value = await api.donationProbe({ id: card.id })
      setNotice(value.ok ? `${card.label} answers with ${value.models?.join(', ') ?? value.model}.` : `${card.label}: ${value.error}`)
    } catch (failure) {
      setError(String((failure as Error)?.message ?? failure))
    } finally { setBusy(null) }
  }

  const live = (cards ?? []).filter((card) => card.phase !== 'after')

  return <section data-testid="community-models" className="mt-5 rounded-xl border border-line-strong bg-card p-4">
    <div className="flex flex-wrap items-start justify-between gap-2">
      <div>
        <h2 className="text-[15px] font-medium text-ink">{heading}</h2>
        <p className="mt-1 max-w-2xl text-[12px] leading-relaxed text-ink-2">{intro}</p>
      </div>
      {onOpen && <Button size="sm" variant="default" onClick={onOpen}>{live.length ? 'Lend or import' : 'Lend or import a card'}</Button>}
    </div>

    {(error || notice) && <p role="status" className={error ? 'mt-3 rounded-lg border border-danger/25 bg-danger/10 px-3 py-2 text-[12px] text-danger' : 'mt-3 rounded-lg border border-line bg-card-2 px-3 py-2 text-[12px] text-ink-2'}>{error ?? notice}</p>}

    {cards === null ? <p className="mt-3 text-[13px] text-ink-2">Reading borrowed machines…</p> : live.length === 0 ? <p className="mt-3 text-[13px] text-ink-2">No community machines are lent to this profile yet. Lend this machine, or paste a card somebody handed you.</p> : <div className="mt-3 grid gap-3 sm:grid-cols-2">
      {live.map((card) => <article key={card.id} className="rounded-lg border border-line bg-card-2 p-3">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <h3 className="truncate text-[13px] font-medium text-ink">{card.model}</h3>
            <p className="truncate text-[12px] text-ink-3">{card.label}</p>
          </div>
          <span className={card.phase === 'open' ? 'shrink-0 rounded-md bg-accent/10 px-2 py-0.5 text-[11px] text-accent' : 'shrink-0 rounded-md bg-surface px-2 py-0.5 text-[11px] text-ink-3'}>{card.phase === 'open' ? left(card.remainingMs) : 'opens later'}</span>
        </div>
        <div className="mt-2 space-y-0.5">
          {card.spec?.hardware && <Row label="Hardware" value={card.spec.hardware} />}
          {card.spec?.parameters && <Row label="Parameters" value={card.spec.parameters} />}
          {card.spec?.quantization && <Row label="Quantization" value={card.spec.quantization} />}
          {card.spec?.contextTokens ? <Row label="Context" value={`${card.spec.contextTokens.toLocaleString()} tokens`} /> : null}
          {card.spec?.tokensPerSecond ? <Row label="Decode" value={`${card.spec.tokensPerSecond} tok/s`} /> : null}
          {card.spec?.firstTokenMs ? <Row label="First token" value={`${card.spec.firstTokenMs} ms`} /> : null}
          {card.spec?.engineVersion && <Row label="Engine" value={card.spec.engineVersion} />}
          {card.spec?.memoryBytes ? <Row label="Memory" value={formatBytes(card.spec.memoryBytes)} /> : null}
          <Row label="Caps" value={`${card.caps.maxConcurrent} at a time · ${card.caps.maxTokensPerRequest} tokens`} />
          <Row label="Endpoint" value={card.endpoint.replace(/^https?:\/\//, '')} />
        </div>
        {card.spec?.notes && <p className="mt-2 border-t border-line pt-2 text-[11px] leading-relaxed text-ink-3">{card.spec.notes}</p>}
        <div className="mt-2 flex gap-1">
          <Button size="xs" variant="primary" disabled={busy === card.id} onClick={() => void use(card)}>{busy === card.id ? 'Working…' : 'Use'}</Button>
          <Button size="xs" variant="ghost" disabled={busy === card.id} onClick={() => void check(card)}>Check</Button>
        </div>
      </article>)}
    </div>}
  </section>
}
