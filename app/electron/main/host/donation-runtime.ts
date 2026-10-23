/**
 * Borrowing side of donation cards: import a card somebody handed over, prove it answers, and
 * register it as a provider. POC scope — no ledger, no proof of computation, no directory.
 */
import { handle } from './ipc'
import { requireSecurityRole } from './audit'
import { store } from '../state'
import { saveSettings, vontraHomeDir } from '../../store'
import { sendToRenderer } from './renderer-events'
import { assertProviderEndpoint, redactSettings, resolveProviderRequest } from '../../agent/security'
import { createPinnedFetch } from '../../agent/pinned-fetch'
import { donationPhase, donationProvider, donationProviderId, donationRemainingMs, parseDonationCard, type DonationCard, type DonationCardView, type DonationSnapshot } from '../../../shared/donation'
import { loadDonationCards, saveDonationCards } from './donation/store'

interface RuntimeOptions {
  serializeSettings<T>(operation: () => Promise<T>): Promise<T>
}

/** A donor's answer is untrusted input: read at most this much of it. */
const MAX_PROBE_BYTES = 256 * 1024
const MAX_MODELS = 200
const MAX_MODEL_ID = 200

async function readBounded(answer: Response): Promise<string | null> {
  if (!answer.body) return ''
  const reader = answer.body.getReader()
  const chunks: Buffer[] = []
  let total = 0
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      total += value.byteLength
      if (total > MAX_PROBE_BYTES) { await reader.cancel().catch(() => {}); return null }
      chunks.push(Buffer.from(value))
    }
  } finally {
    reader.releaseLock()
  }
  return Buffer.concat(chunks).toString('utf8')
}

function view(card: DonationCard, now = Date.now()): DonationCardView {
  return {
    id: card.id,
    label: card.label,
    endpoint: card.endpoint,
    model: card.model,
    ...(card.revision === undefined ? {} : { revision: card.revision }),
    window: card.window,
    caps: card.caps,
    ...(card.spec === undefined ? {} : { spec: card.spec }),
    security: card.security,
    phase: donationPhase(card, now),
    remainingMs: donationRemainingMs(card, now),
    providerId: donationProviderId(card),
    createdAt: card.createdAt,
  }
}

async function probe(card: DonationCard, signal: AbortSignal): Promise<{ ok: true; models: string[]; pending: boolean } | { ok: false; error: string }> {
  // A pasted card is untrusted input to this process: https, or plain HTTP only to loopback,
  // never link-local or metadata. Each request is pinned to the address it resolved to.
  const provider = donationProvider(card)
  assertProviderEndpoint(provider)
  const send = async (rawUrl: string, init: RequestInit) => {
    const target = await resolveProviderRequest(provider, rawUrl)
    return createPinnedFetch(target)(rawUrl, { ...init, redirect: 'manual', signal })
  }
  try {
    const health = await send(`${card.endpoint}/health`, { method: 'GET' })
    if (health.status >= 300 && health.status < 400) { await health.body?.cancel().catch(() => {}); return { ok: false, error: 'The endpoint redirected; no request was sent further.' } }
    const text = health.ok ? await readBounded(health) : null
    await health.body?.cancel().catch(() => {})
    if (text === null) return { ok: false, error: "The donor's health answer was larger than this app reads." }
    let phase: unknown
    try { phase = text ? (JSON.parse(text) as { phase?: unknown }).phase : undefined } catch { /* not a donation node: fall through to the model probe */ }
    // A window that has not opened refuses inference on purpose; the card is accepted now and
    // verified by the first request inside its window.
    if (phase === 'before') return { ok: true, models: [], pending: true }
  } catch (error) {
    return { ok: false, error: `The endpoint did not answer: ${(error as Error).message}` }
  }
  try {
    const answer = await send(`${card.endpoint}/v1/models`, { method: 'GET', headers: { authorization: `Bearer ${card.token}` } })
    if (answer.status === 401 || answer.status === 403) { await answer.body?.cancel().catch(() => {}); return { ok: false, error: 'The card token was refused by the donor.' } }
    if (!answer.ok) { await answer.body?.cancel().catch(() => {}); return { ok: false, error: `The donor answered /v1/models with HTTP ${answer.status}.` } }
    const text = await readBounded(answer)
    await answer.body?.cancel().catch(() => {})
    if (text === null) return { ok: false, error: "The donor's model list was larger than this app reads." }
    let data: unknown
    try { data = (JSON.parse(text) as { data?: unknown }).data } catch { return { ok: false, error: "The donor's model list is not JSON." } }
    const models = Array.isArray(data) ? data.slice(0, MAX_MODELS).map((row) => String((row as { id?: unknown })?.id ?? '').slice(0, MAX_MODEL_ID)).filter(Boolean) : []
    if (!models.includes(card.model)) return { ok: false, error: `The donor does not list the card's model \`${card.model}\`. It lists: ${models.slice(0, 5).join(', ') || 'none'}.` }
    return { ok: true, models, pending: false }
  } catch (error) {
    return { ok: false, error: `The donor's model list could not be read: ${(error as Error).message}` }
  }
}

/** Add or replace the provider row. The caller holds the settings serialization. */
async function putProvider(card: DonationCard): Promise<void> {
  const provider = donationProvider(card)
  const previous = store.settings, captured = JSON.stringify(previous)
  const next = { ...previous, providers: previous.providers.filter((row: { id: string }) => row.id !== provider.id).concat(provider) }
  const validateCommit = () => { if (store.settings !== previous || JSON.stringify(previous) !== captured) throw new Error('Provider settings changed while saving the borrowed machine.') }
  validateCommit()
  await saveSettings(next, { validateCommit })
  validateCommit()
  store.settings = next
  sendToRenderer('desktop:modeChanged', redactSettings(next))
}

/** Remove a provider row. The caller holds the settings serialization. */
async function dropProvider(providerId: string): Promise<void> {
  const previous = store.settings, captured = JSON.stringify(previous)
  if (!previous.providers.some((row: { id: string }) => row.id === providerId)) return
  const next = { ...previous, providers: previous.providers.filter((row: { id: string }) => row.id !== providerId) }
  const validateCommit = () => { if (store.settings !== previous || JSON.stringify(previous) !== captured) throw new Error('Provider settings changed while removing the borrowed machine.') }
  validateCommit()
  await saveSettings(next, { validateCommit })
  validateCommit()
  store.settings = next
  sendToRenderer('desktop:modeChanged', redactSettings(next))
}

export function registerDonationRuntime(options: RuntimeOptions): void {
  const owner = (action: string) => requireSecurityRole('owner', action)

  handle('donation:list', async (): Promise<DonationSnapshot> => {
    owner('review borrowed machines')
    const { cards, errors } = await loadDonationCards(vontraHomeDir())
    return { cards: cards.map((card) => view(card)), errors }
  })

  handle('donation:import', async (_event, input?: unknown): Promise<DonationSnapshot> => {
    owner('import a donation card')
    const text = typeof (input as { text?: unknown })?.text === 'string' ? (input as { text: string }).text : ''
    const parsed = parseDonationCard(text)
    if ('error' in parsed) throw new Error(parsed.error)
    const card = parsed.card
    if (donationPhase(card) === 'after') throw new Error('That card\'s window has already closed.')
    const checked = await probe(card, AbortSignal.timeout(8_000))
    if (!checked.ok) throw new Error(checked.error)
    const root = vontraHomeDir()
    // One serialized mutation: the card and its provider row land together or not at all.
    const next = await options.serializeSettings(async () => {
      const { cards } = await loadDonationCards(root)
      const updated = cards.filter((row) => row.id !== card.id).concat(card)
      await saveDonationCards(root, updated)
      try {
        await putProvider(card)
      } catch (error) {
        await saveDonationCards(root, cards).catch(() => {})
        throw error
      }
      return updated
    })
    return { cards: next.map((row) => view(row)), errors: [] }
  })

  handle('donation:remove', async (_event, input?: unknown): Promise<DonationSnapshot> => {
    owner('remove a borrowed machine')
    const id = typeof (input as { id?: unknown })?.id === 'string' ? (input as { id: string }).id : ''
    if (!id) throw new Error('Choose a borrowed machine to remove.')
    const root = vontraHomeDir()
    const next = await options.serializeSettings(async () => {
      const { cards } = await loadDonationCards(root)
      const card = cards.find((row) => row.id === id)
      if (!card) throw new Error('That borrowed machine is not in this profile.')
      const updated = cards.filter((row) => row.id !== id)
      await saveDonationCards(root, updated)
      try {
        await dropProvider(donationProviderId(card))
      } catch (error) {
        await saveDonationCards(root, cards).catch(() => {})
        throw error
      }
      return updated
    })
    return { cards: next.map((row) => view(row)), errors: [] }
  })

  handle('donation:probe', async (_event, input?: unknown) => {
    owner('check a borrowed machine')
    const id = typeof (input as { id?: unknown })?.id === 'string' ? (input as { id: string }).id : ''
    const { cards } = await loadDonationCards(vontraHomeDir())
    const card = cards.find((row) => row.id === id)
    if (!card) throw new Error('That borrowed machine is not in this profile.')
    const checked = await probe(card, AbortSignal.timeout(8_000))
    return { ...view(card), ...(checked.ok ? { ok: true, models: checked.models } : { ok: false, error: checked.error }) }
  })
}
