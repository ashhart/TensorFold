/**
 * Supervises the donor on this machine. The app builds the card, starts the donor proxy inside
 * the main process (a packaged Electron app cannot spawn itself as Node: the runAsNode fuse is
 * off), counts each receipt, and stops lending when the window closes or the user switches it off.
 */
import { randomBytes } from 'node:crypto'
import { handle } from './ipc'
import { requireSecurityRole } from './audit'
import { startDonationNode, type DonationNodeHandle } from '../../donation-node-main'
import { makeDonationCard, type DonationCard, type DonationHostInput, type DonationHostStatus } from '../../../shared/donation'

const MAX_HOURS = 24 * 30
let state: DonationHostStatus = { state: 'idle' }
let running: DonationNodeHandle | null = null
let expiry: NodeJS.Timeout | null = null
const LOOPBACK = new Set(['127.0.0.1', 'localhost', '::1'])

function assertInput(input: unknown): DonationHostInput {
  const value = input as Partial<DonationHostInput> | null
  if (!value || typeof value !== 'object') throw new Error('The donation settings are missing.')
  const label = typeof value.label === 'string' ? value.label.trim() : ''
  if (!label || label.length > 120) throw new Error('Give this machine a label of 1 to 120 characters.')
  const upstream = typeof value.upstream === 'string' ? value.upstream.trim() : ''
  let origin: URL
  try { origin = new URL(upstream) } catch { throw new Error('The local engine address is not a URL.') }
  if (origin.protocol !== 'http:' && origin.protocol !== 'https:') throw new Error('The local engine address must be http or https.')
  if (origin.username || origin.password || origin.search || origin.hash || (origin.pathname !== '/' && origin.pathname !== '')) throw new Error('The local engine address must be an origin, e.g. http://127.0.0.1:8000.')
  const model = typeof value.model === 'string' ? value.model.trim() : ''
  if (!model) throw new Error('Name the model this machine serves.')
  const listenPort = Number(value.listenPort)
  if (!Number.isSafeInteger(listenPort) || listenPort < 1024 || listenPort > 65535) throw new Error('Choose a listen port from 1024 to 65535.')
  const endpointHost = typeof value.endpointHost === 'string' && value.endpointHost.trim() ? value.endpointHost.trim() : '127.0.0.1'
  if (!LOOPBACK.has(endpointHost)) throw new Error('The app lends over loopback only; a renter reaches it through an SSH tunnel or a TLS proxy.')
  const hours = Number(value.hours)
  if (!Number.isFinite(hours) || hours <= 0 || hours > MAX_HOURS) throw new Error(`Choose a window from 1 hour to ${MAX_HOURS / 24} days.`)
  const caps = value.caps
  if (caps !== undefined && (typeof caps !== 'object' || caps === null || Array.isArray(caps))) throw new Error('The caps must be an object.')
  const upstreamToken = typeof value.upstreamToken === 'string' ? value.upstreamToken.trim() : undefined
  if (upstreamToken !== undefined && (upstreamToken.length < 8 || upstreamToken.length > 512 || /\s/.test(upstreamToken))) throw new Error('The engine token must be 8 to 512 characters with no whitespace.')
  return { label, upstream: origin.origin, ...(upstreamToken ? { upstreamToken } : {}), model, listenPort, endpointHost, hours, caps, spec: value.spec, upstreamTimeoutMs: value.upstreamTimeoutMs }
}

function snapshot(): DonationHostStatus {
  return { ...state, ...(state.card ? { card: { ...state.card } } : {}) }
}

async function stop(): Promise<DonationHostStatus> {
  if (expiry) { clearTimeout(expiry); expiry = null }
  const ending = running
  running = null
  if (ending) await ending.close().catch(() => {})
  state = { state: 'idle' }
  return snapshot()
}

async function start(input: unknown): Promise<DonationHostStatus> {
  if (running) throw new Error('This machine is already lending. Stop it before changing the settings.')
  const settings = assertInput(input)
  const now = Date.now()
  const made = makeDonationCard({
    id: `donate-${randomBytes(6).toString('hex')}`,
    label: settings.label,
    endpoint: `http://127.0.0.1:${settings.listenPort}`,
    token: randomBytes(24).toString('hex'),
    model: settings.model,
    window: { from: now, until: now + Math.round(settings.hours * 3_600_000) },
    caps: settings.caps ?? {},
    ...(settings.spec ? { spec: settings.spec } : {}),
    security: 'poc-insecure-lan',
    createdAt: now,
  })
  if ('error' in made) throw new Error(made.error)
  const card: DonationCard = made.card
  state = { state: 'starting', card, startedAt: now, until: card.window.until, receipts: 0 }
  try {
    running = await startDonationNode(
      { upstream: settings.upstream, ...(settings.upstreamToken ? { upstreamToken: settings.upstreamToken } : {}), listen: { host: '127.0.0.1', port: settings.listenPort }, card, ...(settings.upstreamTimeoutMs ? { upstreamTimeoutMs: settings.upstreamTimeoutMs } : {}) },
      { onReceipt: () => { state = { ...state, receipts: (state.receipts ?? 0) + 1 } } },
    )
  } catch (error) {
    state = { state: 'failed', card, error: (error as Error).message }
    throw new Error(`The donor did not start: ${(error as Error).message}`)
  }
  state = { state: 'running', card, startedAt: now, until: card.window.until, receipts: 0 }
  expiry = setTimeout(() => { void stop() }, Math.max(1_000, card.window.until - Date.now()))
  expiry.unref?.()
  return snapshot()
}

export function registerDonationHost(): void {
  handle('donation:hostStart', async (_event, input?: unknown) => {
    requireSecurityRole('owner', 'lend this machine')
    return start(input)
  })
  handle('donation:hostStop', async () => {
    requireSecurityRole('owner', 'stop lending this machine')
    return stop()
  })
  handle('donation:hostStatus', async () => {
    requireSecurityRole('owner', 'review this machine as a donor')
    return snapshot()
  })
}
