/**
 * Donation cards: a machine lent for a window, with caps and no payment.
 *
 * A card is the whole contract between two people's TensorFold installs: where to reach the
 * lent engine, the token the renter presents, which model it serves, when the window is open,
 * and the caps the donor enforces. It is pure data so both processes and the renderer agree.
 *
 * POC scope: no money, no proof of computation, no public directory. A card is handed over
 * out of band; `security: 'poc-insecure-lan'` is the only way to lend over plain HTTP, and it
 * is named so a UI can badge it.
 */

export const DONATION_SCHEMA = 'tensorfold-donation/1'

export interface DonationWindow {
  /** Epoch milliseconds. */
  from: number
  /** Epoch milliseconds; must be after `from`. */
  until: number
}

export interface DonationCaps {
  maxConcurrent: number
  maxTokensPerRequest: number
  maxRequestBytes: number
}

/** What the donor says the lent machine is and how it performs. All optional; shown, not trusted. */
export interface DonationSpec {
  /** e.g. "NVIDIA GB10 · 128 GB unified" */
  hardware?: string
  /** Bytes of device or unified memory available to the model. */
  memoryBytes?: number
  /** e.g. "27B", "8B", "3B-A3B" */
  parameters?: string
  /** e.g. "EXL3 3.5 bpw", "Q4_K_M", "MLX 4-bit" */
  quantization?: string
  contextTokens?: number
  tokensPerSecond?: number
  firstTokenMs?: number
  engineVersion?: string
  notes?: string
}

export interface DonationCard {
  schema: typeof DONATION_SCHEMA
  /** Stable id chosen by the donor; provider ids are `donation:<id>`. */
  id: string
  label: string
  /** Origin only, e.g. `http://192.168.0.45:8010`. No path, query, hash or credentials. */
  endpoint: string
  token: string
  model: string
  revision?: string
  window: DonationWindow
  caps: DonationCaps
  spec?: DonationSpec
  security: 'tls' | 'poc-insecure-lan'
  createdAt: number
}

export type DonationPhase = 'before' | 'open' | 'after'
export type DonationParse = { card: DonationCard } | { error: string }

const CARD_KEYS = new Set(['schema', 'id', 'label', 'endpoint', 'token', 'model', 'revision', 'window', 'caps', 'spec', 'security', 'createdAt'])
const SPEC_KEYS = new Set(['hardware', 'memoryBytes', 'parameters', 'quantization', 'contextTokens', 'tokensPerSecond', 'firstTokenMs', 'engineVersion', 'notes'])
const WINDOW_KEYS = new Set(['from', 'until'])
const CAPS_KEYS = new Set(['maxConcurrent', 'maxTokensPerRequest', 'maxRequestBytes'])

const LIMITS = {
  maxConcurrent: [1, 64],
  maxTokensPerRequest: [1, 131_072],
  maxRequestBytes: [1_024, 32 * 1024 * 1024],
} as const

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function unknownKeys(value: Record<string, unknown>, allowed: Set<string>): string[] {
  return Object.keys(value).filter((key) => !allowed.has(key))
}

function safeInteger(value: unknown, min: number, max: number): number | null {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < min || value > max) return null
  return value
}

function parseEndpoint(raw: unknown): { value: string; origin: string; https: boolean } | { error: string } {
  if (typeof raw !== 'string' || raw.length === 0 || raw.length > 512) return { error: 'The card endpoint is missing or too long.' }
  let url: URL
  try {
    url = new URL(raw)
  } catch {
    return { error: 'The card endpoint is not a URL.' }
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return { error: 'The card endpoint must be http or https.' }
  if (url.username || url.password) return { error: 'The card endpoint must not carry credentials.' }
  if (url.search || url.hash) return { error: 'The card endpoint must not carry a query or a fragment.' }
  // A relay serves one card per path segment, so `https://relay.example/donate-abc123` is an
  // endpoint too. Anything deeper is a mistake: the app appends `/v1` itself.
  const path = url.pathname.replace(/\/+$/, '')
  if (path !== '' && !/^\/[a-z0-9][a-z0-9-]{2,63}$/.test(path)) {
    return { error: 'The card endpoint may carry at most one path segment: the card id a relay serves.' }
  }
  return { value: url.origin + path, origin: url.origin, https: url.protocol === 'https:' }
}

function parseSpec(raw: unknown): DonationSpec | { error: string } | undefined {
  if (raw === undefined) return undefined
  if (!isRecord(raw)) return { error: 'The card spec must be an object.' }
  const extra = unknownKeys(raw, SPEC_KEYS)
  if (extra.length) return { error: `The card spec has unknown fields: ${extra.join(', ')}.` }
  const spec: DonationSpec = {}
  for (const key of ['hardware', 'parameters', 'quantization', 'engineVersion', 'notes'] as const) {
    const value = raw[key]
    if (value === undefined) continue
    if (typeof value !== 'string' || value.length === 0 || value.length > 200) return { error: `The card spec \`${key}\` must be 1 to 200 characters.` }
    spec[key] = value
  }
  for (const key of ['memoryBytes', 'contextTokens'] as const) {
    const value = raw[key]
    if (value === undefined) continue
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value <= 0) return { error: `The card spec \`${key}\` must be a positive integer.` }
    spec[key] = value
  }
  for (const key of ['tokensPerSecond', 'firstTokenMs'] as const) {
    const value = raw[key]
    if (value === undefined) continue
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return { error: `The card spec \`${key}\` must be a non-negative number.` }
    spec[key] = value
  }
  return spec
}

function parseWindow(raw: unknown, now: number): DonationWindow | { error: string } {
  if (!isRecord(raw)) return { error: 'The card window is missing.' }
  const extra = unknownKeys(raw, WINDOW_KEYS)
  if (extra.length) return { error: `The card window has unknown fields: ${extra.join(', ')}.` }
  const from = safeInteger(raw.from, 0, 8.64e15)
  const until = safeInteger(raw.until, 0, 8.64e15)
  if (from === null || until === null) return { error: 'The card window needs integer `from` and `until` milliseconds.' }
  if (until <= from) return { error: 'The card window ends before it starts.' }
  if (until - from > 366 * 24 * 60 * 60 * 1000) return { error: 'The card window is longer than a year.' }
  if (until < now - 365 * 24 * 60 * 60 * 1000) return { error: 'The card window is more than a year in the past.' }
  return { from, until }
}

function parseCaps(raw: unknown): DonationCaps | { error: string } {
  if (!isRecord(raw)) return { error: 'The card caps are missing.' }
  const extra = unknownKeys(raw, CAPS_KEYS)
  if (extra.length) return { error: `The card caps have unknown fields: ${extra.join(', ')}.` }
  const caps = {} as DonationCaps
  for (const key of CAPS_KEYS) {
    const [min, max] = LIMITS[key as keyof typeof LIMITS]
    const value = safeInteger(raw[key], min, max)
    if (value === null) return { error: `The card cap \`${key}\` must be an integer from ${min} to ${max}.` }
    caps[key as keyof DonationCaps] = value
  }
  return caps
}

/** Parse and validate a card from JSON text or an already-decoded value. */
export function parseDonationCard(raw: unknown, now = Date.now()): DonationParse {
  let value = raw
  if (typeof raw === 'string') {
    if (raw.length > 8_192) return { error: 'The card text is too long.' }
    try {
      value = JSON.parse(raw)
    } catch {
      return { error: 'The card is not valid JSON.' }
    }
  }
  if (!isRecord(value)) return { error: 'The card must be a JSON object.' }
  const extra = unknownKeys(value, CARD_KEYS)
  if (extra.length) return { error: `The card has unknown fields: ${extra.join(', ')}.` }
  if (value.schema !== DONATION_SCHEMA) return { error: `The card schema must be ${DONATION_SCHEMA}.` }
  const id = typeof value.id === 'string' ? value.id.trim() : ''
  if (!/^[a-z0-9][a-z0-9-]{2,63}$/.test(id)) return { error: 'The card id must be 3 to 64 lowercase letters, digits or hyphens.' }
  const label = typeof value.label === 'string' ? value.label.trim() : ''
  if (!label || label.length > 120) return { error: 'The card label must be 1 to 120 characters.' }
  const endpoint = parseEndpoint(value.endpoint)
  if ('error' in endpoint) return endpoint
  const token = typeof value.token === 'string' ? value.token : ''
  if (token.length < 16 || token.length > 512 || /\s/.test(token)) return { error: 'The card token must be 16 to 512 characters with no whitespace.' }
  const model = typeof value.model === 'string' ? value.model.trim() : ''
  if (!model || model.length > 200 || /[\u0000-\u001f]/.test(model)) return { error: 'The card model must be 1 to 200 printable characters.' }
  const revision = value.revision === undefined ? undefined : typeof value.revision === 'string' && value.revision.length <= 200 ? value.revision : null
  if (revision === null) return { error: 'The card revision must be a string of at most 200 characters.' }
  const window = parseWindow(value.window, now)
  if ('error' in window) return window
  const caps = parseCaps(value.caps)
  if ('error' in caps) return caps
  const spec = parseSpec(value.spec)
  if (spec && 'error' in spec) return spec
  const security = value.security
  if (security !== 'tls' && security !== 'poc-insecure-lan') return { error: 'The card security must be `tls` or `poc-insecure-lan`.' }
  if (security === 'tls' && !endpoint.https) return { error: 'A `tls` card must use an https endpoint.' }
  if (security === 'poc-insecure-lan' && endpoint.https) return { error: 'A `poc-insecure-lan` card must use a plain http endpoint.' }
  const createdAt = safeInteger(value.createdAt, 0, 8.64e15)
  if (createdAt === null) return { error: 'The card needs a `createdAt` timestamp in milliseconds.' }
  return {
    card: {
      schema: DONATION_SCHEMA,
      id,
      label,
      endpoint: endpoint.value,
      token,
      model,
      ...(revision === undefined ? {} : { revision }),
      window,
      caps,
      ...(spec === undefined ? {} : { spec }),
      security,
      createdAt,
    },
  }
}

/** Where the card sits relative to `now`. */
export function donationPhase(card: DonationCard, now = Date.now()): DonationPhase {
  if (now < card.window.from) return 'before'
  if (now >= card.window.until) return 'after'
  return 'open'
}

export function donationOpen(card: DonationCard, now = Date.now()): boolean {
  return donationPhase(card, now) === 'open'
}

/** The OpenAI-compatible base URL the app's provider entry uses. */
export function donationBaseUrl(card: DonationCard): string {
  return `${card.endpoint}/v1`
}

export function donationProviderId(card: DonationCard): string {
  return `donation:${card.id}`
}

/**
 * The provider row a card becomes. Kept identical in shape to the managed compute provider so
 * the router, the outbound guard and the spend budget treat a lent machine the same way:
 * remote computation, chat-completions transport, a bearer key.
 */
export function donationProvider(card: DonationCard): {
  id: string
  name: string
  transport: 'chat-completions'
  auth: 'api-key'
  baseUrl: string
  apiKey: string
  models: string[]
  kind: 'cloud'
  remoteCompute: true
  source: 'custom'
} {
  return {
    id: donationProviderId(card),
    name: card.label,
    transport: 'chat-completions',
    auth: 'api-key',
    baseUrl: donationBaseUrl(card),
    apiKey: card.token,
    models: [card.model],
    kind: 'cloud',
    remoteCompute: true,
    source: 'custom',
  }
}

/** A card the donor can hand over, from the values collected for lending. */
export function makeDonationCard(input: {
  id: string
  label: string
  endpoint: string
  token: string
  model: string
  revision?: string
  window: DonationWindow
  caps: Partial<DonationCaps>
  spec?: DonationSpec
  security: DonationCard['security']
  createdAt?: number
}): DonationParse {
  return parseDonationCard({
    schema: DONATION_SCHEMA,
    id: input.id,
    label: input.label,
    endpoint: input.endpoint,
    token: input.token,
    model: input.model,
    ...(input.revision === undefined ? {} : { revision: input.revision }),
    window: input.window,
    caps: {
      maxConcurrent: input.caps.maxConcurrent ?? 1,
      maxTokensPerRequest: input.caps.maxTokensPerRequest ?? 1_024,
      maxRequestBytes: input.caps.maxRequestBytes ?? 1024 * 1024,
    },
    ...(input.spec === undefined ? {} : { spec: input.spec }),
    security: input.security,
    createdAt: input.createdAt ?? Date.now(),
  })
}

/** Remaining milliseconds in an open window, or 0 before it opens; negative after it closes. */
export function donationRemainingMs(card: DonationCard, now = Date.now()): number {
  return card.window.until - Math.max(now, card.window.from)
}

/** A card as the renderer sees it: the token is deliberately absent. */
export interface DonationCardView {
  id: string
  label: string
  endpoint: string
  model: string
  revision?: string
  window: DonationWindow
  caps: DonationCaps
  spec?: DonationSpec
  security: DonationCard['security']
  phase: DonationPhase
  remainingMs: number
  providerId: string
  createdAt: number
}

export interface DonationSnapshot {
  cards: DonationCardView[]
  errors: string[]
}

export interface DonationProbeView extends DonationCardView {
  ok: boolean
  models?: string[]
  error?: string
}

/** Starting the donor process on this machine, supervised by the app. */
export interface DonationHostInput {
  label: string
  /** Local engine origin, e.g. http://127.0.0.1:8000. */
  upstream: string
  /** Bearer token the local engine requires, when it requires one. */
  upstreamToken?: string
  model: string
  /** Where the donor listens; renters reach it through a tunnel or TLS. */
  listenPort: number
  endpointHost?: string
  hours: number
  caps?: Partial<DonationCaps>
  spec?: DonationSpec
  upstreamTimeoutMs?: number
}

export interface DonationHostStatus {
  state: 'idle' | 'starting' | 'running' | 'stopping' | 'failed'
  /** The card to hand over, token included: the donor must be able to copy it. */
  card?: DonationCard
  startedAt?: number
  until?: number
  receipts?: number
  active?: number
  error?: string
}

export interface DonationApi {
  donationList(): Promise<DonationSnapshot>
  donationImport(input: { text: string }): Promise<DonationSnapshot>
  donationRemove(input: { id: string }): Promise<DonationSnapshot>
  donationProbe(input: { id: string }): Promise<DonationProbeView>
  donationHostStart(input: DonationHostInput): Promise<DonationHostStatus>
  donationHostStop(): Promise<DonationHostStatus>
  donationHostStatus(): Promise<DonationHostStatus>
}
