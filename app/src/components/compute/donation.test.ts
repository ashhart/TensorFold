import assert from 'node:assert/strict'
import { donationBaseUrl, donationPhase, donationProvider, donationProviderId, donationRemainingMs, makeDonationCard, parseDonationCard, DONATION_SCHEMA } from '../../../shared/donation'

const now = Date.UTC(2026, 9, 23, 12, 0, 0)
const base = {
  schema: DONATION_SCHEMA,
  id: 'spark-a1',
  label: 'Spark 27B',
  endpoint: 'http://192.168.0.45:8010/',
  token: 'token-abcdefghijklmnop',
  model: 'qwen3.6',
  window: { from: now - 60_000, until: now + 3_600_000 },
  caps: { maxConcurrent: 2, maxTokensPerRequest: 1_024, maxRequestBytes: 1_048_576 },
  security: 'poc-insecure-lan' as const,
  createdAt: now - 120_000,
}

const parsed = parseDonationCard(base, now)
assert.ok('card' in parsed, 'A complete card parses')
if ('card' in parsed) {
  assert.equal(parsed.card.endpoint, 'http://192.168.0.45:8010', 'The endpoint is normalized to an origin')
  assert.equal(parsed.card.schema, DONATION_SCHEMA)
  assert.equal(donationBaseUrl(parsed.card), 'http://192.168.0.45:8010/v1')
  assert.equal(donationProviderId(parsed.card), 'donation:spark-a1')
  assert.equal(donationPhase(parsed.card, now), 'open')
  assert.equal(donationPhase(parsed.card, now - 120_000), 'before')
  assert.equal(donationPhase(parsed.card, now + 4_000_000), 'after')
  assert.equal(donationRemainingMs(parsed.card, now), 3_600_000)
  assert.equal(donationRemainingMs(parsed.card, now - 120_000), 3_660_000, 'Before the window, the whole window remains')
  const provider = donationProvider(parsed.card)
  assert.deepEqual(provider, {
    id: 'donation:spark-a1', name: 'Spark 27B', transport: 'chat-completions', auth: 'api-key',
    baseUrl: 'http://192.168.0.45:8010/v1', apiKey: 'token-abcdefghijklmnop', models: ['qwen3.6'],
    kind: 'cloud', remoteCompute: true, source: 'custom',
  })
}

const fromText = parseDonationCard(JSON.stringify(base), now)
assert.ok('card' in fromText, 'A card parses from JSON text')

const cases: Array<[string, unknown, RegExp]> = [
  ['unknown field', { ...base, price: 1 }, /unknown fields: price/],
  ['wrong schema', { ...base, schema: 'nope' }, /schema must be/],
  ['credentials in endpoint', { ...base, endpoint: 'http://user:pass@example.test:8010' }, /must not carry credentials/],
  ['not-a-card-id path in endpoint', { ...base, endpoint: 'http://example.test:8010/v1' }, /at most one path segment/],
  ['deep path in endpoint', { ...base, endpoint: 'http://example.test:8010/donate-abc123/v1' }, /at most one path segment/],
  ['query in endpoint', { ...base, endpoint: 'http://example.test:8010/?x=1' }, /query or a fragment/],
  ['tls card on http', { ...base, security: 'tls' }, /must use an https endpoint/],
  ['plain card on https', { ...base, endpoint: 'https://example.test:8010', security: 'poc-insecure-lan' }, /must use a plain http endpoint/],
  ['window reversed', { ...base, window: { from: now, until: now - 1 } }, /ends before it starts/],
  ['window too long', { ...base, window: { from: now, until: now + 367 * 24 * 3600_000 } }, /longer than a year/],
  ['cap zero', { ...base, caps: { ...base.caps, maxConcurrent: 0 } }, /maxConcurrent/],
  ['cap too high', { ...base, caps: { ...base.caps, maxRequestBytes: 64 * 1024 * 1024 } }, /maxRequestBytes/],
  ['unknown cap', { ...base, caps: { ...base.caps, burst: 3 } }, /unknown fields: burst/],
  ['short token', { ...base, token: 'short' }, /token must be/],
  ['bad id', { ...base, id: 'A B' }, /card id must be/],
  ['missing label', { ...base, label: '  ' }, /label must be/],
  ['bad createdAt', { ...base, createdAt: -1 }, /createdAt/],
  ['not json', '{oops', /not valid JSON/],
  ['not an object', 42, /must be a JSON object/],
]
const relayCard = parseDonationCard({ ...base, endpoint: 'https://relay.example/donate-abc123', security: 'tls' }, now)
assert.ok('card' in relayCard, 'A relay endpoint that carries the card id parses')
if ('card' in relayCard) {
  assert.equal(relayCard.card.endpoint, 'https://relay.example/donate-abc123', 'The relay path survives parsing')
  assert.equal(donationBaseUrl(relayCard.card), 'https://relay.example/donate-abc123/v1', 'The provider base URL keeps the relay path')
  assert.equal(donationProvider(relayCard.card).baseUrl, 'https://relay.example/donate-abc123/v1')
}

for (const [name, input, pattern] of cases) {
  const result = parseDonationCard(input, now)
  assert.ok('error' in result, `${name} is refused`)
  if ('error' in result) assert.match(result.error, pattern, name)
}

const made = makeDonationCard({
  id: 'made-card', label: 'Made', endpoint: 'http://127.0.0.1:8010', token: 'token-abcdefghijklmnop', model: 'm',
  window: { from: now, until: now + 60_000 }, caps: {}, security: 'poc-insecure-lan', createdAt: now,
})
assert.ok('card' in made, 'makeDonationCard fills defaults')
if ('card' in made) assert.deepEqual(made.card.caps, { maxConcurrent: 1, maxTokensPerRequest: 1_024, maxRequestBytes: 1_048_576 })
console.log('donation contract: ok')
