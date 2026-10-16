import assert from 'node:assert/strict'
import { credentialTextStream, credentialChunkStream, scrubProviderCredential, scrubProviderFailure, scrubProviderHeaders, scrubProviderErrorBody, scrubProviderResearchMetadata } from './provider-credential-privacy'
import { providerErrorText } from './openai'
const key = 'FICTIONAL_ONLY_key+"quote"/back\\slash%=5837'
const forms = [key, JSON.stringify(key).slice(1, -1), JSON.stringify(key).slice(1, -1).replace(/\//g, '\\/'), encodeURIComponent(key), encodeURIComponent(key).replace(/%[0-9A-F]{2}/g, (part) => part.toLowerCase())]
for (const form of forms) for (let at = 1; at < form.length; at++) {
  const stream = credentialTextStream(key)
  const actual = stream.push(`Before ${form.slice(0, at)}`) + stream.push(`${form.slice(at)} after.`) + stream.finish()
  assert.equal(actual, 'Before [redacted credential] after.', 'every credential split and common escaping must preserve surrounding text only')
  const tagged = credentialChunkStream(key)
  const chunks = [...tagged.push('content', `Before ${form.slice(0, at)}`), ...tagged.push('reasoning', `${form.slice(at)} after.`), ...tagged.finish()]
  assert.equal(chunks.map((chunk) => chunk.text).join(''), 'Before [redacted credential] after.', 'a credential cannot be reconstructed across content/reasoning channels')
  assert.equal(chunks[0].kind, 'content'); assert.equal(chunks.at(-1)!.kind, 'reasoning')
}
const normal = credentialTextStream(key)
assert.equal(normal.push('Ordinary F') + normal.push('ollow-up text with a final FI') + normal.finish(), 'Ordinary Follow-up text with a final FI', 'a non-secret credential-prefix suffix is flushed without lost normal text')
assert.equal(scrubProviderCredential('Ordinary model diagnosis.', key), 'Ordinary model diagnosis.')
assert.ok(!providerErrorText('x'.repeat(390) + key, key).includes('FICTIONAL'), 'scrubbing precedes the diagnostic character bound so truncation cannot leak a credential prefix')
const normalTagged = credentialChunkStream(key)
const normalChunks = [...normalTagged.push('content', 'Ordinary F'), ...normalTagged.push('reasoning', 'ollow up.'), ...normalTagged.finish()]
assert.equal(normalChunks.filter((chunk) => chunk.kind === 'content').map((chunk) => chunk.text).join(''), 'Ordinary F')
assert.equal(normalChunks.filter((chunk) => chunk.kind === 'reasoning').map((chunk) => chunk.text).join(''), 'ollow up.')
assert.equal(scrubProviderCredential('Ordinary text with spaces.', ' '), 'Ordinary text with spaces.')
assert.equal(scrubProviderCredential(`Echo ${key}`, `  ${key}  `), 'Echo [redacted credential]')
const escapedBody = JSON.stringify({error:{message:key},available_models:['alpha','beta']}).replace('FICTIONAL','\\u0046ICTIONAL')
const parsedBody = JSON.parse(scrubProviderErrorBody(escapedBody, key))
assert.equal(parsedBody.error.message, '[redacted credential]')
assert.deepEqual(parsedBody.available_models,['alpha','beta'])
const headers = new Headers({ 'retry-after': '17', 'x-ratelimit-reset': '2000000000', 'x-fixture-echo': key, [`x-fixture-${encodeURIComponent(key)}`]: 'echoed-name' })
const cleanHeaders = scrubProviderHeaders(headers, key)
assert.equal(cleanHeaders.get('retry-after'), '17')
assert.equal(cleanHeaders.get('x-ratelimit-reset'), '2000000000')
assert.equal(cleanHeaders.get('x-fixture-echo'), '[redacted credential]')
assert.ok(![...cleanHeaders.keys()].some((name) => name.includes(encodeURIComponent(key).toLowerCase())))
assert.equal(headers.get('x-fixture-echo'), key, 'sanitizing a failure does not mutate the response object')
const source = Object.assign(new Error(`Synthetic ${key}`), { status: 429, headers, detail: { nested: key }, cause: Object.assign(new Error(`Nested ${key}`), { code: 'ECONNRESET' }) })
const clean = scrubProviderFailure(source, key) as typeof source
assert.equal(clean.status, 429)
assert.equal(clean.cause.code, 'ECONNRESET')
assert.equal(clean.headers.get('retry-after'), '17')
assert.ok(!JSON.stringify({ message: clean.message, stack: clean.stack, detail: clean.detail, cause: clean.cause.message }).includes(key))
assert.equal(source.detail.nested, key, 'the original failure remains untouched')
console.log('Credential privacy helper: every split/escaping, normal-prefix flush, pre-truncation scrub, echoed header names/values and structured retry/cause metadata pass.')

const urlKey = `${key}~edge`
const contaminated = new URL('https://example.invalid/research'); contaminated.searchParams.set('token', urlKey)
const report = scrubProviderResearchMetadata({ task: urlKey, searches: 2, pages_read: 3, sources: [contaminated.href, 'https://example.invalid/ordinary?q=green'] }, urlKey) as any
assert.deepEqual(report.sources, ['https://example.invalid/ordinary?q=green'])
assert.equal(report.searches, 2); assert.equal(report.pages_read, 3); assert.equal(report.task, '[redacted credential]')
const percentLetter = new URL('https://example.invalid/research?token=' + encodeURIComponent(urlKey).replace('FICTIONAL', '%46ICTIONAL'))
assert.equal((scrubProviderResearchMetadata({ sources: [percentLetter.href] }, urlKey) as any).sources.length, 0, 'standard URL component decoding compares the exact credential rather than only one generated encoding')

const hostKey='fictional-host-token-5837'
const hostSource=`https://%66${hostKey.slice(1)}.example.invalid/path`
assert.ok(new URL(hostSource).hostname.includes(hostKey))
assert.deepEqual((scrubProviderResearchMetadata({sources:[hostSource,'https://example.invalid/ordinary']},hostKey) as any).sources,['https://example.invalid/ordinary'])
const uppercaseHostKey = hostKey.toUpperCase()
const uppercaseHostSource = `https://%46${uppercaseHostKey.slice(1)}.example.invalid/path`
assert.ok(decodeURIComponent(uppercaseHostSource).includes(uppercaseHostKey))
assert.deepEqual((scrubProviderResearchMetadata({ sources: [uppercaseHostSource] }, uppercaseHostKey) as any).sources, [], 'raw URL decoding must preserve credential case before hostname canonicalization')
