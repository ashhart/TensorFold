// Run: jiti electron/agent/slack-live.test.ts
import assert from 'node:assert/strict'
import { extractSlackFileRefs, editSlackMessage, isLiveSlackSource, postSlackMessage, validateLiveSlackSource } from './slack-live.ts'
import { parseLiveSlackId } from './slack-connector.ts'
import type { SlackSourceConfig } from './types.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

function makeSource(partial: Partial<SlackSourceConfig>): SlackSourceConfig {
  return {
    id: 'slack_1', label: 'Work', path: '', kind: 'api_token', enabled: true,
    createdAt: 0, updatedAt: 0, lastSyncAt: null,
    token: 'xoxb-test-token-12345', channels: '',
    ...partial,
  }
}

console.log('slack-live tests:')

{
  assert.equal(isLiveSlackSource(makeSource({ kind: 'api_token' })), true)
  assert.equal(isLiveSlackSource(makeSource({ kind: 'export_dir' })), false)
  ok('isLiveSlackSource distinguishes live from export sources')
}

{
  // A valid token + xoxb prefix passes.
  assert.equal(validateLiveSlackSource(makeSource({})), null)
  // Missing token is rejected before any network call.
  assert.match(validateLiveSlackSource(makeSource({ token: '' }))!, /token is required/i)
  // A non-Slack-looking token is rejected.
  assert.match(validateLiveSlackSource(makeSource({ token: 'abc123' }))!, /not look like/i)
  ok('validateLiveSlackSource requires a Slack-shaped bot token')
}

{
  // Live message ids ("channel:ts" or "channel/ts") round-trip through parseLiveSlackId.
  const a = parseLiveSlackId('C12345:1700000000.000100')
  assert.equal(a.channel, 'C12345')
  assert.equal(a.ts, '1700000000.000100')
  const b = parseLiveSlackId('C12345/1700000000.000100')
  assert.equal(b.channel, 'C12345')
  assert.equal(b.ts, '1700000000.000100')
  // A malformed id yields empty fields (caller treats it as "not found").
  const c = parseLiveSlackId('garbage')
  assert.equal(c.channel, '')
  assert.equal(c.ts, '')
  ok('parseLiveSlackId parses channel:ts and channel/ts ids')
}

{
  // The id the AGENT actually receives is slackSourceId(message) =
  // "slack:<sourceId>:<id>" where a live message's id is "<sourceId>:<channel>:<ts>".
  // The old parser took everything before the last ":" as the channel, yielding a bogus
  // channel like "slack:slack_1:slack_1:C12345"; it must extract just "C12345".
  const live = parseLiveSlackId('slack:slack_1:slack_1:C12345:1700000000.000100')
  assert.equal(live.channel, 'C12345')
  assert.equal(live.ts, '1700000000.000100')

  // Same, with the slash delimiter variant in the path-style id.
  const liveSlash = parseLiveSlackId('slack:slack_1:slack_1/C12345/1700000000.000100')
  assert.equal(liveSlash.channel, 'C12345')
  assert.equal(liveSlash.ts, '1700000000.000100')

  // A numeric-looking source token before the channel must not be mistaken for the ts.
  const numericSource = parseLiveSlackId('slack:12345678:12345678:G98765:1700000000.000200')
  assert.equal(numericSource.channel, 'G98765')
  assert.equal(numericSource.ts, '1700000000.000200')
  ok('parseLiveSlackId extracts channel/ts from prefixed live source ids')
}

// ---- Slack file-ref extraction (pure) -----------------------------------------

{
  // A message with a real file: url_private is captured along with name + mimetype. The example
  // host is example.com (never a real Slack host) so the leak scanner stays happy.
  const refs = extractSlackFileRefs({ files: [{ id: 'F1', name: 'design.pdf', mimetype: 'application/pdf', url_private: 'https://files.example.com/F1/design.pdf' }] })
  assert.deepEqual(refs, [{ urlPrivate: 'https://files.example.com/F1/design.pdf', name: 'design.pdf', mimetype: 'application/pdf' }])
  ok('extractSlackFileRefs: captures url_private + name + mimetype')
}

{
  // url_private_download (the force-download variant) wins over url_private when both are present.
  const refs = extractSlackFileRefs({ files: [{ name: 'a.png', mimetype: 'image/png', url_private: 'https://files.example.com/a-inline.png', url_private_download: 'https://files.example.com/a-dl.png' }] })
  assert.equal(refs[0].urlPrivate, 'https://files.example.com/a-dl.png')
  ok('extractSlackFileRefs: prefers url_private_download over url_private')
}

{
  // Tombstoned/hidden files and files without a url_private are skipped; duplicates are de-duped.
  const refs = extractSlackFileRefs({ files: [
    { id: 'F1', mode: 'tombstone', url_private: 'https://files.example.com/F1' },
    { id: 'F2', name: 'no-url' },
    { id: 'F3', name: 'dupe.png', mimetype: 'image/png', url_private: 'https://files.example.com/dupe.png' },
    { id: 'F4', name: 'dupe-again.png', mimetype: 'image/png', url_private: 'https://files.example.com/dupe.png' },
  ] })
  assert.deepEqual(refs.map((r) => r.urlPrivate), ['https://files.example.com/dupe.png'], 'tombstone + url-less skipped, dupes collapsed')
  // No files array / non-message shape => no refs (and never throws).
  assert.deepEqual(extractSlackFileRefs({}), [])
  assert.deepEqual(extractSlackFileRefs(null), [])
  assert.deepEqual(extractSlackFileRefs({ files: 'nope' as unknown as [] }), [])
  ok('extractSlackFileRefs: skips tombstones/url-less, de-dupes, tolerates missing files')
}

// ---- write-path allowlist (postSlackMessage / editSlackMessage) -----------------------

// a tiny fetch stub (save/restore) returning canned Slack JSON by path, tracking calls made.
const realFetch = globalThis.fetch
const TEST_HTTP_DEPS = {
  lookup: async () => [{ address: '8.8.8.8', family: 4 }],
  fetchImpl: ((input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => globalThis.fetch(input, init)) as typeof fetch,
}
function stubFetch(routes: Record<string, unknown>) {
  const calls: string[] = []
  globalThis.fetch = (async (input: any) => {
    const url = typeof input === 'string' ? input : input?.url ?? String(input)
    const path = url.replace(/^https?:\/\/[^/]+/, '').split('?')[0]! // strip origin + query
    calls.push(path)
    for (const [needle, body] of Object.entries(routes)) {
      if (path.startsWith(needle)) return { ok: true, status: 200, json: async () => body } as any
    }
    return { ok: true, status: 200, json: async () => ({ ok: false, error: `no stub for ${path}` }) } as any
  }) as typeof fetch
  return calls
}
const restoreFetch = () => { globalThis.fetch = realFetch }

await (async () => {
  // The allowlist restricts this source to ONE channel (C_ALLOWED). A post targeting a
  // DIFFERENT channel (C_DENIED) must be rejected BEFORE any chat.postMessage call is made,
  // even though conversations.list shows the bot can technically see both channels.
  const source = makeSource({ channels: 'C_ALLOWED' })
  const calls = stubFetch({
    '/api/conversations.list': { ok: true, channels: [{ id: 'C_ALLOWED', name: 'eng', is_private: false }, { id: 'C_DENIED', name: 'secret', is_private: false }] },
    '/api/chat.postMessage': { ok: true, ts: '1700000000.000100' },
    '/api/chat.update': { ok: true, ts: '1700000000.000100' },
  })
  try {
    await assert.rejects(
      () => postSlackMessage(source, 'C_DENIED', 'should not send', undefined, undefined, TEST_HTTP_DEPS),
      /not.*allowed channels/i,
      'post to denied channel rejected',
    )
    assert.ok(!calls.includes('/api/chat.postMessage'), 'chat.postMessage must never be called for a denied channel')

    await assert.rejects(
      () => editSlackMessage(source, 'C_DENIED', '1700000000.000100', 'should not edit', TEST_HTTP_DEPS),
      /not.*allowed channels/i,
      'edit on denied channel rejected',
    )
    assert.ok(!calls.includes('/api/chat.update'), 'chat.update must never be called for a denied channel')

    // Sending to the allowed channel still works.
    const sent = await postSlackMessage(source, 'C_ALLOWED', 'shipping it', undefined, undefined, TEST_HTTP_DEPS)
    assert.equal(sent.ok, true)
    assert.equal(sent.ts, '1700000000.000100')
    const edited = await editSlackMessage(source, 'C_ALLOWED', '1700000000.000100', 'shipped', TEST_HTTP_DEPS)
    assert.equal(edited.ok, true)
    ok('postSlackMessage/editSlackMessage enforce the write allowlist (fail closed) and still send when allowed')
  } finally {
    restoreFetch()
  }
})()

console.log(`\n${passed} checks passed.`)
