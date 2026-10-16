// Run: jiti electron/agent/cli-update-feeds.test.ts
import assert from 'node:assert/strict'
import { CLI_UPDATE_FEEDS, npmLatestUrl, latestNpmVersion, fetchCliLatestVersions, type FetchLike } from './cli-update-feeds'
import { builtinClis } from './cli-registry'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('cli-update-feeds tests:')

{
  const builtinIds = new Set(builtinClis().map((cli) => cli.id))
  for (const id of Object.keys(CLI_UPDATE_FEEDS)) {
    assert.ok(builtinIds.has(id), `feed key '${id}' must be a builtin launcher id`)
  }
  assert.ok(Object.values(CLI_UPDATE_FEEDS).every((feed) => feed.npmPackage && feed.updateCommand.includes(feed.npmPackage)), 'every feed has a package and a matching update command')
  ok('CLI_UPDATE_FEEDS: keys are builtin ids, commands reference the package')
}

{
  assert.equal(npmLatestUrl('@openai/codex'), 'https://registry.npmjs.org/@openai%2Fcodex/latest', 'scoped packages encode the slash, keep the @')
  assert.equal(npmLatestUrl('opencode-ai'), 'https://registry.npmjs.org/opencode-ai/latest')
  ok('npmLatestUrl: scoped + unscoped shapes')
}

const respond = (status: number, body: any): ReturnType<FetchLike> => Promise.resolve({ ok: status >= 200 && status < 300, status, json: async () => body })

{
  const fetchImpl: FetchLike = async (url) => {
    assert.ok(url.includes('registry.npmjs.org'))
    return respond(200, { name: '@openai/codex', version: '0.150.0' })
  }
  assert.equal(await latestNpmVersion('@openai/codex', fetchImpl), '0.150.0')
  ok('latestNpmVersion: happy path returns the version')
}

{
  assert.equal(await latestNpmVersion('gone', async () => respond(404, { error: 'Not found' })), null, '404 → null')
  assert.equal(await latestNpmVersion('boom', async () => { throw new Error('network down') }), null, 'network failure → null, never throws')
  assert.equal(await latestNpmVersion('weird', async () => respond(200, { version: 42 })), null, 'non-string version → null')
  ok('latestNpmVersion: 404 / network failure / bad shape all yield null')
}

{
  const wrongMime = new Response(JSON.stringify({ version: '9.9.9' }), { headers: { 'content-type': 'text/html' } })
  assert.equal(await latestNpmVersion('wrong-mime', async () => wrongMime), null, 'wrong MIME is rejected')

  const declaredHuge = new Response(JSON.stringify({ version: '9.9.9' }), {
    headers: { 'content-type': 'application/json', 'content-length': String(65 * 1024) },
  })
  assert.equal(await latestNpmVersion('declared-huge', async () => declaredHuge), null, 'declared oversized metadata is rejected')

  const chunkedHuge = new Response('x'.repeat(65 * 1024), { headers: { 'content-type': 'application/json' } })
  assert.equal(await latestNpmVersion('chunked-huge', async () => chunkedHuge), null, 'chunked oversized metadata is rejected')
  ok('latestNpmVersion: MIME and declared/chunked response caps fail closed')
}

{
  const versions: Record<string, string> = { '@openai/codex': '0.150.0', 'opencode-ai': '1.15.0' }
  const fetchImpl: FetchLike = async (url) => {
    const pkg = decodeURIComponent(url.replace('https://registry.npmjs.org/', '').replace('/latest', ''))
    if (pkg === '@anthropic-ai/claude-code') throw new Error('registry hiccup')
    if (pkg === '@google/gemini-cli') return respond(500, {})
    return respond(200, { version: versions[pkg] })
  }
  const latest = await fetchCliLatestVersions(fetchImpl)
  assert.deepEqual(Object.keys(latest).sort(), ['codex', 'opencode'], 'failed feeds are dropped, not fatal')
  assert.equal(latest.codex.latestVersion, '0.150.0')
  assert.equal(latest.codex.updateCommand, 'npm install -g @openai/codex')
  assert.equal(latest.opencode.latestVersion, '1.15.0')
  ok('fetchCliLatestVersions: per-feed isolation, shaped results')
}

{
  const latest = await fetchCliLatestVersions(async () => respond(200, { version: '9.9.9' }), {
    feeds: { codex: { npmPackage: 'x', updateCommand: 'npm install -g x' } },
  })
  assert.deepEqual(Object.keys(latest), ['codex'], 'custom feed table is honoured')
  ok('fetchCliLatestVersions: custom feeds override the default table')
}

console.log(`\n${passed} passed`)
