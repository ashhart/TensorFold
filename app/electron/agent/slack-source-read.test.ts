import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
const reads: string[] = []
const imports: Record<string, unknown> = {
  'node:crypto': {}, 'node:fs': {}, 'node:path': {}, './summary-citations': {}, './connector-sync': {},
  './slack-live': {
    isLiveSlackSource: (source: any) => source.kind === 'api_token',
    readSlackThread: async (source: any, channel: string, ts: string) => { reads.push(source.id); return [{ ts, channelId: channel, sourceId: source.id }] },
  },
}
const output = ts.transpileModule(readFileSync(path.join(__dirname, 'slack-connector.ts'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
const loaded = { exports: {} as any }
new Function('require', 'module', 'exports', output)((name: string) => { assert.ok(name in imports); return imports[name] }, loaded, loaded.exports)
const settings = { sources: [{ id: 'slack_a', kind: 'api_token', enabled: true }, { id: 'slack_b', kind: 'api_token', enabled: true }], maxMessages: 50, lastSyncAt: null }
const found = await loaded.exports.readSlackMessage(settings, 'slack:slack_b:slack_b:C123:1.0')
assert.equal(found.sourceId, 'slack_b')
assert.deepEqual(reads, ['slack_b'], 'source-bound citations use exactly the matching account token')
await assert.rejects(() => loaded.exports.readSlackMessage(settings, 'slack:missing:C123:1.0'), /unavailable/)
await assert.rejects(() => loaded.exports.readSlackMessage(settings, 'C123:1.0'), /source-bound/)
await assert.rejects(() => loaded.exports.readSlackMessage({ ...settings, sources: settings.sources.map((source) => ({ ...source, enabled: source.id !== 'slack_b' })) }, 'slack:slack_b:C123:1.0'), /disconnected/)
assert.deepEqual(reads, ['slack_b'], 'missing, disconnected and ambiguous inputs never probe another account')
console.log('Slack source-bound read citations, disconnected/unknown refusal and ambiguous raw multi-account read regressions passed')
