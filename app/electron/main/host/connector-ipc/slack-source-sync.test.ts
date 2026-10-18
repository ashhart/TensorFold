import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
import { mergeConnectorSyncSources } from '../../../agent/connector-sync-state'

const handlers = new Map<string, Function>()
const scanned: string[][] = []
const store: any = { settings: { slack: { sources: [{ id: 'workspace_a', label: 'A', enabled: true, token: 'example-token-a' }, { id: 'workspace_b', label: 'B', enabled: true, token: 'example-token-b' }], maxMessages: 50, lastSyncAt: null } }, connectorEvents: [], contacts: [] }
const imports: Record<string, unknown> = {
  '../../../agent/slack-live': {},
  '../../../agent/slack-connector': { syncSlackConnector: async (settings: any) => { scanned.push(settings.sources.map((source: any) => source.id)); return { syncedAt: 100, messages: 0, events: [], sources: settings.sources.map((source: any) => ({ ...source, lastSyncAt: 100 })) } } },
  '../../../agent/connector-sync-state': { mergeConnectorSyncSources },
  '../../../agent/connector-spine': {},
  '../../../agent/security': {},
  '../../../agent/secret-rotation': {},
  '../../../store': { saveSettings: async () => {} },
  '../persistence': { persistContacts: () => {} },
  '../audit': { requireSecurityRole: () => {} },
  '../ipc': { handle: () => { throw new Error('Test must inject handle') } },
  '../../state': { store },
  '../connector-enrich': { enrichContacts: (_contacts: unknown, events: unknown) => ({ directory: [], events }), slackContactObservations: () => [], attachAuthedMediaPaths: async () => {} },
  '../connector-event-spine': { publicConnectorSettings: () => store.settings },
}
const source = readFileSync(path.join(__dirname, 'slack.ts'), 'utf8')
const output = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
const loaded = { exports: {} as any }
new Function('require', 'module', 'exports', output)((id: string) => { assert.ok(id in imports, `Explicit reviewed import ${id}`); return imports[id] }, loaded, loaded.exports)
loaded.exports.registerSlackIpc({ handle: (name: string, action: Function) => handlers.set(name, action), syncQueue: { run: (_key: string, work: () => Promise<unknown>) => work() }, spine: { rememberConnectorEvents: () => {} } })
const sync = handlers.get('slack:sync')!
const first = await sync({}, { id: 'workspace_a' })
assert.deepEqual(scanned, [['workspace_a']], 'a per-source read never scans another workspace')
assert.deepEqual(first.sources.map((item: any) => item.id), ['workspace_a'], 'source feedback is scoped to the requested account')
assert.equal(store.settings.slack.sources.length, 2, 'scoped sync never removes unrelated accounts')
assert.equal(store.settings.slack.sources[1].lastSyncAt, undefined, 'unrelated account status is untouched')
assert.equal(store.settings.slack.sources[1].token, 'example-token-b')
await assert.rejects(() => sync({}, { id: 'missing' }), /not found/)
await assert.rejects(() => sync({}, { id: 1 }), /not found/)
await assert.rejects(() => sync({}, 'workspace_a'), /input/)
assert.equal(scanned.length, 1, 'invalid source requests never fall back to a global sync')
await sync({})
assert.deepEqual(scanned[1], ['workspace_a', 'workspace_b'], 'the explicit existing global sync remains compatible')
console.log('Slack IPC per-source read, source/status preservation, malformed-scope refusal and legacy global sync regressions passed')
