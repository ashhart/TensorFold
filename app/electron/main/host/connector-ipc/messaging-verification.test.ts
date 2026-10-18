import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'

for (const brand of ['slack', 'discord']) {
  const handlers = new Map<string, Function>()
  const source = { id: `${brand}_test`, kind: 'api_token', token: 'old-fixture-token' }
  const store: any = { settings: { [brand]: { sources: [source] } } }
  let resolve!: (result: unknown) => void
  const pending = new Promise((done) => { resolve = done })
  const allowed = new Set(['../../../agent/slack-live', '../../../agent/slack-connector', '../../../agent/discord-connector', '../../../agent/connector-sync-state', '../../../agent/connector-spine', '../../../agent/security', '../../../agent/secret-rotation', '../../../store', '../persistence', '../audit', '../ipc', '../../state', '../connector-enrich', '../connector-event-spine', '../connector-runtime'])
  const imports: Record<string, unknown> = {
    '../../../agent/slack-live': { verifySlackToken: () => pending },
    '../../../agent/discord-connector': { verifyDiscordToken: () => pending },
    '../../state': { store },
    '../audit': { requireSecurityRole: () => {} },
  }
  const output = ts.transpileModule(readFileSync(path.join(__dirname, `${brand}.ts`), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
  const loaded = { exports: {} as any }
  new Function('require', 'module', 'exports', output)((name: string) => { assert.ok(allowed.has(name), `Reviewed dependency ${name}`); return imports[name] ?? {} }, loaded, loaded.exports)
  loaded.exports[brand === 'slack' ? 'registerSlackIpc' : 'registerDiscordIpc']({ handle: (name: string, action: Function) => handlers.set(name, action), spine: {} })
  const checking = handlers.get(`${brand}:verify`)!({}, { id: source.id })
  store.settings[brand].sources = [{ ...source, token: 'replacement-fixture-token' }]
  resolve({ ok: true, bot: 'Old token bot' })
  await assert.rejects(() => checking, /credentials changed/)
}
console.log('Slack/Discord host verification refuses late results after credential replacement')
