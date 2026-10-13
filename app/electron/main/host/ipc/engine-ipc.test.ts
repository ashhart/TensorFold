import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { engineRepo, engineRevision, engineSearchText } from './engine-ipc'
import { createMockEngineApi } from '../../../../src/lib/mock/engine'

// Renderer input is only ever a repo id, a revision or search text.
assert.equal(engineRepo('TensorFold/Qwen3.8-27B-MLX-4bit'), 'TensorFold/Qwen3.8-27B-MLX-4bit')
for (const bad of ['../../etc/passwd', '/abs/path', 'a/b/c', 'owner/..', '', null, 42, { repo: 'a/b' }]) {
  assert.throws(() => engineRepo(bad), /owner\/name/, String(bad))
}
assert.equal(engineRevision(undefined), null)
assert.equal(engineRevision(''), null)
assert.equal(engineRevision('93760be4f1f69842a46bc13dbdc0f19e291392a3'), '93760be4f1f69842a46bc13dbdc0f19e291392a3')
assert.throws(() => engineRevision('../main'), /branch, tag or commit/)
assert.throws(() => engineRevision(7), /branch, tag or commit/)
assert.equal(engineSearchText('  qwen  '), 'qwen')
assert.throws(() => engineSearchText(['qwen']), /must be a string/)
assert.throws(() => engineSearchText('x'.repeat(201)), /too long/)

// Every channel goes through handle() with a role check; delete needs the owner role.
const source = readFileSync(path.join(__dirname, 'engine-ipc.ts'), 'utf8')
const channels = [...source.matchAll(/handle\('(engine:[A-Za-z]+)'/g)].map((m) => m[1])
assert.deepEqual(channels, ['engine:snapshot', 'engine:connection', 'engine:refresh', 'engine:search', 'engine:preflight', 'engine:pull', 'engine:cancelPull', 'engine:load', 'engine:setCortheon', 'engine:unload', 'engine:delete'])
assert.match(source, /handle\('engine:setCortheon'[\s\S]*?typeof enabled !== 'boolean'/, 'Cortheon takes only a boolean')
assert.equal((source.match(/requireSecurityRole\(/g) ?? []).length, channels.length, 'each channel checks the caller role')
assert.match(source, /handle\('engine:delete'[\s\S]*?requireSecurityRole\('owner'/)
assert.doesNotMatch(source, /ipcMain\.handle/, 'no channel bypasses the trusted-sender gate')
const connectionHandler = source.slice(source.indexOf("handle('engine:connection'"), source.indexOf("handle('engine:refresh'"))
assert.match(connectionHandler, /requireSecurityRole\('operator'/, 'connection credentials require an operator')
assert.match(connectionHandler, /return svc\(\)\.connection\(\)/, 'connection IPC uses only the live service connection')
assert.doesNotMatch(connectionHandler, /auditEvent|emit|console\.|apiKey/, 'credentials are never copied to audits, events or logs')

const preload = readFileSync(path.join(__dirname, '..', '..', '..', 'preload.ts'), 'utf8')
for (const channel of channels) assert.ok(preload.includes(`'${channel}'`), `preload exposes ${channel}`)
assert.match(preload, /engineConnection: \(\): Promise<EngineConnection \| null> => ipcRenderer\.invoke\('engine:connection'\)/)

// Browser demos have no server or measured requests, even after their simulated model is ready.
const mock = createMockEngineApi()
assert.equal(await mock.engineConnection(), null)
assert.equal((await mock.engineSnapshot()).performance, null)
await mock.engineLoad('mlx-community/Qwen3.5-2B-MLX-4bit')
assert.equal(await mock.engineConnection(), null, 'the demo cannot supply fake working credentials')
assert.equal((await mock.engineSnapshot()).performance, null, 'the demo never invents throughput')
await mock.engineUnload()

console.log('engine ipc tests passed')
