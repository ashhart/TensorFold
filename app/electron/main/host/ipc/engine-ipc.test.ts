import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { engineRepo, engineRevision, engineSearchText, engineLibraryFolderId, engineLibraryScanOptions } from './engine-ipc'
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
assert.equal(engineLibraryFolderId('a'.repeat(64)), 'a'.repeat(64))
for (const bad of ['/tmp/model', '../escape', null, { path: '/tmp/model' }]) assert.throws(() => engineLibraryFolderId(bad), /saved model folder/)
assert.deepEqual(engineLibraryScanOptions(undefined), {})
assert.deepEqual(engineLibraryScanOptions({ includeHuggingFace: false }), { includeHuggingFace: false })
for (const bad of [{ path: '/tmp/model' }, { includeHuggingFace: 'yes' }, [], '/tmp/model']) assert.throws(() => engineLibraryScanOptions(bad))

// Every channel goes through handle() with a role check; delete needs the owner role.
const source = readFileSync(path.join(__dirname, 'engine-ipc.ts'), 'utf8')
const channels = [...source.matchAll(/handle\('(engine:[A-Za-z]+)'/g)].map((m) => m[1])
assert.deepEqual(channels, ['engine:snapshot', 'engine:connection', 'engine:refresh', 'engine:search', 'engine:preflight', 'engine:pull', 'engine:cancelPull', 'engine:load', 'engine:compareModels', 'engine:comparisonStatus', 'engine:cancelComparison', 'engine:tryModel', 'engine:cancelTryModel', 'engine:setCortheon', 'engine:learning', 'engine:setLearning', 'engine:envOverrides', 'engine:setEnvOverrides', 'engine:teach', 'engine:cancelTeach', 'engine:unload', 'engine:delete', 'engine:library', 'engine:scanLibrary', 'engine:addLibraryFolder', 'engine:removeLibraryFolder', 'engine:cancelLibraryScan'])
assert.match(source, /handle\('engine:setCortheon'[\s\S]*?typeof enabled !== 'boolean'/, 'Cortheon takes only a boolean')
assert.equal((source.match(/requireSecurityRole\(/g) ?? []).length, channels.length, 'each channel checks the caller role')
assert.match(source, /handle\('engine:delete'[\s\S]*?requireSecurityRole\('owner'/)
assert.doesNotMatch(source, /ipcMain\.handle/, 'no channel bypasses the trusted-sender gate')
const connectionHandler = source.slice(source.indexOf("handle('engine:connection'"), source.indexOf("handle('engine:refresh'"))
assert.match(connectionHandler, /requireSecurityRole\('operator'/, 'connection credentials require an operator')
assert.match(connectionHandler, /return svc\(\)\.connection\(\)/, 'connection IPC uses only the live service connection')
assert.doesNotMatch(connectionHandler, /auditEvent|emit|console\.|apiKey/, 'credentials are never copied to audits, events or logs')
const teachHandler = source.slice(source.indexOf("handle('engine:teach'"), source.indexOf("handle('engine:cancelTeach'"))
assert.match(teachHandler, /Buffer.byteLength\(text, 'utf8'\) > 8192/, 'lessons have a main-process byte limit')
assert.doesNotMatch(teachHandler, /auditEvent\([^\n]*text/, 'private teaching text never enters security audit logs')
assert.match(source, /handle\('engine:setLearning'[\s\S]*?typeof input\?\.enabled !== 'boolean'/, 'the renderer cannot supply flags, credentials or paths')
const envHandler = source.slice(source.indexOf("handle('engine:setEnvOverrides'"), source.indexOf("handle('engine:teach'"))
assert.match(envHandler, /normalizeEngineEnvOverrides\(value\)/, 'advanced variables are re-validated in the main process')
assert.match(envHandler, /Object\.keys\(overrides\)\.sort\(\)/, 'only variable names enter the audit, never the values')
const folderHandler = source.slice(source.indexOf("handle('engine:addLibraryFolder'"), source.indexOf("handle('engine:removeLibraryFolder'"))
assert.match(folderHandler, /dialog\.showOpenDialog/, 'only the native picker admits filesystem paths')
assert.match(folderHandler, /svc\(\)\.addLibraryFolder\(result\.filePaths\[0\]\)/)
assert.doesNotMatch(folderHandler, /auditEvent|console\./, 'private folder paths do not enter audit or process logs')

const preload = readFileSync(path.join(__dirname, '..', '..', '..', 'preload.ts'), 'utf8')
for (const channel of channels) assert.ok(preload.includes(`'${channel}'`), `preload exposes ${channel}`)
assert.match(preload, /engineConnection: \(\): Promise<EngineConnection \| null> => ipcRenderer\.invoke\('engine:connection'\)/)

// Browser demos have no server or measured requests, even after their simulated model is ready.
const mock = createMockEngineApi()
const beforeEnable = await mock.engineSnapshot()
await assert.rejects(mock.engineSetCortheon(true), /temporarily disabled in this beta/)
assert.deepEqual(await mock.engineSnapshot(), beforeEnable, 'the browser mock also refuses stale enable calls without mutation')
assert.match(beforeEnable.status.cortheon?.disabledReason ?? '', /temporarily disabled in this beta/)
assert.equal(await mock.engineConnection(), null)
assert.equal((await mock.engineSnapshot()).performance, null)
await mock.engineLoad('mlx-community/Qwen3.5-2B-MLX-4bit')
assert.equal(await mock.engineConnection(), null, 'the demo cannot supply fake working credentials')
assert.equal((await mock.engineSnapshot()).performance, null, 'the demo never invents throughput')
await mock.engineUnload()

console.log('engine ipc tests passed')
