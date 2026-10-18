import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { createJiti } from 'jiti'
import ts from 'typescript'

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'google-host-storage-'))
const previousHome = process.env.VONTRA_HOME
const previousKey = process.env.VONTRA_SECRET_KEY_BASE64
process.env.VONTRA_HOME = root
delete process.env.VONTRA_SECRET_KEY_BASE64
const jiti = createJiti(__filename, { moduleCache: false, fsCache: false })
const serviceModule = await jiti.import('../../agent/google-workspace')
const storage = await jiti.import<typeof import('../../store/write-infra')>('../../store/write-infra')
const paths = await jiti.import('../../store/paths')
const source = await fs.readFile(new URL('./google-workspace-runtime.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
const clients: Array<{ dispose(): void }> = []
const shutdown: Array<() => void> = []
function runtime() {
  const loaded = { exports: {} as any }
  const imports: Record<string, any> = {
    electron: { app: { once: (_event: string, callback: () => void) => shutdown.push(callback) } },
    'node:fs': { promises: fs }, '../../agent/google-workspace': serviceModule,
    '../../store/write-infra': storage, '../../store/paths': paths,
    './external': { openExternalSafely: async () => { assert.fail('Storage qualification never opens an external browser') } },
  }
  new Function('require', 'module', 'exports', compiled)((name: string) => { assert.ok(name in imports, `Reviewed runtime import ${name}`); return imports[name] }, loaded, loaded.exports)
  const instance = loaded.exports.googleWorkspaceService()
  assert.equal(loaded.exports.googleWorkspaceService(), instance, 'one protected service per host bootstrap')
  clients.push(instance)
  return instance
}
const file = path.join(root, 'state', 'google-workspace.protected.json')
const syntheticClient = { installed: { client_id: '123-synthetic.apps.googleusercontent.com', client_secret: 'synthetic-local-config', auth_uri: 'https://accounts.google.com/o/oauth2/auth', token_uri: 'https://oauth2.googleapis.com/token', redirect_uris: ['http://localhost'] } }
try {
  assert.equal((await runtime().snapshot()).clientConfigured, false)
  await runtime().configureClient(JSON.stringify(syntheticClient))
  const protectedBytes = await fs.readFile(file, 'utf8')
  assert.doesNotMatch(protectedBytes, /synthetic-local-config|synthetic\.apps|clientSecret/, 'client credentials never appear in cleartext on disk')
  assert.match(JSON.parse(protectedBytes).encrypted, /^safe-json:v1:/)
  assert.equal((await fs.stat(file)).mode & 0o777, 0o600)
  assert.equal((await fs.stat(path.join(root, '.secretkey'))).mode & 0o777, 0o600)
  assert.equal((await runtime().snapshot()).clientConfigured, true, 'a fresh host service decrypts the saved client')
  const backup = file + '.original'
  await fs.rename(file, backup)
  await fs.symlink(backup, file)
  await assert.rejects(runtime().snapshot(), /unsafe type or owner/)
  await fs.unlink(file)
  await fs.link(backup, file)
  await assert.rejects(runtime().snapshot(), /unsafe type or owner/)
  await fs.unlink(file)
  const plaintext = JSON.stringify({ version: 1, client: { clientId: syntheticClient.installed.client_id, clientSecret: syntheticClient.installed.client_secret }, accounts: [], drafts: [], actions: [] })
  await fs.writeFile(file, plaintext, { mode: 0o600 })
  assert.deepEqual(await storage.readProtectedJson(file, null, false), JSON.parse(plaintext), 'legacy-compatible reader intentionally accepts plaintext; strict collections must opt out')
  await assert.rejects(runtime().snapshot(), /encrypted|protected|unlock/, 'Google storage must reject unencrypted account collections')
  assert.equal(await fs.readFile(file, 'utf8'), plaintext, 'rejected plaintext is preserved for explicit recovery')
  await assert.rejects(runtime().configureClient(JSON.stringify(syntheticClient)), /encrypted|protected|unlock/)
  assert.equal(await fs.readFile(file, 'utf8'), plaintext, 'reconfiguration cannot silently replace rejected account bytes')
  await fs.unlink(file)
  await fs.rename(backup, file)
  assert.equal((await runtime().snapshot()).clientConfigured, true)
  await fs.writeFile(file, '{invalid-json')
  await assert.rejects(runtime().snapshot(), /encrypted|protected|unlock/)
  assert.equal(await fs.readFile(file, 'utf8'), '{invalid-json')
} finally {
  clients.forEach(client => client.dispose()); shutdown.forEach(callback => callback())
  if (previousHome === undefined) delete process.env.VONTRA_HOME; else process.env.VONTRA_HOME = previousHome
  if (previousKey === undefined) delete process.env.VONTRA_SECRET_KEY_BASE64; else process.env.VONTRA_SECRET_KEY_BASE64 = previousKey
  await fs.rm(root, { recursive: true, force: true })
}
console.log('Actual Google host encrypted-store restart, ownership/link checks and fail-closed recovery passed')
