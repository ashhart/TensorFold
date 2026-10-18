import assert from 'node:assert/strict'
import { constants, promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import ts from 'typescript'
import { googleWorkspacePreload } from '../../../preload/google-workspace'
import { GOOGLE_WORKSPACE_API_SCHEMA } from '../../../../shared/google-workspace-api'
import { requireNativeConfirmation } from '../sensitive-confirmation'

const handlers = new Map<string, (...args: any[]) => Promise<any>>()
let denied = false; let approved = false; let executions = 0; let disconnections = 0; let imports = 0
let selection = { canceled: true, filePaths: [] as string[] }
const roles: string[] = []; const audits: unknown[] = []; const confirmations: any[] = []
const canonical = { id: 'review', accountEmail: 'owner@example.com', action: { kind: 'gmail.send', email: { to: ['recipient@example.com'], bcc: ['hidden@example.com'], subject: 'Canonical subject', text: 'Exact approved content' } } }
const service: any = {
  snapshot: async () => ({ clientConfigured: true, accounts: [] }),
  configureClient: async (json: string) => { imports++; assert.equal(json, '{"installed":{}}'); return { clientConfigured: true, accounts: [] } },
  getReview: async (receipt: any) => { assert.equal(receipt.id, 'review'); return canonical },
  executeReviewed: async (receipt: any) => { executions++; assert.equal(receipt.id, 'review'); return { id: 'review', status: 'succeeded' } },
  disconnect: async () => { disconnections++ },
}
const dependencies = {
  dialog: { showOpenDialog: async () => selection }, fs, constants,
  googleWorkspaceService: () => service,
  auditEvent: async (kind: string, data: unknown) => { audits.push({ kind, data }) },
  requireSecurityRole: (role: string) => { roles.push(role); if (denied) throw new Error('Synthetic owner denied') },
  defaultHandle: (name: string, callback: (...args: any[]) => Promise<any>) => handlers.set(name, callback),
  requireNativeConfirmation,
}
// Execute the actual registration function with explicit host dependencies.
const source = await fs.readFile(new URL('./google-workspace-ipc.ts', import.meta.url), 'utf8')
const ast = ts.createSourceFile('google-workspace-ipc.ts', source, ts.ScriptTarget.ES2022, true)
const declaration = ast.statements.find((item): item is ts.FunctionDeclaration => ts.isFunctionDeclaration(item) && item.name?.text === 'registerGoogleWorkspaceIpc')
assert.ok(declaration)
const printed = ts.createPrinter().printNode(ts.EmitHint.Unspecified, declaration, ast).replace(/^export\s+/, '')
const compiled = ts.transpileModule(printed, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
const register = new Function(...Object.keys(dependencies), `${compiled}; return registerGoogleWorkspaceIpc;`)(...Object.values(dependencies))
register({ confirmSensitiveAction: async (input: unknown) => { confirmations.push(input); return approved } })
const invoke = (channel: string, ...args: unknown[]) => {
  assert.ok(handlers.has(channel), `Missing actual host channel ${channel}`)
  return handlers.get(channel)!({}, ...args)
}
const api = googleWorkspacePreload(invoke)
assert.deepEqual(Object.keys(api).sort(), Object.keys(GOOGLE_WORKSPACE_API_SCHEMA).sort())
denied = true
for (const method of Object.values(api)) await assert.rejects((method as any)({}, {}), /owner denied/)
assert.equal(roles.length, 17, 'every forwarded method is owner gated before reading its arguments')
assert.equal(executions + disconnections + imports, 0)
denied = false
assert.deepEqual(await api.googleWorkspaceStatus(), { clientConfigured: true, accounts: [] })
assert.equal(await api.googleWorkspaceImportClient(), null)
const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'google-ipc-synthetic-'))
try {
  const client = path.join(temp, 'client.json')
  await fs.writeFile(client, '{"installed":{}}')
  await fs.symlink(client, path.join(temp, 'symlink.json'))
  selection = { canceled: false, filePaths: [path.join(temp, 'symlink.json')] }
  await assert.rejects(api.googleWorkspaceImportClient(), /ELOOP/)
  selection.filePaths = [temp]
  await assert.rejects(api.googleWorkspaceImportClient(), /Choose a Google Desktop/)
  selection.filePaths = [client]
  assert.equal((await api.googleWorkspaceImportClient())?.clientConfigured, true)
  assert.equal(imports, 1)
  await fs.writeFile(client, ' '.repeat(65537))
  await assert.rejects(api.googleWorkspaceImportClient(), /smaller than/)
  assert.equal(imports, 1)
  const receipt = { id: 'review', accountId: 'synthetic-account', revision: 1, digest: 'synthetic-digest', action: { text: 'forged renderer content' } }
  await assert.rejects(api.googleWorkspaceExecuteAction(receipt), /cancelled by the local user/)
  assert.equal(executions, 0)
  assert.match(confirmations.at(-1).detail, /Exact approved content/)
  assert.match(confirmations.at(-1).detail, /hidden@example.com/)
  assert.doesNotMatch(confirmations.at(-1).detail, /forged renderer/)
  await assert.rejects(api.googleWorkspaceDisconnect('synthetic-account'), /cancelled by the local user/)
  assert.match(confirmations.at(-1).detail, /all services and clients/)
  assert.equal(disconnections, 0)
  approved = true
  assert.equal((await api.googleWorkspaceExecuteAction(receipt)).status, 'succeeded')
  assert.equal(executions, 1)
  await api.googleWorkspaceDisconnect('synthetic-account')
  assert.equal(disconnections, 1)
  assert.ok(roles.every(role => role === 'owner'))
  assert.doesNotMatch(JSON.stringify(audits), /hidden@example|Exact approved content|installed|synthetic-digest/)
} finally { await fs.rm(temp, { recursive: true, force: true }) }
console.log('Actual Google preload and IPC owner gates, bounded client import and canonical native confirmation passed')
