import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import vm from 'node:vm'
import ts from 'typescript'
import { isExternalProvider } from '../../../shared/provider-trust'
import { isRestrictedBot } from '../../../shared/bot-tool-policy'
import { executeTool } from '../tools'
import { PhoneService, type PhoneBackend } from './service'

// Execute the actual host callbacks with inert collaborators. No phone, engine,
// provider endpoint or user profile is opened by this regression.
const root = process.cwd()
function callback(file: string, name: string, call?: string): string {
  const source = readFileSync(path.join(root, file), 'utf8')
  const tree = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true)
  let found: string | undefined
  function visit(node: ts.Node) {
    if (call && ts.isCallExpression(node) && node.expression.getText(tree) === call) {
      const options = node.arguments.find(ts.isObjectLiteralExpression)
      const property = options?.properties.find((entry) => ts.isPropertyAssignment(entry) && entry.name.getText(tree) === name)
      if (property && ts.isPropertyAssignment(property)) found = property.initializer.getText(tree)
    } else if (!call && ts.isPropertyAssignment(node) && node.name.getText(tree) === name) found = node.initializer.getText(tree)
    ts.forEachChild(node, visit)
  }
  visit(tree)
  assert.ok(found, `Production ${name} callback exists`)
  return ts.transpileModule(`const callback = ${found};`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText + '\ncallback'
}
const provider = { id: 'custom-local-alias', name: 'Synthetic local alias', kind: 'local', transport: 'chat-completions', baseUrl: 'http://127.0.0.1:9/v1', apiKey: '', models: ['fixture'] }
const thread: any = { id: 'local-phone-fixture', kind: 'chat', providerId: provider.id, phoneLocalOnly: true }
const store = { threads: [thread], settings: { providers: [provider], activeProviderId: provider.id } }
let research = true
const collaborators = { store, HEADLESS_MODE: false, runtimeProvider: (value: unknown) => value, isExternalProvider, engineService: () => ({ hasExternalResearchEnabled: () => research }) }
const authorizeThread = vm.runInNewContext(callback('electron/main/composition/device-services.ts', 'authorizeThread', 'initializePhoneHost'), collaborators)
const privacyGuard = vm.runInNewContext(callback('electron/main/host/tool-context-runtime.ts', 'phonePrivacyGuard'), { ...collaborators, run: { thread, provider } })
const enableGuard = vm.runInNewContext(callback('electron/main/composition/device-services.ts', 'assertCanEnableCortheon', 'registerEngineIpc'), { store, aborters: new Map([[thread.id, new AbortController()]]), isRestrictedBot })
assert.equal(isExternalProvider(provider), false)
assert.throws(() => authorizeThread(thread.id), /Cortheon off/, 'a local API alias cannot bypass the known engine research setting')
assert.match(privacyGuard() ?? '', /Cortheon off/)
assert.throws(() => enableGuard(), /phone|execution boundaries/, 'an active local-only phone chat blocks enabling hidden research')
let toolCalls = 0, reads = 0, actions = 0, flipDuringRead = false
const sentinel = 'OWNED_SYNTHETIC_SCREEN_SENTINEL'
const emitted: any[] = []
const backend: PhoneBackend = {
  observe: async () => { reads++; if (flipDuringRead) research = true; return { width: 300, height: 600, image: null, app: 'example.synthetic', warnings: [], nodes: [{ text: `${sentinel}-${reads}`, description: '', role: 'Button', bounds: { x: 0, y: 0, width: 100, height: 100 }, editable: false, secure: false, enabled: true, focused: false, interactive: true }] } },
  act: async () => { actions++ }, disconnect: async () => {},
}
const device = { id: 'android:owned-fixture', name: 'Synthetic fixture', platform: 'android' as const, transport: 'usb' as const, state: 'ready' as const, detail: null }
const service = new PhoneService({ discover: async () => ({ adb: { available: false, path: null, version: null }, devices: [device], mirroring: { available: false, detail: null } }), connect: async () => backend, connectDeveloper: async () => ({ device, backend }), authorizeThread, cancelThread() {}, emit: (snapshot) => emitted.push(snapshot) })
try {
  await service.refresh(); await service.connect({ deviceId: device.id })
  assert.throws(() => service.grant(thread.id), /Cortheon off/)
  assert.equal(reads, 0, 'refused grant never captures a screen')
  const context: any = { cwd: '', currentThread: thread, phoneControlAvailable: true, phonePrivacyGuard: privacyGuard, runPhoneControl: async () => { toolCalls++; return service.tool({ action: 'observe' }, thread.id) } }
  assert.equal((await executeTool('phone_observe', {}, context)).ok, false)
  assert.equal(toolCalls, 0, 'direct tool dispatch checks the actual host privacy guard before its callback')
  research = false
  assert.equal(privacyGuard(), null)
  service.grant(thread.id)
  assert.equal((await executeTool('phone_observe', {}, context)).ok, true)
  assert.equal(reads, 1); assert.equal(toolCalls, 1)
  assert.ok(emitted.some((value) => value.observation?.nodes.some((node: any) => node.text === `${sentinel}-1`)), 'Cortheon-off local aliases retain explicitly granted phone inspection')
  emitted.length = 0; flipDuringRead = true
  const refused = await service.tool({ action: 'observe' }, thread.id)
  assert.equal(refused.ok, false)
  assert.equal(JSON.stringify(refused).includes(sentinel), false)
  assert.equal(emitted.some((value) => value.observation?.nodes.some((node: any) => node.text === `${sentinel}-2`)), false, 'a policy flip while the backend reads never projects the newly captured screen')
  assert.equal(service.snapshot().session?.threadId, null)
  assert.equal(service.snapshot().observation, null)
  assert.equal(actions, 0)
  console.log('Actual host phone boundaries passed: custom local alias grant/direct-tool/global-enable checks, normal explicit access and post-observation privacy-race refusal before screen projection.')
} finally { await service.shutdown() }
