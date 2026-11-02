import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
import { qualificationImage } from '../../../agent/model-qualification/fixture-image'
import { capturedWindowAttachments, inferWindowKind } from '../../../agent/quick-ask/window-context'

const png = Buffer.from(qualificationImage().split(',')[1], 'base64')
const thumbnail = { isEmpty: () => false, toDataURL: qualificationImage, toPNG: () => png }
const sources = [{ id: 'own', name: 'TensorFold' }, { id: 'secret', name: 'Passwords' }, { id: 'doc', name: 'sample.txt - TextEdit' }].map(source => ({ ...source, thumbnail }))
let enumerate = async () => sources, extracts = 0
const requests: any[] = []
const source = readFileSync(path.join(import.meta.dirname, 'window-context.ts'), 'utf8')
const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText
const output = { exports: {} as any }
new Function('require', 'module', 'exports', js)((name: string) => {
  if (name === 'electron') return { BrowserWindow: { getAllWindows: () => [{ isDestroyed: () => false, getMediaSourceId: () => 'own' }] }, desktopCapturer: { getSources: (request: unknown) => { requests.push(request); return enumerate() } } }
  if (name === 'node:crypto') return { randomUUID: (() => { let id = 0; return () => `token-${++id}` })() }
  if (name.endsWith('computer-use-host')) return { computerHelperPath: () => '/unused' }
  if (name.endsWith('attachment-text')) return { createAttachmentTextService: () => ({ extract: async () => { extracts++; return { ok: true, text: 'Visible sample' } }, shutdown() {} }) }
  if (name.endsWith('native-window-picker')) return { supportsNativeAskPicker: () => false, createNativeAskPicker: () => ({ busy: false, cancel() { return true } }) }
  if (name.endsWith('agent/quick-ask/window-context')) return { capturedWindowAttachments, inferWindowKind }
  throw new Error(`Unexpected dependency ${name}`)
}, output, output.exports)
const picker = output.exports.createAskContextPicker()
let selection = await picker.pick()
assert.equal(selection.kind, 'choose'); assert.deepEqual(selection.windows.map((item: any) => item.title), ['sample.txt - TextEdit'])
assert.equal(JSON.stringify(selection).includes('"sourceId"'), false)
await assert.rejects(picker.use('doc'), /expired/)
assert.equal(extracts, 0, 'renderer-supplied source IDs never reach capture')
selection = await picker.pick(); const attachments = await picker.use(selection.windows[0].token)
assert.equal(attachments.length, 2); assert.equal(extracts, 1)
await assert.rejects(picker.use(selection.windows[0].token), /expired/)
assert.ok(requests.every(request => JSON.stringify(request.types) === '["window"]'), 'there is no full-display fallback')
selection = await picker.pick(); enumerate = async () => sources.map(item => item.id === 'doc' ? { ...item, name: 'Different title' } : item)
await assert.rejects(picker.use(selection.windows[0].token), /changed or closed/)
let complete!: (value: any) => void
enumerate = () => new Promise(resolve => { complete = resolve })
const listing = picker.pick(); picker.cancel(); complete(sources)
assert.deepEqual(await listing, { kind: 'cancelled' }, 'cancelled enumeration cannot repopulate choice tokens')
picker.dispose()
console.log('Fallback picker excludes own and protected windows; opaque single-use choices, stale-window checks and cancellation prevent unintended capture.')
