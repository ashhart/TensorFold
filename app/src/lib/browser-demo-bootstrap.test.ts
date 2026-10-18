import assert from 'node:assert/strict'
import type { TensorFoldStudioApi } from '../../shared/studio-api'
import { prepareBrowserDemo, preparedBrowserDemo } from './browser-demo-bootstrap'

const globals = globalThis as any
let token: string | null = null
Object.defineProperty(globals, 'window', { configurable: true, value: {} })
Object.defineProperty(globals, 'localStorage', { configurable: true, value: { getItem: () => token } })
let loads = 0
const demo = { platform: 'browser' } as TensorFoldStudioApi
const load = async () => { loads++; return { createMockApi: () => demo } }
window.tensorfoldStudio = { platform: 'darwin' } as TensorFoldStudioApi
await prepareBrowserDemo(load)
assert.equal(loads, 0, 'desktop must not import or initialize fixtures')
delete window.tensorfoldStudio
window.vontra = { platform: 'darwin' } as TensorFoldStudioApi
await prepareBrowserDemo(load)
assert.equal(loads, 0, 'legacy desktop bridge must not import fixtures')
delete window.vontra
token = 'synthetic-device-session'
await prepareBrowserDemo(load)
assert.equal(loads, 0, 'paired production must not import fixtures')
token = null
assert.throws(preparedBrowserDemo, /has not been prepared/)
await assert.rejects(prepareBrowserDemo(async () => { throw new Error('Synthetic import failed') }), /Synthetic import failed/)
await Promise.all([prepareBrowserDemo(load), prepareBrowserDemo(load)])
assert.equal(loads, 1, 'browser initialization is once-only and can retry failed imports')
assert.equal(preparedBrowserDemo(), demo)
console.log('Lazy browser bootstrap desktop/paired/demo guards passed')
