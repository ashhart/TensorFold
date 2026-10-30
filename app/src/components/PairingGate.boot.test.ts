// Run: node src/components/PairingGate.boot.test.ts
// #1410: main.tsx shows PairingGate without preparing the demo, so its imports must not reach ipc.ts or the store.
import assert from 'node:assert/strict'
import { createJiti } from 'jiti'
import { renderToStaticMarkup } from 'react-dom/server'
import { createElement } from 'react'

const globals = globalThis as any
Object.defineProperty(globals, 'window', { configurable: true, value: { addEventListener() {}, removeEventListener() {}, location: { href: 'https://host.example/' } } })
Object.defineProperty(globals, 'localStorage', { configurable: true, value: { getItem: () => null, setItem() {}, removeItem() {} } })

const jiti = createJiti(__filename, { jsx: { runtime: 'automatic' }, moduleCache: false })
const { PairingGate } = await jiti.import<typeof import('./PairingGate')>('./PairingGate.tsx')
const markup = renderToStaticMarkup(createElement(PairingGate))
assert.match(markup, /Pair this browser/)
assert.match(markup, /Pairing link or code/)
const { prepareBrowserDemo } = await jiti.import<typeof import('../lib/browser-demo-bootstrap')>('../lib/browser-demo-bootstrap.ts')
let demoLoads = 0
await prepareBrowserDemo(async () => { demoLoads++; return { createMockApi: () => ({}) as any } })
assert.equal(demoLoads, 1, 'pairing must not have prepared or required the browser demo')
console.log('Unpaired host page renders the pairing screen without the browser demo')
