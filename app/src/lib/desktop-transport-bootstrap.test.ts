import assert from 'node:assert/strict'
import { preparePairedSession } from './remote-pairing'
const globals = globalThis as any
Object.defineProperty(globals, 'window', { configurable: true, value: { tensorfoldStudio: { platform: 'darwin' } } })
let probes = 0
const fetchImpl = (async () => { probes++; throw new Error('Desktop must not probe network for browser pairing') }) as typeof fetch
assert.equal(await preparePairedSession(fetchImpl), 'desktop')
assert.equal(probes, 0)
console.log('Modern desktop bridge bypasses browser pairing and all network probes')
