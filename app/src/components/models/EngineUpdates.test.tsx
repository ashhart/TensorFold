import assert from 'node:assert/strict'
import path from 'node:path'
import { harness, jsx, load, nodes, settle, text } from '../reading/reading-test-harness'
import type { EngineRuntimeSnapshot } from '../../../electron/agent/types/engine-runtime'
const h = harness()
let snapshot: EngineRuntimeSnapshot = { bundledVersion: '1.1.6', active: null, installed: [], available: [], state: 'idle', progress: null, checkedAt: null, error: null }
const calls: string[] = []
const release = { id: 'runtime-fixture', version: '1.1.7', platform: 'linux' as const, arch: 'x64' as const, backend: 'cuda', notes: 'Fixture release notes.' }
const api = {
  engineRuntimeStatus: async () => snapshot,
  engineRuntimeCheck: async () => { calls.push('check'); return snapshot = { ...snapshot, available: [release], checkedAt: Date.now() } },
  engineRuntimeInstall: async (id: string) => { calls.push(`install:${id}`); return snapshot = { ...snapshot, installed: [release] } },
  engineRuntimeUse: async (id: string | null) => { calls.push(`use:${id}`); return snapshot = { ...snapshot, active: id ? release : null } },
  engineRuntimeCancel: async () => { calls.push('cancel'); return true },
}
const { EngineUpdates } = load(path.join(__dirname, 'EngineUpdates.tsx'), {
  react: h.hooks, 'react/jsx-runtime': jsx, '../../lib/ipc': { api }, '../ui': { Button: 'button' },
  '../../lib/preferences/useAuthoredText': { useAuthoredText: () => (value: string) => value },
})
const render = () => h.render(() => EngineUpdates())
const click = async (label: string) => {
  const button = nodes(render()).find(n => n.type === 'button' && text(n) === label)
  assert.ok(button, label); assert.equal(!!button.props.disabled, false)
  button.props.onClick(); await settle()
}
try {
  render(); h.flush(); await settle()
  assert.match(text(render()), /1.1.6.*bundled/)
  await click('Check for engine updates')
  assert.match(text(render()), /Fixture release notes/)
  await click('Download'); assert.deepEqual(calls, ['check', 'install:runtime-fixture'])
  assert.equal(snapshot.active, null, 'downloading does not activate behind the user')
  await click('Use this runtime'); assert.match(text(render()), /1.1.7.*cuda.*active/)
  await click('Use bundled engine'); assert.equal(snapshot.active, null)
  api.engineRuntimeCheck = async () => { throw new Error('Offline fixture') }
  await click('Check for engine updates'); assert.match(text(render()), /Offline fixture/)
} finally { h.cleanup() }
console.log('Engine update UI check, release notes, download, explicit activation, rollback and error journeys passed.')
