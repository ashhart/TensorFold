import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createJiti } from 'jiti'

// Run the actual hook body/effect with in-memory React/IPC seams; no renderer or app starts.
const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-wa-remote-hook-'))
const key = '__tensorfoldWhatsAppRemoteHookFixture'
const state = { calls: [] as string[], cleanups: [] as Array<() => void>, api: {} as any, messages: [] as string[] }
;(globalThis as any)[key] = state
try {
  const fixture = path.join(tmp, 'fixture.mjs')
  await fs.writeFile(fixture, `
const f=globalThis.${key};
export const api=new Proxy({},{get:(_target,key)=>f.api[key]});
export const useState=(initial)=>[typeof initial==='function'?initial():initial,()=>{}];
export const useRef=(initial)=>({current:initial});
export const useEffect=(work)=>{const cleanup=work();if(typeof cleanup==='function')f.cleanups.push(cleanup)};
`, { mode: 0o600 })
  const load = async (platform: string) => {
    state.calls = []; state.messages = []; state.cleanups = []
    state.api = {
      platform,
      onWhatsAppRuntime: () => { state.calls.push('runtime-subscription'); return () => {} },
      onWhatsAppStatus: () => { state.calls.push('status-subscription'); return () => {} },
      whatsappRuntime: async () => { state.calls.push('runtime-read'); return [] },
      addWhatsAppSource: async () => { state.calls.push('add'); return { sources: [], lastSyncAt: null } },
    }
    const jiti = createJiti(__filename, { alias: { react: fixture, '../../lib/ipc': fixture }, moduleCache: false, fsCache: false })
    const { useBotConnectors } = await jiti.import<any>(path.join(__dirname, 'useBotConnectors.ts'))
    return useBotConnectors({ draft: { whatsapp: { sources: [] }, botControl: { telegram: {}, whatsapp: {} }, imessage: {} }, setDraft: () => {}, commit: () => {} }, (_key: string, message: string) => state.messages.push(message))
  }
  const remote = await load('web-remote')
  assert.equal(remote.whatsappLocalOnly, true)
  assert.deepEqual(state.calls, [], 'unrelated Settings mount over paired transport must not request forbidden QR/local runtime IPC')
  await remote.addWhatsAppAccount()
  assert.deepEqual(state.calls, [], 'paired caller cannot invoke local account mutation')
  assert.match(state.messages[0], /host Mac/)
  for (const platform of ['darwin', 'web']) {
    const local = await load(platform)
    assert.equal(local.whatsappLocalOnly, false)
    assert.deepEqual(state.calls, ['runtime-subscription', 'status-subscription', 'runtime-read'], 'desktop and synthetic demo retain local runtime snapshot wiring')
    for (const cleanup of state.cleanups) cleanup()
  }
  console.log('WhatsApp paired-web hook isolation and desktop/demo runtime wiring regressions passed')
} finally {
  for (const cleanup of state.cleanups) cleanup()
  delete (globalThis as any)[key]
  await fs.rm(tmp, { recursive: true, force: true })
}
