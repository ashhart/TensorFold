import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { createJiti } from 'jiti'
import { getWhatsAppBridgeLifecycleEpoch, makeWhatsAppBridgeSource, stopAllWhatsAppBridges, validateWhatsAppSourcePatch } from '../../agent/whatsapp-bridge'

const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-wa-host-fixture-'))
const key = '__tensorfoldWhatsAppHostFixture'
const source = (id: string, patch = {}) => ({ ...makeWhatsAppBridgeSource('Synthetic account'), id, ...patch })
const state = {
  store: { settings: { whatsapp: { sources: [source('paired', { paired: true }), source('unlinked'), source('disabled', { enabled: false, paired: true }), source('missing', { paired: true })], lastSyncAt: null } }, win: { webContents: { send: (...args: any[]) => state.pushes.push(args) } } },
  handlers: new Map<string, (...args: any[]) => Promise<any>>(), pushes: [] as any[], starts: [] as any[], stops: [] as string[], saved: 0,
  registered: new Set(['paired', 'disabled']), viewer: false,
  beforeAuthRead: null as null | ((id: string) => void),
  failSave: false, duringSave: null as null | (() => void),
  preloadApi: null as any,
  validateWhatsAppSourcePatch,
  getBridgeEpoch: getWhatsAppBridgeLifecycleEpoch,
}
;(globalThis as any)[key] = state
try {
  const fixture = path.join(tmp, 'fixture.mjs')
  await fs.writeFile(fixture, `
const f=globalThis.${key};
export const store=f.store;
export const handle=(channel,cb)=>f.handlers.set(channel,cb);
export const saveSettings=async()=>{f.saved++;if(f.failSave)throw new Error('Synthetic disk refusal');f.duringSave?.()};
export const vontraPath=()=>'/tmp/unused-offline-fixture';
export const persistSettings=()=>{};
export const auditPreview=(value)=>value;
export const auditEvent=async()=>{};
export const requireSecurityRole=()=>{if(f.viewer)throw new Error('requires owner')};
export const makeWhatsAppBridgeSource=(label)=>({id:'added',label,enabled:true,paired:false,allowedChatJids:[],allowSelfChat:false});
export const validateWhatsAppSourcePatch=(patch)=>f.validateWhatsAppSourcePatch(patch);
export const getWhatsAppBridgeRuntime=()=>[{sourceId:'paired',phase:'disconnected',qr:null},{sourceId:'removed',phase:'connected',qr:'never expose'}];
export const getWhatsAppBridgeLifecycleEpoch=()=>f.getBridgeEpoch();
export const forgetWhatsAppBridgeRuntime=()=>{};
export const stopWhatsAppBridge=(id)=>f.stops.push(id);
export const startWhatsAppBridge=async(...args)=>{f.starts.push(args);return args[0]};
export const sendWhatsAppMessage=async()=>({ok:true});
export const whatsappEventFromMessage=(input)=>input;
export const whatsappPairingErrorPatch=(error)=>({paired:false,lastError:String(error)});
export const hasProtectedRegisteredWhatsAppAuthState=async(_base,id)=>{f.beforeAuthRead?.(id);return f.registered.has(id)};
export const createProtectedWhatsAppAuthState=async()=>({state:{creds:{registered:true}}});
export const shredWhatsAppAuthState=async()=>{};
export const removeWhatsAppSourceWithNativeConfirmation=async()=>{};
export const publicConnectorSettings=()=>f.store.settings;
export const contextBridge={exposeInMainWorld:(name,value)=>{if(name==='tensorfoldStudio')f.preloadApi=value}};
export const ipcRenderer={invoke:(channel,...args)=>f.handlers.get(channel)({},...args),on:()=>{},removeListener:()=>{}};
`, { mode: 0o600 })
  const alias = Object.fromEntries(['electron', '../../../agent/whatsapp-bridge', '../whatsapp-auth-state', '../../../store', '../persistence', '../audit', '../../../agent/security', '../ipc', '../../state', '../connector-runtime', '../connector-event-spine'].map((name) => [name, fixture]))
  const jiti = createJiti(__filename, { alias, moduleCache: false, fsCache: false })
  const { registerWhatsAppIpc, resumeSavedWhatsAppAccounts } = await jiti.import<any>(path.join(__dirname, 'connector-ipc/whatsapp.ts'))
  const deps = { channels: { rememberWhatsAppEvents: () => {} }, spine: { rememberConnectorEvents: () => {} }, callbacks: {} }
  registerWhatsAppIpc(deps)
  // Execute the actual preload object with an in-memory Electron seam, then call through
  // its real IPC payload. Positional-only host tests previously hid this wire mismatch.
  await jiti.import<any>(path.join(__dirname, '../../preload.ts'))
  const updateSource = (id: string, patch: unknown) => state.preloadApi.updateWhatsAppSource(id, patch)
  await resumeSavedWhatsAppAccounts(deps)
  assert.deepEqual(state.starts.map((call) => call[0].id), ['paired'], 'only enabled paired accounts with existing protected registered auth auto-resume')
  assert.equal(state.starts[0][6].requireRegistered, true, 'auto-resume cannot create an unregistered QR session')
  assert.equal(state.store.settings.whatsapp.sources.find((item) => item.id === 'missing')!.paired, false, 'missing protected auth requires explicit relink')
  const runtime = await state.handlers.get('whatsapp:runtime')!({})
  assert.equal(runtime.length, 1, 'snapshot excludes removed accounts')
  assert.equal(runtime[0].phase, 'disconnected', 'snapshot returns ephemeral lifecycle, not durable paired')
  await state.handlers.get('whatsapp:disconnect')!({}, 'paired')
  assert.equal(state.store.settings.whatsapp.sources.find((item) => item.id === 'paired')!.enabled, false)
  const beforeResume = state.starts.length
  await resumeSavedWhatsAppAccounts(deps)
  assert.equal(state.starts.length, beforeResume, 'explicit disconnect disables boot auto-resume')
  await state.handlers.get('whatsapp:startPairing')!({}, 'paired')
  assert.equal(state.store.settings.whatsapp.sources.find((item) => item.id === 'paired')!.enabled, true)
  const liveSource = state.starts.at(-1)![6].getSource
  await updateSource('paired', { allowedChatJids: ['+15551230001'], allowSelfChat: true })
  assert.equal(liveSource().allowSelfChat, true, 'runtime scope always reads current saved source')
  assert.deepEqual(liveSource().allowedChatJids, [['15551230001', 's.whatsapp.net'].join('@')])
  state.failSave = true
  await assert.rejects(() => updateSource('paired', { allowSelfChat: false }), /disk refusal/)
  assert.equal(liveSource().allowSelfChat, true, 'failed persistence leaves the accepted runtime scope unchanged')
  state.failSave = false
  state.duringSave = () => {
    state.duringSave = null
    state.store.settings = { ...state.store.settings, whatsapp: { ...state.store.settings.whatsapp, sources: state.store.settings.whatsapp.sources.map((item) => item.id === 'paired' ? { ...item, phoneNumber: '+15551230000' } : item) } }
  }
  await assert.rejects(() => updateSource('paired', { allowSelfChat: false }), /changed while saving/)
  assert.equal(liveSource().phoneNumber, '+15551230000', 'background identity update survives a stale scope-save transaction')
  assert.equal(liveSource().allowSelfChat, true, 'a rejected race does not commit a stale scope')
  await assert.rejects(() => updateSource('paired', { allowedChatJids: ['Display name'] }), /exact/)
  state.beforeAuthRead = (id) => { if (id === 'paired') stopAllWhatsAppBridges() }
  const beforeStop = state.starts.length
  await resumeSavedWhatsAppAccounts(deps)
  assert.equal(state.starts.length, beforeStop, 'global Quit/purge stop during protected-auth await prevents any later startup socket')
  state.beforeAuthRead = (id) => { if (id === 'paired') state.store.settings.whatsapp.sources = state.store.settings.whatsapp.sources.filter((item) => item.id !== id) }
  const beforeRace = state.starts.length
  await resumeSavedWhatsAppAccounts(deps)
  assert.equal(state.starts.length, beforeRace, 'source removal while reading protected auth wins over resume')
  state.viewer = true
  for (const name of ['whatsapp:runtime', 'whatsapp:startPairing', 'whatsapp:disconnect', 'whatsapp:updateSource']) {
    await assert.rejects(() => state.handlers.get(name)!({}, 'unlinked', {}), /owner/, `${name} rejects non-owner callers`)
  }
  console.log('Offline WhatsApp host resume, disconnect, snapshot, scope, source-removal race and role regressions passed')
} finally { delete (globalThis as any)[key]; await fs.rm(tmp, { recursive: true, force: true }) }
