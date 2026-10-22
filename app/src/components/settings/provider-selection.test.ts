import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createJiti } from 'jiti'
import type { Provider, Settings } from '../../lib/types'
import type { SettingsHookDeps } from './types'

const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-provider-selection-'))
const key = '__tensorfoldProviderSelectionFixture'
const state = { hooks: [] as any[], cursor: 0, draft: {} as Settings, fetchModels: async (_provider: Provider): Promise<string[]> => [] }
;(globalThis as any)[key] = state
let passed = 0
let failed = 0
const check = async (name: string, work: () => Promise<void>) => {
  try { await work(); passed++; console.log(`  ✓ ${name}`) }
  catch (error) { failed++; console.error(`  FAIL ${name}`); console.error(error) }
}
const provider = (over: Partial<Provider> = {}): Provider => ({
  id: 'example', name: 'Example endpoint', baseUrl: 'https://example.invalid/v1', apiKey: '', models: ['m1', 'm2'], ...over,
})

try {
  const fixture = path.join(tmp, 'fixture.mjs')
  await fs.writeFile(fixture, `
const f=globalThis.${key};
export const api={fetchModels:(provider)=>f.fetchModels(provider)};
export const useStore={getState:()=>({settings:f.draft})};
export const useState=(initial)=>{
  const index=f.cursor++;
  if(!(index in f.hooks)) f.hooks[index]=typeof initial==='function'?initial():initial;
  return [f.hooks[index],(value)=>{f.hooks[index]=typeof value==='function'?value(f.hooks[index]):value}];
};
export const useRef=(initial)=>useState({current:initial})[0];
export const DEFAULT_PROVIDER_SETUP={mode:'local',localRuntime:'custom',localLocation:'this-mac',providerName:'',apiKey:'',port:'8000',basePath:'/v1'};
export const LOCAL_RUNTIME_PRESETS=[{key:'custom',label:'vLLM / llama.cpp',defaultPort:'8000',defaultApiKey:'',defaultModels:['local-model'],source:'custom'}];
export const CLOUD_PROVIDER_PRESETS=[{key:'example',auth:'api-key',provider:{name:'Example endpoint',baseUrl:'https://example.invalid/v1'}}];
export const newProviderId=()=>'new-provider';
export const patchSpendBudget=()=>{};
export const providerPresetNeedsApiKey=()=>true;
`, { mode: 0o600 })
  const jiti = createJiti(__filename, {
    alias: { react: fixture, '../../lib/ipc': fixture, '../../lib/store': fixture, './shared': fixture },
    moduleCache: false, fsCache: false,
  })
  const { useProviderSettings } = await jiti.import<typeof import('./useProviderSettings')>(path.join(__dirname, 'useProviderSettings.ts'))
  let commits: Settings[] = []
  const commit = (next: Settings) => { commits.push(next); state.draft = next }
  const editDraft: SettingsHookDeps['editDraft'] = (next) => { state.draft = typeof next === 'function' ? next(state.draft) : next }
  const render = () => {
    state.cursor = 0
    return useProviderSettings({ draft: state.draft, editDraft, setDraft: editDraft, commit, patch: () => {}, save: commit })
  }
  const reset = (providers: Provider[], models = ['m1', 'm2']) => {
    state.hooks = []; commits = []
    state.draft = { providers, activeProviderId: providers[0]?.id ?? null, activeModel: providers[0]?.models[0] ?? null } as Settings
    state.fetchModels = async () => models
  }
  const connect = async () => { await render().connect(state.draft.providers[0]); return state.draft.providers[0] }

  await check('Add and connect saves the catalogue before narrowing and reconnecting', async () => {
    reset([])
    await render().addConfiguredProvider(true)
    assert.deepEqual(state.draft.providers[0].discoveredModels, ['m1', 'm2'])
    render().updateProvider('new-provider', { models: ['m1'] })
    assert.deepEqual((await connect()).models, ['m1'])
  })
  await check('Connect preserves a legacy narrowed selection without a saved catalogue', async () => {
    reset([provider({ models: ['m1'] })])
    assert.deepEqual((await connect()).models, ['m1'])
    assert.deepEqual(state.draft.providers[0].discoveredModels, ['m1', 'm2'])
    assert.deepEqual((await connect()).models, ['m1'])
  })
  await check('Connect preserves deselect-all with a saved catalogue', async () => {
    reset([provider({ models: [], discoveredModels: ['m1', 'm2'] })])
    assert.deepEqual((await connect()).models, [])
    assert.deepEqual((await connect()).models, [])
  })
  await check('unchanged catalogues preserve selection order and unrelated providers', async () => {
    const other = provider({ id: 'other' })
    reset([provider({ models: ['m2', 'm1'], discoveredModels: ['m1', 'm2', 'm3'] }), other], ['m1', 'm2', 'm3'])
    assert.deepEqual((await connect()).models, ['m2', 'm1'])
    assert.equal(state.draft.providers[1], other)
  })
  await check('refresh removes unavailable selections and enables only genuinely new models', async () => {
    reset([provider({ models: ['m1', 'removed'], discoveredModels: ['m1', 'm2', 'removed'] })], ['m1', 'm2', 'new'])
    assert.deepEqual((await connect()).models, ['m1', 'new'])
    assert.deepEqual(state.draft.providers[0].discoveredModels, ['m1', 'm2', 'new'])
    assert.deepEqual((await connect()).models, ['m1', 'new'])
  })
  await check('an empty selection still enables genuinely new models, not disabled ones', async () => {
    reset([provider({ models: [], discoveredModels: ['m1', 'm2'] })], ['m1', 'm2', 'new'])
    assert.deepEqual((await connect()).models, ['new'])
  })
  await check('a known catalogue with every selected model removed does not reset selection', async () => {
    reset([provider({ models: ['removed'], discoveredModels: ['removed', 'm1', 'm2'] })])
    assert.deepEqual((await connect()).models, [])
  })
  await check('first Connect populates a blank provider or replaces unavailable preset defaults', async () => {
    for (const models of [[], ['local-model']]) {
      reset([provider({ models })])
      assert.deepEqual((await connect()).models, ['m1', 'm2'])
    }
  })
  await check('failed and empty Connect responses leave settings unchanged', async () => {
    for (const fetchModels of [async () => [], async () => { throw new Error('Synthetic fetch failure') }]) {
      reset([provider({ models: ['m1'], discoveredModels: ['m1', 'm2'] })])
      const before = state.draft
      state.fetchModels = fetchModels
      await connect()
      assert.equal(state.draft, before)
      assert.equal(commits.length, 0)
      assert.equal(render().conn.example.state, 'error')
    }
  })
  console.log(`provider-selection.test.ts: ${passed} passed, ${failed} failed, 0 skipped`)
  if (failed) process.exitCode = 1
} finally {
  delete (globalThis as any)[key]
  await fs.rm(tmp, { recursive: true, force: true })
}
