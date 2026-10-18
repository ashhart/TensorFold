import { reviewedLanguageImport } from '../../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
import type { ModelQualificationReport } from '../../../../shared/model-fit'

async function main() {
  let providers: any[] = [{ id: 'synthetic', accounts: [{ id: 'a' }] }]
  let scope = 'profile-a'
  const requests: Array<{ resolve: (reports: ModelQualificationReport[]) => void; reject: (error: Error) => void }> = []
  const client = { modelQualificationReports: () => new Promise<ModelQualificationReport[]>((resolve, reject) => requests.push({ resolve, reject })) }
  let cursor = 0
  const slots: any[] = []; const effects = new Map<number, () => void>()
  const hooks = {
    useRef(initial: any) { const id = cursor++; return slots[id] ??= { current: initial } },
    useState(initial: any) { const id = cursor++; if (!(id in slots)) slots[id] = initial; return [slots[id], (next: any) => { slots[id] = typeof next === 'function' ? next(slots[id]) : next }] },
    useCallback(fn: any, deps: any[]) { const id = cursor++; const prior = slots[id]; if (!prior || deps.some((value, i) => value !== prior.deps[i])) slots[id] = { fn, deps }; return slots[id].fn },
    useEffect(fn: any, deps: any[]) { const id = cursor++; const prior = slots[id]; if (!prior || deps.some((value, i) => value !== prior.deps[i])) effects.set(id, () => { prior?.cleanup?.(); slots[id] = { deps, cleanup: fn() } }) },
  }
  const imports: Record<string, any> = { react: hooks, '../../../lib/ipc': { api: client }, '../../../lib/store': { useStore: (selector: any) => selector({ settings: { providers } }) } }
  const module = { exports: {} as any }
  new Function('require', 'module', 'exports', ts.transpileModule(readFileSync(path.join(__dirname, 'useModelQualifications.ts'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText)((name: string) => { const reviewed = reviewedLanguageImport(name, imports); if (reviewed.matched) return reviewed.value;  assert.ok(name in imports, name); return imports[name] }, module, module.exports)
  const render = (flush = true) => { cursor = 0; const result = module.exports.useModelQualifications(client, scope); if (flush) { const queued = [...effects.values()]; effects.clear(); for (const effect of queued) effect() }; return result }
  const settle = async () => { for (let i = 0; i < 10; i++) await Promise.resolve() }
  const report = (identity: string) => [{ model: 'fixture', identity }] as ModelQualificationReport[]
  const listeners = new Map<string, () => void>()
  const previousWindow = globalThis.window
  ;(globalThis as any).window = { addEventListener: (name: string, fn: () => void) => listeners.set(name, fn), removeEventListener: (name: string, fn: () => void) => { if (listeners.get(name) === fn) listeners.delete(name) } }
  try {
    render(); requests[0].resolve(report('original')); await settle()
    assert.equal(render().reports[0].identity, 'original')
    void render().refresh()
    providers = [{ id: 'synthetic', baseUrl: 'http://127.0.0.1:99/v1', accounts: [{ id: 'a' }] }]
    assert.equal(render(false).reports.length, 0, 'Config ownership changes mask cached reports synchronously before effects run')
    render(); requests[1].resolve(report('late-old-config')); await settle()
    assert.equal(render().reports.length, 0, 'A late old request cannot repopulate the changed route')
    requests[2].resolve(report('new-config')); await settle()
    assert.equal(render().reports[0].identity, 'new-config')
    providers = [{ ...providers[0], accounts: [{ id: 'b' }] }]
    assert.equal(render(false).reports.length, 0, 'Account changes mask earlier capability evidence')
    render(); requests[3].resolve(report('account-b')); await settle()
    assert.equal(render().reports[0].identity, 'account-b')
    providers = []
    assert.equal(render(false).reports.length, 0, 'Removed providers cannot keep observed labels')
    render(); requests[4].resolve([]); await settle()
    scope = 'profile-b'
    assert.equal(render(false).reports.length, 0)
    render(); requests[5].resolve(report('profile-b')); await settle()
    assert.equal(render().reports[0].identity, 'profile-b')
    scope = 'profile-c'
    assert.equal(render(false).reports.length, 0, 'Another profile/host never borrows prior evidence')
    render(); requests[6].resolve(report('profile-c')); await settle()
    assert.equal(render().reports[0].identity, 'profile-c')
    void render().refresh(); requests[7].reject(new Error('Synthetic refresh failure')); await settle()
    assert.equal(render().reports.length, 0, 'Refresh failure clears old observed results')
    assert.match(render().error, /remain unknown/)
    for (const slot of slots) slot?.cleanup?.()
    console.log('PASS actual qualification hook: synchronous config/account/removal/profile masking, stale-result rejection and empty-on-refresh-failure')
  } finally { (globalThis as any).window = previousWindow }
}
main().catch(error => { console.error(error); process.exitCode = 1 })
