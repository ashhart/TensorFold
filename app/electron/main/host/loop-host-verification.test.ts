import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createJiti } from 'jiti'
import { WorkspaceMutationGuard } from './workspace-mutation-guard'

const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'loop-verification-config-'))
const fixtureKey = '__loopVerificationConfigFixture'
const provider = { id: 'fixture', models: ['fixture'] }
const state = {
  win: null, projects: [], rooms: [], tickets: [],
  threads: [{ id: 'thread', projectId: null, cwd: tmp, providerId: provider.id, model: 'fixture' }],
  settings: { providers: [provider], activeProviderId: provider.id, activeModel: 'fixture' },
}
;(globalThis as any)[fixtureKey] = state
try {
  const fixture = path.join(tmp, 'fixture.mjs')
  await fs.writeFile(fixture, [
    `export const store = globalThis.${fixtureKey};`,
    'export const persistThreads = () => {};',
    'export const persistTickets = () => {};',
    'export const resolveWorkspaceCwd = async cwd => cwd;',
    'export const readProtectedJson = async (_path, fallback) => fallback;',
    'export const writeProtectedJson = async () => {};',
    'export const vontraPath = (...parts) => parts.join("/");',
  ].join('\n'))
  const alias = { './workspace': fixture, '../state': fixture, './persistence': fixture, '../../store': fixture }
  const jiti = createJiti(__filename, { alias, moduleCache: false, fsCache: false })
  const { createLoopHost } = await jiti.import<typeof import('./loop-host')>(path.join(__dirname, 'loop-host.ts'))
  let baselineCalls = 0
  let roleCalls = 0
  const guard = new WorkspaceMutationGuard()
  const host = createLoopHost({
    workspaceMutationGuard: guard,
    startThreadRun: async () => { roleCalls++; throw new Error('No role may start with invalid verification.') },
    baselineRunner: async () => { baselineCalls++; throw new Error('Invalid verification must not touch Git.') },
  })
  try {
    for (const verify of [[], ['test\0skip'], Array(33).fill('test'), ['x'.repeat(2001)]]) {
      const result = await host.startLoop({
        source: 'solo', surfaceId: 'thread', mode: 'build', goal: 'Fixture', baseCwd: tmp, verify,
      })
      assert.equal(result.status, 'blocked')
      assert.equal(result.cycle, 0)
      assert.equal(result.lastVerdict?.passed, false)
      assert.equal(result.blockers[0].kind, 'NEEDS_PRODUCT_DECISION')
      assert.equal(baselineCalls, 0, 'no Git snapshot is attempted')
      assert.equal(roleCalls, 0, 'no model role is started')
      assert.equal(state.threads.length, 1, 'no throwaway role threads are created')
      guard.acquire({ workspaceKey: `local:${tmp}` }).release()
    }
  } finally {
    await host.quiesceAndClear()
  }
  console.log('Loop host rejects missing, NUL, excessive and oversized checks before Git or role side effects.')
} finally {
  delete (globalThis as any)[fixtureKey]
  await fs.rm(tmp, { recursive: true, force: true })
}
