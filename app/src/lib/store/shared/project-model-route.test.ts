import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { createJiti } from 'jiti'
import type { Project, Provider, Settings, Thread } from '../../types'
import type { AppState } from '../types'

const root = await mkdtemp(path.join(os.tmpdir(), 'tf-project-model-route-'))
try {
  const stub = path.join(root, 'ipc.cjs')
  await writeFile(stub, 'exports.api = { configureThread: async (...args) => globalThis.projectModelCalls.push(args) }')
  const calls: unknown[][] = []
  const globals = globalThis as unknown as { projectModelCalls: unknown[][] }
  globals.projectModelCalls = calls
  const load = createJiti(__filename, { alias: { '../../ipc': stub } })
  const { applyProjectModel } = await load.import<typeof import('./model-policy')>('./model-policy')
  const provider = {
    id: 'fixture', name: 'Fixture', kind: 'local', models: ['base-model'], defaultAccountId: 'named',
    accounts: [{ id: 'named', name: 'Named fixture', models: ['named-model'] }],
  } as Provider
  const settings = { providers: [provider], activeProviderId: provider.id, activeProviderAccountId: 'named', activeModel: 'named-model' } as Settings
  const project = { id: 'project', defaultProviderId: provider.id, defaultProviderAccountId: null, defaultModel: 'base-model' } as Project

  for (const kind of ['chat', 'task'] as const) {
    const thread = { id: kind, kind } as Thread
    applyProjectModel(() => ({ projects: [project] }) as AppState, settings, project.id, thread)
    assert.deepEqual([thread.providerId, thread.providerAccountId, thread.model], [provider.id, null, 'base-model'])
    assert.deepEqual(calls.at(-1), [kind, provider.id, 'base-model', null])
  }

  const legacy = { ...project, defaultProviderAccountId: undefined }
  const oldThread = { id: 'older-project', kind: 'chat' } as Thread
  applyProjectModel(() => ({ projects: [legacy] }) as AppState, settings, legacy.id, oldThread)
  assert.deepEqual([oldThread.providerAccountId, oldThread.model], ['named', 'named-model'])

  const withoutProviderDefault = { ...provider, defaultAccountId: null }
  const task = { id: 'global-account', kind: 'task' } as Thread
  applyProjectModel(() => ({ projects: [project] }) as AppState, { ...settings, providers: [withoutProviderDefault] }, project.id, task)
  assert.deepEqual([task.providerAccountId, task.model], [null, 'base-model'])

  const legacyTask = { id: 'legacy-global-account', kind: 'task' } as Thread
  applyProjectModel(() => ({ projects: [legacy] }) as AppState, { ...settings, providers: [withoutProviderDefault] }, legacy.id, legacyTask)
  assert.deepEqual([legacyTask.providerAccountId, legacyTask.model], ['named', 'named-model'])
} finally {
  await rm(root, { recursive: true, force: true })
}

console.log('Project Chat and Task inherit the selected provider key and keep legacy named defaults')
