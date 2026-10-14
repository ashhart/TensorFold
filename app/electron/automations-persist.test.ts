// Run: jiti electron/automations-persist.test.ts
//
// Reported bug: in Automations, changing the MODEL does not save — after reload it
// reverts to the previous/default model. This test exercises the REAL persistence
// round-trip (the same saveAutomations -> writeProtectedJson -> file -> readProtectedState
// -> loadAutomations path the running app uses) plus the EXACT mutation the IPC handler
// `automations:update` performs (Object.assign(a, patch, { updatedAt })), to prove whether
// the model field survives a persist -> reload at the store/persistence layer.
import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

// Isolate persistence to a temp dir so we touch neither ~/.vontra nor an Electron app.
process.env.VONTRA_HOME = mkdtempSync(path.join(os.tmpdir(), 'vontra-automations-'))

const { saveAutomations, loadAutomations } = await import('./store.ts')
import type { Automation } from './agent/types'

function mkAutomation(p: Partial<Automation>): Automation {
  return {
    id: 'auto-1',
    name: p.name || 'Untitled automation',
    prompt: p.prompt || '',
    scheduleKind: p.scheduleKind || 'manual',
    everyMinutes: p.everyMinutes ?? 60,
    atHour: p.atHour ?? 9,
    atMinute: p.atMinute ?? 0,
    providerId: p.providerId ?? null,
    providerAccountId: p.providerAccountId ?? null,
    model: p.model ?? null,
    mode: p.mode || 'chat',
    cwd: p.cwd ?? null,
    deliverTo: p.deliverTo || 'inbox',
    enabled: p.enabled ?? false,
    lastRunAt: null,
    nextRunAt: null,
    lastStatus: null,
    runs: [],
    createdAt: Date.now(),
    updatedAt: Date.now(),
  }
}

// Mirror of electron/main/host/automations.ts `automations:update` handler mutation.
function applyUpdateHandler(a: Automation, patch: Partial<Automation>): Automation {
  Object.assign(a, patch, { updatedAt: Date.now() })
  return a
}

async function modelRoundTrip() {
  // 1. Create an automation pinned to a NON-default model on a NAMED provider account.
  const initial = mkAutomation({
    name: 'News brief',
    providerId: 'openai',
    providerAccountId: 'acct-work',
    model: 'gpt-5.5',
    scheduleKind: 'daily',
  })
  await saveAutomations([initial])

  // 2. User edits the automation in the UI and picks a DIFFERENT model. The renderer's
  //    editablePatch() ships { providerId, providerAccountId, model, ... }; the IPC handler
  //    Object.assign()s it onto the stored record. Reproduce exactly that.
  const reloadedForEdit = await loadAutomations()
  assert.equal(reloadedForEdit.length, 1, 'precondition: one automation persisted')
  const edited = applyUpdateHandler(reloadedForEdit[0], {
    providerId: 'openai',
    providerAccountId: 'acct-work',
    model: 'gpt-5.5-codex-high', // the NEW, non-default model
  })
  await saveAutomations([edited])

  // 3. Reload from disk (simulating an app restart) and assert the change survived.
  const afterReload = await loadAutomations()
  assert.equal(afterReload.length, 1, 'automation still present after reload')
  const a = afterReload[0]
  assert.equal(a.model, 'gpt-5.5-codex-high', 'changed model must survive persist -> reload (the reported bug)')
  assert.equal(a.providerId, 'openai', 'providerId preserved')
  assert.equal(a.providerAccountId, 'acct-work', 'providerAccountId preserved')
  console.log('  ✓ automation model change survives the real save -> reload round-trip')
}

await modelRoundTrip()
console.log('automations-persist round-trip tests passed')
