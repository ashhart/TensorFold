import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, mkdirSync, existsSync, readFileSync, symlinkSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// Regression test for: "Security purge reports success even when on-disk writes fail."
// purgeLocalMemory clears the in-memory store then persists empty collections via
// Promise.allSettled. allSettled never throws and discards rejection results — so a failed
// disk write (disk full / EACCES / encryption error) used to be invisible and the operator
// was told their sensitive data was wiped when it still sat on disk. The fix inspects every
// settled result, records a FAILURE audit, and throws when any save rejected.
//
// Regression test for: "security:purgeLocalMemory IPC handler has no requireSecurityRole()
// gate." Every comparable destructive handler in core-ipc.ts (settings:save, webhooks:delete,
// archives:archive, threads:delete, ...) is role-gated; this one previously was not, letting
// ANY renderer-side caller (devtools console, etc) wipe all local app state regardless of the
// current security role. core-ipc.ts registers its handlers via the real ipcMain.handle
// (unavailable outside an Electron runtime), so it can't be imported and invoked directly in
// this headless test — instead we assert the actual source text of the
// 'security:purgeLocalMemory' registration calls requireSecurityRole('owner', ...) before
// purgeLocalMemory(), then separately prove requireSecurityRole('owner', ...) really does
// reject non-owner roles and allow 'owner' (the same real gate function core-ipc.ts calls).

function freshHome(prefix: string): string {
  return mkdtempSync(path.join(os.tmpdir(), prefix))
}

// We force a deterministic save rejection by making $VONTRA_HOME/state a regular FILE.
// Every state save (saveThreads/saveProjects/…) writes under $VONTRA_HOME/state/, so
// atomicWriteJson's `fs.mkdir(path.dirname(p), { recursive: true })` rejects with ENOTDIR.
function makeStateUnwritable(home: string): void {
  writeFileSync(path.join(home, 'state'), 'not-a-directory', { mode: 0o600 })
}

const noopDeps = () => ({
  aborters: new Map(),
  roomAborters: new Map(),
  missionAborters: new Map(),
  pendingApprovals: new Map(),
  pendingApprovalMeta: new Map(),
  missionApprovalResolvers: new Map(),
  abortAutomations() {},
  clearThreadCues() {},
  checkpointRecordCount: () => 0,
  pairingRecordCount: () => 0,
  eventSpineRecordCount: () => 0,
  providerStatusRecordCount: () => 0,
  clearCheckpoints: async () => {},
  clearPairingSessions: async () => {},
  clearEventSpine: async () => {},
  clearProviderStatus: async () => {},
  clearUsageMemory: async () => {},
  purgeBrowserAutomation: async () => {},
  quiesceOrchestrationJournal: async () => {},
  resumeOrchestrationJournal() {},
  quiesceProviderStatus: async () => {},
  resumeProviderStatus() {},
  quiesceLoops: async () => {},
  resumeLoops() {},
  workspaceMutationGuard: {
    acquireGlobal: async () => ({ release() {} }),
  },
})

function lastFailedAuditEvent(home: string): any {
  const file = path.join(home, 'security-events.jsonl')
  if (!existsSync(file)) return null
  const lines = readFileSync(file, 'utf8').trimEnd().split('\n').filter(Boolean).map((l) => JSON.parse(l))
  return [...lines].reverse().find((e) => e.kind === 'security.purge_local_memory_failed') ?? null
}

async function main() {
  // ---- Case 1: a save rejection must surface, NOT report success ----
  const failHome = freshHome('vontra-purge-fail-')
  makeStateUnwritable(failHome)
  process.env.VONTRA_HOME = failHome

  const { createSecurityPurgeRuntime, localProjectGitCandidates } = await import('./security-purge-runtime.ts')
  const { store } = await import('../state.ts')
  const { defaultSettings } = await import('../../store.ts')

  assert.deepEqual(
    localProjectGitCandidates([
      { id: 'local', cwd: '/srv/lookalike', workspace: { kind: 'local', path: '/srv/lookalike' } } as any,
      { id: 'ssh', cwd: '/srv/lookalike', workspace: { kind: 'ssh', hostId: 'host', path: '/srv/lookalike' } } as any,
      { id: 'empty', cwd: '' } as any,
    ]).map((project) => project.id),
    ['local'],
    'purge Git inspection excludes SSH paths so a local lookalike can never be pruned',
  )

  // Seed the in-memory store with "sensitive" data so the failed write would otherwise lie.
  store.threads = [{ id: 't1' } as any]
  store.projects = [{ id: 'p1' } as any]
  // contacts is wiped + saved during purge but was historically left out of the rollback snapshot,
  // so a failed purge silently emptied the learned contact directory in memory. Seed a record to
  // prove the rollback now restores it like every other collection.
  store.contacts = { records: { 'slack:U123': { id: 'slack:U123' } as any }, updatedAt: 1 } as any

  const runtime = createSecurityPurgeRuntime(noopDeps() as any)

  let threw: Error | null = null
  try {
    await runtime.purgeLocalMemory()
  } catch (e: any) {
    threw = e
  }
  assert.ok(threw, 'purgeLocalMemory must throw when an on-disk save rejects (not return { ok: true })')
  assert.match(String(threw?.message), /fail/i, 'error explains the purge failed to persist')
  assert.match(String(threw?.message), /threads/, 'error names the failed collection(s)')

  const failEvent = lastFailedAuditEvent(failHome)
  assert.ok(failEvent, 'a security.purge_local_memory_failed audit event is recorded on failure')
  assert.ok(Array.isArray(failEvent.failedLabels) && failEvent.failedLabels.includes('threads'), 'failure audit lists failed labels')
  assert.deepEqual(store.threads.map((thread: any) => thread.id), ['t1'], 'failed purge restores in-memory threads instead of leaving an empty false-success state')
  assert.deepEqual(store.projects.map((project: any) => project.id), ['p1'], 'failed purge restores in-memory projects instead of leaving an empty false-success state')
  assert.deepEqual(Object.keys((store.contacts as any)?.records ?? {}), ['slack:U123'], 'failed purge restores the in-memory contact directory instead of silently emptying it')

  // ---- Case 2: when every save resolves, it reports success exactly once ----
  const okHome = freshHome('vontra-purge-ok-')
  mkdirSync(path.join(okHome, 'state'), { recursive: true, mode: 0o700 })
  process.env.VONTRA_HOME = okHome

  store.threads = [{ id: 't2' } as any]
  store.artifacts = [{ id: 'artifact-secret' } as any]
  store.suggestions = [{ id: 'suggestion-secret', title: 'SUGGESTION-PLAINTEXT-SECRET' } as any]
  store.settings = {
    ...defaultSettings(),
    whatsapp: {
      ...defaultSettings().whatsapp,
      sources: [{
        id: 'wa_purge', label: 'Paired device', enabled: true, phoneNumber: '15551230000', paired: true,
        lastQrAt: null, lastEventAt: null, lastError: null, credentialRotatedAt: Date.now(),
        createdAt: 1, updatedAt: 1,
      }],
    },
  }
  for (const [relativeDir, file] of [
    ['email-outbox', 'message.eml'],
    ['usage', 'tokens.json'],
    ['browser-automation', 'page.json'],
    ['updates', 'verified-installer.dmg'],
    ['whatsapp-auth/wa_purge', 'auth-state.json'],
  ]) {
    mkdirSync(path.join(okHome, relativeDir), { recursive: true, mode: 0o700 })
    writeFileSync(path.join(okHome, relativeDir, file), `secret-${relativeDir}`, { mode: 0o600 })
  }
  const outsideAuthTarget = path.join(okHome, 'outside-auth-target.txt')
  writeFileSync(outsideAuthTarget, 'DO-NOT-FOLLOW', { mode: 0o600 })
  symlinkSync(outsideAuthTarget, path.join(okHome, 'whatsapp-auth', 'wa_purge', 'hostile-link'))
  writeFileSync(path.join(okHome, 'event-spine.jsonl'), 'secret-spine\n', { mode: 0o600 })
  mkdirSync(path.join(okHome, 'journals', 'missions'), { recursive: true, mode: 0o700 })
  writeFileSync(path.join(okHome, 'journals', 'missions', 'mission.ndjson'), 'SECRET-LATE-JOURNAL\n', { mode: 0o600 })
  writeFileSync(path.join(okHome, 'loops.json'), 'SECRET-LOOP-RUNTIME\n', { mode: 0o600 })
  writeFileSync(path.join(okHome, 'state', 'provider-status.json'), 'SECRET-PROVIDER-STATUS\n', { mode: 0o600 })
  const managedWorktree = path.join(okHome, 'rooms', 'project', 'room', 'code')
  mkdirSync(managedWorktree, { recursive: true, mode: 0o700 })
  execFileSync('git', ['init', '-q'], { cwd: managedWorktree })
  writeFileSync(path.join(managedWorktree, 'UNMERGED-USER-WORK.txt'), 'DO-NOT-DELETE', { mode: 0o600 })
  mkdirSync(path.join(okHome, 'rooms', 'project', 'room', 'scratchpad'), { recursive: true, mode: 0o700 })
  writeFileSync(path.join(okHome, 'rooms', 'project', 'room', 'scratchpad', 'notes.txt'), 'ROOM-PLAINTEXT-SECRET', { mode: 0o600 })
  writeFileSync(path.join(okHome, 'rooms', 'project', 'room', 'room.json'), 'ROOM-DUPLICATE-SECRET', { mode: 0o600 })
  writeFileSync(path.join(okHome, 'security-events.jsonl'), 'secret-old-audit\n', { mode: 0o600 })
  writeFileSync(path.join(okHome, 'security-events.anchor.json'), 'secret-old-anchor\n', { mode: 0o600 })
  let checkpointsCleared = 0
  let pairingsCleared = 0
  let eventSpineCleared = 0
  let providerStatusCleared = 0
  let usageMemoryCleared = 0
  let browserAutomationPurged = 0
  let journalQuiesced = 0
  let journalResumed = 0
  let providerStatusQuiesced = 0
  let providerStatusResumed = 0
  let loopsQuiesced = 0
  let loopsResumed = 0
  const { auditEvent: queueAuditEvent } = await import('./audit.ts')
  void queueAuditEvent('security.pre_purge_queued', { marker: 'QUEUED-PRE-PURGE-SECRET' })
  const okRuntime = createSecurityPurgeRuntime({
    ...noopDeps(),
    checkpointRecordCount: () => 2,
    pairingRecordCount: () => 3,
    eventSpineRecordCount: () => 4,
    providerStatusRecordCount: () => 5,
    clearCheckpoints: async () => { checkpointsCleared++ },
    clearPairingSessions: async () => { pairingsCleared++ },
    clearEventSpine: async () => { eventSpineCleared++ },
    clearProviderStatus: async () => { providerStatusCleared++ },
    clearUsageMemory: async () => { usageMemoryCleared++ },
    purgeBrowserAutomation: async () => { browserAutomationPurged++ },
    quiesceOrchestrationJournal: async () => { journalQuiesced++ },
    resumeOrchestrationJournal: () => { journalResumed++ },
    quiesceProviderStatus: async () => { providerStatusQuiesced++ },
    resumeProviderStatus: () => { providerStatusResumed++ },
    quiesceLoops: async () => { loopsQuiesced++ },
    resumeLoops: () => { loopsResumed++ },
  } as any)
  const result = await okRuntime.purgeLocalMemory()
  assert.equal(result.ok, true, 'happy path still reports ok: true when all saves resolve')
  assert.equal(checkpointsCleared, 1, 'purge clears the live turn-checkpoint ledger and awaits it')
  assert.equal(pairingsCleared, 1, 'purge clears live pairing sessions and awaits their durable write')
  assert.equal(eventSpineCleared, 1, 'purge clears the live event-spine replay state and awaits its journal rewrite')
  assert.equal(providerStatusCleared, 1, 'purge clears persisted and live provider status')
  assert.equal(usageMemoryCleared, 1, 'purge clears renderer-held usage state')
  assert.equal(browserAutomationPurged, 1, 'purge closes browser automation and clears its persistent partition')
  assert.equal(journalQuiesced, 1, 'purge quiesces orchestration journal writers before shredding')
  assert.equal(journalResumed, 1, 'orchestration journalling resumes only after the purge barrier')
  assert.equal(providerStatusQuiesced, 1, 'provider-status writers are quiesced before their file is shredded')
  assert.equal(providerStatusResumed, 1, 'provider-status collection resumes after purge without resurrecting old snapshots')
  assert.equal(loopsQuiesced, 1, 'purge stops and clears Loop Mode before shredding loops.json')
  assert.equal(loopsResumed, 1, 'new Loop Mode starts are admitted after purge completes')
  assert.equal(result.counts.turnCheckpoints, 2, 'purge result truthfully counts turn checkpoints')
  assert.equal(result.counts.pairingSessions, 3, 'purge result truthfully counts pairing records')
  assert.equal(result.counts.eventSpine, 4, 'purge result truthfully counts event-spine replay records')
  assert.equal(result.counts.providerStatus, 5, 'purge result truthfully counts provider-status records')
  assert.ok(result.counts.residualFiles >= 6, 'purge result includes residual plaintext files in its count')
  assert.ok(existsSync(path.join(okHome, 'state', 'threads.json')), 'threads.json was actually written on success')
  // State files are encrypted envelopes; the point is the (empty) purge result reached disk
  // and contains none of the seeded id.
  assert.ok(!readFileSync(path.join(okHome, 'state', 'threads.json'), 'utf8').includes('t2'), 'seeded data is not present on disk after a successful purge')
  assert.ok(!readFileSync(path.join(okHome, 'state', 'artifacts.json'), 'utf8').includes('artifact-secret'), 'artifact records are removed from protected state')
  for (const relative of ['email-outbox', 'usage', 'browser-automation', 'updates', 'whatsapp-auth', 'event-spine.jsonl', 'journals', 'loops.json']) {
    assert.equal(existsSync(path.join(okHome, relative)), false, `${relative} residual data is removed`)
  }
  assert.equal(readFileSync(outsideAuthTarget, 'utf8'), 'DO-NOT-FOLLOW', 'credential purge unlinks auth symlinks without following them')
  assert.equal(store.settings.whatsapp.sources[0]?.paired, false, 'credential purge marks linked WhatsApp sources unpaired')
  assert.equal(store.settings.whatsapp.sources[0]?.credentialRotatedAt, null, 'credential purge clears the retired credential age')
  assert.equal(existsSync(path.join(okHome, 'state', 'provider-status.json')), false, 'provider-status residual file is removed')
  assert.ok(!readFileSync(path.join(okHome, 'state', 'skill-suggestions.json'), 'utf8').includes('SUGGESTION-PLAINTEXT-SECRET'), 'skill suggestions are cleared from protected state')
  assert.equal(existsSync(path.join(managedWorktree, 'UNMERGED-USER-WORK.txt')), true, 'dirty managed worktree is preserved instead of silently deleting user changes')
  assert.equal(existsSync(path.join(okHome, 'rooms', 'project', 'room', 'scratchpad')), false, 'non-worktree room residuals are still purged')
  assert.equal(existsSync(path.join(okHome, 'rooms', 'project', 'room', 'room.json')), false, 'room metadata duplicate beside a preserved worktree is still purged')
  assert.ok(result.preservedResiduals.some((item) => item.path === managedWorktree && /Dirty managed Git worktree preserved/.test(item.reason)), 'purge result explicitly discloses the preserved dirty worktree')
  assert.ok(result.preservedCredentialsAndSettings.some((item) => /credentials/i.test(item)), 'purge result explicitly discloses preserved credentials and settings')
  assert.ok(!readFileSync(path.join(okHome, 'security-events.jsonl'), 'utf8').includes('secret-old-audit'), 'the old audit trail is erased before the fresh purge receipt is written')
  assert.ok(!readFileSync(path.join(okHome, 'security-events.jsonl'), 'utf8').includes('QUEUED-PRE-PURGE-SECRET'), 'a queued pre-purge audit append cannot race back after erasure')
  assert.ok(!readFileSync(path.join(okHome, 'security-events.anchor.json'), 'utf8').includes('secret-old-anchor'), 'the old audit anchor is replaced with the fresh purge receipt anchor')
  assert.equal(lastFailedAuditEvent(okHome), null, 'no failure audit on the happy path')

  // ---- Case 3: the IPC handler must require the 'owner' role ----
  // core-ipc.ts can't be imported/invoked directly here (it registers via the real
  // ipcMain.handle, which is unavailable outside an Electron runtime — confirmed by every
  // other *-ipc.test.ts in this codebase avoiding it the same way). So we assert against the
  // actual handler source text that the gate call is present and precedes purgeLocalMemory(),
  // which fails today (no requireSecurityRole call exists) and passes once the fix lands.
  const coreIpcPath = fileURLToPath(new URL('./core-ipc.ts', import.meta.url))
  const coreIpcSource = readFileSync(coreIpcPath, 'utf8')
  // Slice between adjacent top-level registrations. A regex that stops at the first
  // `})` is brittle now that native-confirmation input contains a nested object.
  const purgeStart = coreIpcSource.indexOf("handle('security:purgeLocalMemory'")
  const purgeEnd = coreIpcSource.indexOf("\nhandle('system:openFullDiskAccess'", purgeStart)
  assert.ok(purgeStart >= 0 && purgeEnd > purgeStart, "core-ipc.ts must register a bounded 'security:purgeLocalMemory' handler")
  const purgeHandlerBody = coreIpcSource.slice(purgeStart, purgeEnd)
  assert.match(
    purgeHandlerBody,
    /requireSecurityRole\(\s*'owner'\s*,/,
    "security:purgeLocalMemory handler must call requireSecurityRole('owner', ...) — every comparable destructive handler (settings:save, webhooks:delete, archives:archive) is role-gated and a full local-memory wipe deserves the same owner tier",
  )
  assert.ok(
    purgeHandlerBody.indexOf('requireSecurityRole') < purgeHandlerBody.lastIndexOf('purgeLocalMemory'),
    'requireSecurityRole(...) must run BEFORE purgeLocalMemory() so a denied role never triggers the destructive wipe',
  )
  assert.ok(
    purgeHandlerBody.indexOf('requireNativeConfirmation') < purgeHandlerBody.lastIndexOf('purgeLocalMemory'),
    'native confirmation must complete before the destructive purge is invoked',
  )

  // ---- Case 4: requireSecurityRole('owner', ...) — the real gate the handler calls — must
  // actually reject non-owner roles and allow 'owner'. This proves the gate's role semantics
  // independent of the (untestable-via-import) IPC registration checked above.
  const { requireSecurityRole } = await import('./audit.ts')

  for (const deniedRole of ['viewer', 'operator']) {
    process.env.VONTRA_SECURITY_ROLE = deniedRole
    assert.throws(
      () => requireSecurityRole('owner', 'purge all local memory/state'),
      /owner/,
      `requireSecurityRole('owner', ...) must reject role "${deniedRole}"`,
    )
  }

  process.env.VONTRA_SECURITY_ROLE = 'owner'
  assert.doesNotThrow(
    () => requireSecurityRole('owner', 'purge all local memory/state'),
    "requireSecurityRole('owner', ...) must allow role \"owner\"",
  )

  delete process.env.VONTRA_SECURITY_ROLE

  console.log('OK security-purge-runtime: failure surfaces, success still reports ok, owner-only IPC gate enforced')
}

await main()
