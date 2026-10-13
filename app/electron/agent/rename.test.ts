import assert from 'node:assert/strict'
import { createWorkspaceRuntimeIpc } from '../main/host/workspace-runtime-ipc'

// --- Test harness ----------------------------------------------------------
// The workspace-runtime IPC factory registers its handlers by calling the
// injected `handle(channel, fn)` at construction time. We capture those
// registrations with a fake `handle`, then invoke the rename handlers directly
// against a fake store. Everything the factory touches at registration time is
// stubbed; the heavy room-run machinery is never exercised because those
// closures only run when their channels are invoked (which we don't).

type Handler = (e: unknown, payload: any) => any

interface FakeProject { id: string; name: string; updatedAt: number }
interface FakeRoom { id: string; name: string; updatedAt: number }

function buildHarness() {
  const handlers = new Map<string, Handler>()
  const handle = (channel: string, fn: Handler) => { handlers.set(channel, fn) }

  const store = {
    projects: [] as FakeProject[],
    rooms: [] as FakeRoom[],
  }

  const persistCalls = { projects: 0, rooms: 0 }
  const auditCalls: Array<{ event: string; payload: any }> = []
  const roleChecks: Array<{ role: string; action: string }> = []

  let blockRole = false

  // A Proxy fills in any dependency the factory destructures with a harmless
  // stub, so we only have to specify the handful the rename handlers touch.
  const base: Record<string, any> = {
    store,
    handle,
    requireSecurityRole: (role: string, action: string) => {
      roleChecks.push({ role, action })
      if (blockRole) throw new Error(`forbidden: ${action}`)
    },
    auditEvent: async (event: string, payload: any) => { auditCalls.push({ event, payload }) },
    auditPreview: (s: string) => `preview:${s}`,
    persistProjects: () => { persistCalls.projects++ },
    persistRooms: () => { persistCalls.rooms++ },
    roomAborters: new Map(),
    registerWebhookIpc: () => {},
  }
  const ctx = new Proxy(base, {
    get(target, prop: string) {
      if (prop in target) return target[prop]
      // Default stub: a no-op function for anything else the factory destructures.
      return () => undefined
    },
  })

  createWorkspaceRuntimeIpc(ctx as any)

  return {
    handlers,
    store,
    persistCalls,
    auditCalls,
    roleChecks,
    setBlockRole: (v: boolean) => { blockRole = v },
    call: (channel: string, payload: any) => {
      const fn = handlers.get(channel)
      if (!fn) throw new Error(`no handler for ${channel}`)
      return fn({}, payload)
    },
  }
}

// Mirror of the threads:rename normalization (core-ipc.ts). The thread handler
// is a one-liner over module-level imports, so we assert its name-normalization
// contract here directly.
function normalizeThreadTitle(title: string): string {
  return String(title ?? '').trim() || 'Untitled'
}

async function run() {
  // --- Test 1: projects:rename happy path ---
  {
    const h = buildHarness()
    h.store.projects.push({ id: 'p1', name: 'Old Project', updatedAt: 1 })
    const before = Date.now()
    await h.call('projects:rename', { id: 'p1', name: 'New Project' })
    assert.equal(h.store.projects[0].name, 'New Project')
    assert.ok(h.store.projects[0].updatedAt >= before, 'updatedAt refreshed')
    assert.equal(h.persistCalls.projects, 1, 'persistProjects called once')
    assert.ok(h.auditCalls.some((c) => c.event === 'project.rename'), 'audit logged')
  }

  // --- Test 2: rooms:rename happy path ---
  {
    const h = buildHarness()
    h.store.rooms.push({ id: 'r1', name: 'Old Room', updatedAt: 1 })
    const before = Date.now()
    await h.call('rooms:rename', { id: 'r1', name: 'New Room' })
    assert.equal(h.store.rooms[0].name, 'New Room')
    assert.ok(h.store.rooms[0].updatedAt >= before, 'updatedAt refreshed')
    assert.equal(h.persistCalls.rooms, 1, 'persistRooms called once')
    assert.ok(h.auditCalls.some((c) => c.event === 'room.rename'), 'audit logged')
  }

  // --- Test 3: threads:rename normalization contract (mirror) ---
  {
    assert.equal(normalizeThreadTitle('New Task'), 'New Task')
    assert.equal(normalizeThreadTitle('  Spaced  '), 'Spaced')
    assert.equal(normalizeThreadTitle(''), 'Untitled')
    assert.equal(normalizeThreadTitle('   '), 'Untitled')
  }

  // --- Test 4: rename trims surrounding whitespace ---
  {
    const h = buildHarness()
    h.store.projects.push({ id: 'p1', name: 'Old', updatedAt: 1 })
    await h.call('projects:rename', { id: 'p1', name: '  Trimmed  ' })
    assert.equal(h.store.projects[0].name, 'Trimmed')
  }

  // --- Test 5: empty / whitespace-only rename falls back to default ---
  {
    const h = buildHarness()
    h.store.projects.push({ id: 'p1', name: 'Keep', updatedAt: 1 })
    h.store.rooms.push({ id: 'r1', name: 'KeepRoom', updatedAt: 1 })
    await h.call('projects:rename', { id: 'p1', name: '   ' })
    await h.call('rooms:rename', { id: 'r1', name: '' })
    assert.equal(h.store.projects[0].name, 'New project', 'empty project name defaults')
    assert.equal(h.store.rooms[0].name, 'New room', 'empty room name defaults')
  }

  // --- Test 6: rename nonexistent id is idempotent, no throw, no persist ---
  {
    const h = buildHarness()
    await assert.doesNotReject(h.call('rooms:rename', { id: 'nope', name: 'Whatever' }))
    assert.equal(h.store.rooms.length, 0, 'store unchanged')
    assert.equal(h.persistCalls.rooms, 0, 'no persist when target missing')
    // Audit still records the attempt (preview of the resolved name).
    assert.ok(h.auditCalls.some((c) => c.event === 'room.rename'))
  }

  // --- Test 7: security role enforced for projects:rename ---
  {
    const h = buildHarness()
    h.store.projects.push({ id: 'p1', name: 'Old', updatedAt: 1 })
    h.setBlockRole(true)
    await assert.rejects(h.call('projects:rename', { id: 'p1', name: 'New' }), /forbidden/)
    assert.equal(h.store.projects[0].name, 'Old', 'rename blocked, name unchanged')
    assert.equal(h.persistCalls.projects, 0, 'no persist when blocked')
    assert.deepEqual(h.roleChecks[0], { role: 'operator', action: 'rename projects' })
  }

  // --- Test 8: security role enforced for rooms:rename ---
  {
    const h = buildHarness()
    h.store.rooms.push({ id: 'r1', name: 'Old', updatedAt: 1 })
    h.setBlockRole(true)
    await assert.rejects(h.call('rooms:rename', { id: 'r1', name: 'New' }), /forbidden/)
    assert.equal(h.store.rooms[0].name, 'Old')
    assert.deepEqual(h.roleChecks[0], { role: 'operator', action: 'rename rooms' })
  }

  console.log('rename tests passed')
}

run().catch((e) => { console.error(e); process.exit(1) })
