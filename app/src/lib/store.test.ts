// Run: jiti src/lib/store.test.ts
// Regression for: unguarded Promise.all in store.init() wedging the app on a blank
// screen. If any get* IPC call rejects, init() must still set ready:true (so App.tsx
// stops rendering null) and surface a notice — not throw and leave ready:false forever.
import assert from 'node:assert/strict'
import { applicationBootstrapState } from './bootstrap-state'

// Minimal browser shims so ./ipc and ./store (which read window/localStorage/matchMedia
// at module load) can be imported under plain node. window has no `vontra`, so ./ipc
// falls back to the mock api; we then override its methods below to force the failure path.
const g = globalThis as Record<string, unknown>
g.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} }
g.matchMedia = () => ({ matches: false, addEventListener: () => {}, removeEventListener: () => {} })
g.addEventListener = () => {}
g.removeEventListener = () => {}
g.window = g

// Loaded after the shims so module-init-time reads succeed.
await (await import('./browser-demo-bootstrap')).prepareBrowserDemo()
const { api } = await import('./ipc.ts')
const { useStore } = await import('./store.ts')

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('store.init tests:')

// Force every primary loader to reject — simulates a flaky/broken IPC bridge where the
// old code would throw out of Promise.all and never reach set({ ready: true }).
const boom = () => Promise.reject(new Error('ipc down'))
api.getSettings = boom as typeof api.getSettings
api.getThreads = boom as typeof api.getThreads
api.getProjects = boom as typeof api.getProjects
api.getRooms = boom as typeof api.getRooms
api.getTickets = boom as typeof api.getTickets
api.getJobs = boom as typeof api.getJobs
api.getAutomations = boom as typeof api.getAutomations
api.getUpdateStatus = boom as typeof api.getUpdateStatus
api.getArchives = boom as typeof api.getArchives

// The subscription registrations after the load must not blow up either — stub them.
// The on* APIs return an unsubscribe function, so the stub must too (returns a noop).
const noop = () => () => {}
let persistenceSubscribed = false
let persistenceSubscriptions = 0
let persistenceFailureListener: (failure: { label: string; message: string; at: number }) => void = () => { throw new Error('persistence listener was not registered') }
api.onAgentEvent = noop as typeof api.onAgentEvent
api.onProjectUpdate = noop as typeof api.onProjectUpdate
api.onThreadUpdate = noop as typeof api.onThreadUpdate
api.onRoomUpdate = noop as typeof api.onRoomUpdate
api.onRoomEvent = noop as typeof api.onRoomEvent
api.onTicketUpdate = noop as typeof api.onTicketUpdate
api.onJobUpdate = noop as typeof api.onJobUpdate
api.onAutomationUpdate = noop as typeof api.onAutomationUpdate
api.onUpdateStatus = noop as typeof api.onUpdateStatus
api.onPersistenceFailure = ((cb) => { persistenceSubscribed = true; persistenceSubscriptions++; persistenceFailureListener = cb; return () => {} }) as typeof api.onPersistenceFailure
api.listSshHosts = (() => Promise.reject(new Error('no ssh'))) as typeof api.listSshHosts
api.onSshStatus = noop as typeof api.onSshStatus
api.onSshSecretPrompt = noop as typeof api.onSshSecretPrompt
api.onSshTrustPrompt = noop as typeof api.onSshTrustPrompt

await (async () => {
  assert.equal(useStore.getState().ready, false, 'precondition: store starts not-ready')

  // The whole point: init() must resolve, not reject, even when every loader fails.
  await assert.doesNotReject(() => useStore.getState().init(), 'init() must not throw when IPC fails')

  const s = useStore.getState()
  assert.equal(s.ready, true, 'ready must be true so App.tsx renders instead of returning null')
  assert.ok(s.notice && s.notice.kind === 'error', 'a failure notice must be surfaced')
  // Defaults preserved (not undefined) so the rest of the UI does not crash on null arrays.
  assert.deepEqual(s.threads, [], 'threads falls back to default []')
  assert.deepEqual(s.projects, [], 'projects falls back to default []')
  assert.equal(s.settings, null, 'settings falls back to default null')
  assert.equal(applicationBootstrapState(s.ready, s.settings), 'settings-unavailable', 'a failed critical settings read must select the recoverable screen before mounting required-settings views')
  ok('init() with all-failing IPC still reaches ready:true and posts an error notice')
})()

assert.ok(persistenceSubscribed, 'init subscribes to background persistence failures')
api.getSettings = (async () => ({ providers: [], theme: 'system', approvalMode: 'suggest' } as any)) as typeof api.getSettings
await useStore.getState().init({ retry: true })
assert.equal(applicationBootstrapState(useStore.getState().ready, useStore.getState().settings), 'ready', 'retry reads the recovered authoritative settings')
assert.equal(persistenceSubscriptions, 1, 'retry must not duplicate IPC subscriptions')
useStore.setState({ notice: null })
persistenceFailureListener({ label: 'threads', message: 'Disk is full.', at: 42 })
assert.deepEqual(useStore.getState().notice, {
  kind: 'error',
  message: 'Local save failed for threads. Recent changes may not survive restart. Disk is full.',
})
ok('background persistence failure becomes a visible renderer notice')

// ---------------------------------------------------------------------------
// saveSettings: explicit saves confirm with a toast; side-effect ("silent")
// saves (launch / model switch) must NOT pop a misleading "Settings saved".
// ---------------------------------------------------------------------------
console.log('\nsaveSettings notice tests:')

const baseSettings = { providers: [], theme: 'system', approvalMode: 'suggest' } as any
api.saveSettings = (async (s: any) => s) as typeof api.saveSettings

await (async () => {
  useStore.setState({ notice: null })
  await useStore.getState().saveSettings(baseSettings)
  const n1 = useStore.getState().notice
  assert.ok(n1 && n1.kind === 'info' && n1.message === 'Settings saved', 'explicit save shows "Settings saved"')
  ok('explicit saveSettings posts a "Settings saved" notice')

  useStore.setState({ notice: null })
  await useStore.getState().saveSettings(baseSettings, { silent: true })
  assert.equal(useStore.getState().notice, null, 'silent save posts no notice')
  ok('silent saveSettings (launch / model switch) posts no notice')
})()

// ---------------------------------------------------------------------------
// Approval cleanup: terminal run states must clear pending approval rows so the
// composer and Command Centre do not keep showing stale actions after cancel/end.
// ---------------------------------------------------------------------------
console.log('\napproval cleanup tests:')

api.cancelRun = (async () => {}) as typeof api.cancelRun

await (async () => {
  useStore.setState({ status: { th1: 'awaiting' } as any, approvals: { th1: [{ callId: 'call-1', itemId: 'item-1' }] } })
  await useStore.getState().cancel('th1')
  assert.equal(useStore.getState().status.th1, 'awaiting')
  assert.equal(useStore.getState().approvals.th1?.length, 1, 'cancel acknowledgement preserves approval ownership until a terminal event')
  ok('cancel acknowledgement preserves pending work until the host settles')
})()

await (async () => {
  useStore.setState({ threads: [{ id: 'th1', title: 'Local', kind: 'task', projectId: null, cwd: '', status: 'running', apiMessages: [], timeline: [], createdAt: 1, updatedAt: 1 } as any], status: { th1: 'awaiting' } as any, approvals: { th1: [{ callId: 'call-2', itemId: 'item-2' }] } })
  useStore.getState().applyEvent({ runId: 'r1', threadId: 'th1', type: 'done' } as any)
  assert.equal(useStore.getState().status.th1, 'idle')
  assert.equal(useStore.getState().approvals.th1, undefined, 'done clears pending approvals for the thread')
  ok('done clears renderer approvals')
})()

await (async () => {
  useStore.setState({ threads: [{ id: 'th1', title: 'Local', kind: 'task', projectId: null, cwd: '', status: 'running', apiMessages: [], timeline: [], createdAt: 1, updatedAt: 1 } as any], status: { th1: 'awaiting' } as any, approvals: { th1: [{ callId: 'call-3', itemId: 'item-3' }] } })
  useStore.getState().applyEvent({ runId: 'r1', threadId: 'th1', type: 'error', message: 'boom' } as any)
  assert.equal(useStore.getState().status.th1, 'error')
  assert.equal(useStore.getState().approvals.th1, undefined, 'error clears pending approvals for the thread')
  ok('error clears renderer approvals')
})()

await (async () => {
  useStore.setState({ threads: [], status: {}, approvals: {} })
  useStore.getState().applyEvent({ runId: 'remote-run', threadId: 'remote-thread', type: 'status', status: 'thinking' } as any)
  useStore.getState().applyEvent({ runId: 'remote-run', threadId: 'remote-thread', type: 'item_add', item: { id: 'assistant-1', kind: 'assistant', text: '', createdAt: 1 } } as any)
  useStore.getState().applyEvent({ runId: 'remote-run', threadId: 'remote-thread', type: 'item_delta', id: 'assistant-1', field: 'text', chunk: 'streamed from orchestrator' } as any)
  assert.deepEqual(useStore.getState().threads, [], 'agent timeline events are buffered until their thread arrives')
  assert.equal(useStore.getState().status['remote-thread'], undefined, 'threadless status is quarantined until its origin-stamped thread arrives')

  useStore.getState().applyThreadUpdate({
    id: 'remote-thread',
    title: 'Remote task',
    kind: 'task',
    projectId: null,
    cwd: '',
    createdAt: 1,
    updatedAt: 1,
    status: 'idle',
    apiMessages: [],
    timeline: [],
  } as any)

  const thread = useStore.getState().threads.find((candidate) => candidate.id === 'remote-thread')
  assert.equal(thread?.timeline.length, 1)
  assert.equal(thread?.timeline[0].text, 'streamed from orchestrator')
  assert.equal(useStore.getState().status['remote-thread'], 'thinking')
  ok('remote agent events buffer until thread:update, then replay into the timeline')
})()

// Engine selection must restore the saved value when the host rejects a switch,
// while preserving thread updates that arrive during the request.
await (async () => {
  const original = api.setThreadEngine
  const seed = (engine: 'omp' | undefined) => useStore.setState({ threads: [{
    id: 'engine-thread', title: 'Original title', kind: 'task', projectId: null,
    cwd: '/repo', status: 'idle', apiMessages: [], timeline: [], createdAt: 1, updatedAt: 1, engine,
  } as any] })
  try {
    for (const [before, requested] of [[undefined, 'omp'], ['omp', 'native']] as const) {
      seed(before)
      let reject!: (error: Error) => void
      api.setThreadEngine = (async () => new Promise<void>((_resolve, fail) => { reject = fail })) as typeof api.setThreadEngine
      const pending = useStore.getState().setThreadEngine('engine-thread', requested)
      assert.equal(useStore.getState().threads[0].engine, requested === 'native' ? undefined : 'omp', 'the selection changes optimistically')
      useStore.setState((state) => ({ threads: state.threads.map((thread) => ({ ...thread, title: 'Host updated title' })) }))
      reject(new Error('Host rejected engine switch'))
      await assert.rejects(pending, /Host rejected engine switch/)
      assert.equal(useStore.getState().threads[0].engine, before, 'a rejected switch restores the prior engine in either direction')
      assert.equal(useStore.getState().threads[0].title, 'Host updated title', 'rollback preserves intervening non-engine updates')
    }

    seed(undefined)
    api.setThreadEngine = (async () => {}) as typeof api.setThreadEngine
    await useStore.getState().setThreadEngine('engine-thread', 'omp')
    assert.equal(useStore.getState().threads[0].engine, 'omp', 'an accepted switch keeps the chosen engine')

    seed('omp')
    let reject!: (error: Error) => void
    api.setThreadEngine = (async () => new Promise<void>((_resolve, fail) => { reject = fail })) as typeof api.setThreadEngine
    const pending = useStore.getState().setThreadEngine('engine-thread', 'native')
    useStore.setState((state) => ({ threads: state.threads.map((thread) => ({ ...thread, engine: 'omp' })) }))
    reject(new Error('Host rejected engine switch'))
    await assert.rejects(pending, /Host rejected engine switch/)
    assert.equal(useStore.getState().threads[0].engine, 'omp', 'rollback does not replace a newer host engine value')
    ok('engine selection rolls back rejected requests without losing host updates')
  } finally {
    api.setThreadEngine = original
  }
})()

// ---------------------------------------------------------------------------
// Project task startup: project-scoped tasks must inherit the project folder /
// remote workspace immediately, and a startup rejection must not leave the UI
// stuck in "Thinking…" with only Stop available.
// ---------------------------------------------------------------------------
console.log('\nproject task startup tests:')

const provider = { id: 'local', name: 'Local', baseUrl: 'http://127.0.0.1:11434/v1', apiKey: '', models: ['m'], kind: 'local' } as any
const settings = { providers: [provider], activeProviderId: 'local', activeModel: 'm', defaultCwd: '/default', approvalMode: 'suggest', theme: 'system' } as any
api.configureThread = (async () => {}) as typeof api.configureThread

await (async () => {
  const workspace = { kind: 'ssh', hostId: 'pgx', path: '/srv/project' } as const
  const patches: any[] = []
  useStore.setState({
    settings,
    projects: [{ id: 'proj1', name: 'Project', cwd: '', workspace, createdAt: 0, updatedAt: 0 } as any],
    threads: [],
    view: { name: 'home' },
    status: {},
  })
  api.createThread = (async (_cwd: string, kind: 'task' | 'chat' = 'task', projectId: string | null = null) => ({
    id: 'thread-project',
    title: 'New task',
    kind,
    projectId,
    cwd: '',
    workspace: null,
    providerId: null,
    providerAccountId: null,
    model: null,
    createdAt: 0,
    updatedAt: 0,
    status: 'idle',
    apiMessages: [],
    timeline: [],
  } as any)) as typeof api.createThread
  api.setThreadWorkspace = (async (_id: string, nextWorkspace: any, cwd: any) => {
    patches.push({ cwd, workspace: nextWorkspace })
    return {} as any
  }) as typeof api.setThreadWorkspace

  await useStore.getState().newThread('proj1')
  const thread = useStore.getState().threads[0] as any
  assert.equal(thread.cwd, '/srv/project')
  assert.deepEqual(thread.workspace, workspace)
  assert.deepEqual(patches[0], { cwd: '/srv/project', workspace })
  assert.deepEqual(useStore.getState().view, { name: 'thread', id: 'thread-project' })
  ok('new project task inherits project workspace when IPC returns an empty cwd')
})()

await (async () => {
  useStore.setState({
    threads: [{ id: 'th-start', title: 'Task', cwd: '', kind: 'task', projectId: null, timeline: [], createdAt: 0, updatedAt: 0, status: 'idle', apiMessages: [] } as any],
    view: { name: 'thread', id: 'th-start' },
    status: { 'th-start': 'idle' } as any,
    notice: null,
  })
  api.startRun = (async () => { throw new Error('Choose a workspace folder before running workspace tools.') }) as typeof api.startRun
  await useStore.getState().send('do it')
  assert.equal(useStore.getState().status['th-start'], 'error')
  assert.match(useStore.getState().notice?.message ?? '', /Choose a workspace folder/)
  ok('send startup failure clears Thinking state and surfaces the error')
})()

await (async () => {
  const launchProvider = { ...provider, models: ['m', 'assistant-m'] }
  const savedSettings: any[] = []
  const configured: any[] = []
  useStore.setState({
    settings: { ...settings, providers: [launchProvider], activeProviderId: 'local', activeModel: 'm', defaultCwd: '/default' },
    projects: [],
    threads: [],
    status: {},
    view: { name: 'home' },
    notice: null,
  })
  api.saveSettings = (async (next: any) => { savedSettings.push(next); return next }) as typeof api.saveSettings
  api.createThread = (async (cwd: string, kind: 'task' | 'chat' = 'task', projectId: string | null = null) => ({
    id: 'thread-assistant-launch',
    title: 'New task',
    kind,
    projectId,
    cwd,
    workspace: null,
    providerId: null,
    providerAccountId: null,
    model: null,
    createdAt: 0,
    updatedAt: 0,
    status: 'idle',
    apiMessages: [],
    timeline: [],
  } as any)) as typeof api.createThread
  api.configureThread = (async (id: string, providerId: string, model: string, providerAccountId: string | null = null) => {
    configured.push({ id, providerId, model, providerAccountId })
  }) as typeof api.configureThread
  api.startRun = (async () => ({ runId: 'run-assistant-launch' })) as typeof api.startRun

  await useStore.getState().launch({
    text: 'Draft a reply',
    cwd: '/assistant',
    providerId: 'local',
    model: 'assistant-m',
    approvalMode: 'suggest',
    persistActiveModel: false,
  })

  assert.equal(savedSettings.length, 0, 'Assistant launch does not persist over the global active model')
  assert.deepEqual(configured[0], { id: 'thread-assistant-launch', providerId: 'local', model: 'assistant-m', providerAccountId: null })
  assert.equal(useStore.getState().settings?.activeModel, 'm')
  ok('launch({ persistActiveModel:false }) isolates Assistant model selection from global settings')
  useStore.setState({ view: { name: 'assistant' } })
  api.startRun = (async () => { throw new Error('Example model startup failed') }) as typeof api.startRun
  const rejected: any = await useStore.getState().launch({ text: 'Preserve this unsent draft', cwd: '/example/project', providerId: 'local', model: 'assistant-m', approvalMode: 'suggest', persistActiveModel: false })
  assert.equal(rejected?.ok, false, 'the composer must know when run admission failed')
  assert.match(rejected.error, /startup failed/)
  assert.deepEqual(useStore.getState().view, { name: 'assistant' }, 'failed startup must not unmount the assistant and discard its unsent draft')
  ok('launch rejects startup without navigating away from the draft')
})()

await (async () => {
  // Standard sidebar/palette New task + New chat must use the configured surface routes too;
  // previously only the home composer did, while these actions silently fell back to active*.
  const code = { ...provider, id: 'code-provider', models: ['code-model'] }
  const chat = { ...provider, id: 'chat-provider', models: ['chat-model'] }
  const configured: Array<{ id: string; providerId: string; model: string; providerAccountId: string | null }> = []
  useStore.setState({
    settings: {
      ...settings,
      providers: [provider, code, chat],
      codeProviderId: code.id,
      codeProviderAccountId: null,
      codeModel: 'code-model',
      assistantProviderId: chat.id,
      assistantProviderAccountId: null,
      assistantModel: 'chat-model',
    },
    projects: [],
    threads: [],
    view: { name: 'home' },
  })
  api.createThread = (async (cwd: string, kind: 'task' | 'chat' = 'task', projectId: string | null = null) => ({
    id: `surface-${kind}`, title: kind, kind, projectId, cwd, workspace: null,
    providerId: null, providerAccountId: null, model: null, createdAt: 0, updatedAt: 0,
    status: 'idle', apiMessages: [], timeline: [],
  } as any)) as typeof api.createThread
  api.configureThread = (async (id: string, providerId: string, model: string, providerAccountId: string | null = null) => {
    configured.push({ id, providerId, model, providerAccountId })
  }) as typeof api.configureThread

  await useStore.getState().newThread(null)
  await useStore.getState().newChat(null)

  assert.deepEqual(configured, [
    { id: 'surface-task', providerId: 'code-provider', model: 'code-model', providerAccountId: null },
    { id: 'surface-chat', providerId: 'chat-provider', model: 'chat-model', providerAccountId: null },
  ])
  assert.equal(useStore.getState().threads.find((thread) => thread.id === 'surface-task')?.model, 'code-model')
  assert.equal(useStore.getState().threads.find((thread) => thread.id === 'surface-chat')?.model, 'chat-model')
  ok('standard New task/New chat actions honor independent per-surface defaults')
})()

// ---------------------------------------------------------------------------
// Kanban room dispatch: room-mode tickets should be sendable to an existing
// project room. This must attach the ticket to that room before starting it,
// rather than always creating a fresh room.
// ---------------------------------------------------------------------------
console.log('\nkanban room dispatch tests:')

await (async () => {
  const room = {
    id: 'room-existing',
    projectId: 'proj-room',
    name: 'Build room',
    seats: [{ id: 'seat-1', name: 'Builder', role: 'Build', lane: 'development', providerId: 'local', providerAccountId: null, model: 'm' }],
    problem: '',
    state: 'ready',
    events: [],
    finalText: '',
    createdAt: 0,
    updatedAt: 0,
  } as any
  const ticket = {
    id: 'ticket-room',
    projectId: 'proj-room',
    roomId: null,
    executionMode: 'room',
    title: 'Patch the board dispatch',
    body: 'Send this to a room.',
    type: 'task',
    status: 'todo',
    priority: 'medium',
    effort: 'medium',
    tags: [],
    parentId: null,
    assignee: null,
    threadId: null,
    runState: null,
    order: 1,
    createdAt: 0,
    updatedAt: 0,
  } as any
  let createdRoom = false
  let updateRoomArgs: any = null
  let updateTicketArgs: any = null
  let startRoomArgs: any = null
  useStore.setState({
    settings,
    projects: [{ id: 'proj-room', name: 'Project', cwd: '/repo', createdAt: 0, updatedAt: 0 } as any],
    rooms: [room],
    tickets: [ticket],
    roomStatus: {},
    roomLive: {},
    view: { name: 'project', id: 'proj-room', tab: 'board' },
    notice: null,
  })
  api.createRoom = (async () => { createdRoom = true; throw new Error('should not create a room') }) as typeof api.createRoom
  api.updateRoom = (async (id: string, patch: any) => { updateRoomArgs = { id, patch }; return { ...room, ...patch, updatedAt: 1 } }) as typeof api.updateRoom
  api.updateTicket = (async (id: string, patch: any) => { updateTicketArgs = { id, patch }; return { ...ticket, ...patch, updatedAt: 2 } }) as typeof api.updateTicket
  api.startRoom = (async (roomId: string, problem: string) => { startRoomArgs = { roomId, problem }; return { ok: true, room: { ...room, problem, state: 'queued' } } }) as typeof api.startRoom

  await useStore.getState().startRoomFromTicket('ticket-room', 'room-existing')

  assert.equal(createdRoom, false, 'existing room dispatch must not create a new room')
  assert.equal(updateRoomArgs, null, 'dispatch does not route runtime problem/state through the room config patch API')
  assert.deepEqual(updateTicketArgs, { id: 'ticket-room', patch: { roomId: 'room-existing', executionMode: 'room', status: 'ready' } })
  assert.equal(startRoomArgs.roomId, 'room-existing')
  assert.match(startRoomArgs.problem, /Kanban ticket: Patch the board dispatch/)
  assert.equal(useStore.getState().tickets[0].roomId, 'room-existing')
  assert.deepEqual(useStore.getState().view, { name: 'room', id: 'room-existing' })
  ok('room-mode ticket can be dispatched to an existing project room')
})()

await (async () => {
  let room = {
    id: 'room-running',
    projectId: 'proj-room',
    name: 'Running room',
    seats: [{ id: 'seat-1', name: 'Reviewer', role: 'Review', lane: 'review', roleProfileId: 'reviewer', providerId: 'local', providerAccountId: null, model: 'm' }],
    problem: 'Existing run',
    state: 'running',
    events: [{ id: 'event-1', tick: 1, kind: 'task', seat: 'you', target: 'all', text: 'Existing run', createdAt: 1 }],
    finalText: '',
    createdAt: 0,
    updatedAt: 0,
  } as any
  const ticket = {
    id: 'ticket-running-room',
    projectId: 'proj-room',
    roomId: null,
    executionMode: 'room',
    title: 'Review the active plan',
    body: 'Send this to the running room.',
    type: 'task',
    status: 'ready',
    priority: 'high',
    effort: 'small',
    tags: [],
    parentId: null,
    assignee: null,
    threadId: null,
    runState: null,
    order: 1,
    createdAt: 0,
    updatedAt: 0,
  } as any
  let started = false
  const roomPatches: any[] = []
  let steerArgs: any = null
  useStore.setState({
    settings,
    projects: [{ id: 'proj-room', name: 'Project', cwd: '/repo', createdAt: 0, updatedAt: 0 } as any],
    rooms: [room],
    tickets: [ticket],
    roomStatus: { 'room-running': 'running' },
    roomLive: {},
    view: { name: 'project', id: 'proj-room', tab: 'board' },
    notice: null,
  })
  api.createRoom = (async () => { throw new Error('should not create a room') }) as typeof api.createRoom
  api.updateRoom = (async (id: string, patch: any) => {
    roomPatches.push({ id, patch })
    room = { ...room, ...patch, updatedAt: room.updatedAt + 1 }
    return room
  }) as typeof api.updateRoom
  api.updateTicket = (async (id: string, patch: any) => ({ ...ticket, ...patch, updatedAt: 2 })) as typeof api.updateTicket
  api.startRoom = (async () => { started = true; throw new Error('should steer instead of restart') }) as typeof api.startRoom
  api.steerRoom = (async (roomId: string, text: string) => {
    steerArgs = { roomId, text }
    return { id: 'steer-1', tick: 2, kind: 'user', seat: 'you', target: 'all', text, createdAt: 2 }
  }) as typeof api.steerRoom

  await useStore.getState().startRoomFromTicket('ticket-running-room', 'room-running')

  assert.equal(started, false, 'running room dispatch must not restart the room')
  assert.deepEqual(roomPatches, [], 'running room dispatch does not mutate runtime fields through the config patch API')
  assert.equal(steerArgs.roomId, 'room-running')
  assert.match(steerArgs.text, /New Kanban ticket sent to this room/)
  assert.equal(useStore.getState().rooms[0].events.at(-1)?.kind, 'user')
  ok('room-mode ticket steers an already-running room instead of restarting it')
})()

// ---------------------------------------------------------------------------
// applyEvent app-control: a create that REJECTS must not be marked-applied in
// localStorage (so it can retry on reload) and must surface an error notice; a
// create that RESOLVES must persist the applied id and flag the timeline item.
// ---------------------------------------------------------------------------
console.log('\napplyEvent app-control tests:')

// Real in-memory localStorage so we can observe persisted applied-action ids.
const lsStore = new Map<string, string>()
g.localStorage = {
  getItem: (k: string) => (lsStore.has(k) ? lsStore.get(k)! : null),
  setItem: (k: string, v: string) => { lsStore.set(k, String(v)) },
  removeItem: (k: string) => { lsStore.delete(k) },
}

const APPLIED_KEY = 'vontra.appliedAppActions'
const persistedIds = () => { try { return JSON.parse(lsStore.get(APPLIED_KEY) || '[]') as string[] } catch { return [] } }

// One project so create_room/create_kanban can resolve a target project id.
useStore.setState({ projects: [{ id: 'proj1', name: 'P', cwd: '' } as any], view: { name: 'home' } })

const mkEvent = (itemId: string) => ({
  runId: 'r1', threadId: 'th1', type: 'item_add' as const,
  item: { id: itemId, role: 'assistant', kind: 'app_action', appAction: { kind: 'create_room' as const, payload: { name: 'Room' } } } as any,
})

const seedThread = (itemId: string) => useStore.setState({
  threads: [{ id: 'th1', title: 't', cwd: '', kind: 'task', projectId: null, timeline: [{ id: itemId, role: 'assistant', kind: 'app_action', appAction: { kind: 'create_room', payload: { name: 'Room' } } }], createdAt: 0, updatedAt: 0 } as any],
})

// (a) failing create -> NOT persisted, error notice surfaced.
await (async () => {
  lsStore.clear()
  useStore.setState({ notice: null })
  seedThread('act-fail')
  api.createRoom = (async () => { throw new Error('room create boom') }) as typeof api.createRoom
  useStore.getState().applyEvent(mkEvent('act-fail') as any)
  // let the rejected promise settle
  await new Promise((r) => setTimeout(r, 0))
  assert.ok(!persistedIds().includes('act-fail'), 'failed create must NOT be persisted as applied (so it can retry)')
  const n = useStore.getState().notice
  assert.ok(n && n.kind === 'error', 'failed create surfaces an error notice')
  const item = useStore.getState().threads[0].timeline.find((i: any) => i.id === 'act-fail') as any
  assert.notEqual(item?.appAction?.state, 'applied', 'failed create must not flag the timeline item applied')
  ok('failed app-control create stays unmarked + surfaces an error (retryable)')
})()

// (b) succeeding create -> persisted + timeline item flagged applied.
await (async () => {
  lsStore.clear()
  useStore.setState({ notice: null })
  seedThread('act-ok')
  api.createRoom = (async (projectId: string, name: string) => ({ id: 'rm1', projectId, name } as any)) as typeof api.createRoom
  // createRoom store action updates rooms; make sure it doesn't depend on more api.
  useStore.getState().applyEvent(mkEvent('act-ok') as any)
  await new Promise((r) => setTimeout(r, 0))
  assert.ok(persistedIds().includes('act-ok'), 'successful create persists the applied id')
  const item = useStore.getState().threads[0].timeline.find((i: any) => i.id === 'act-ok') as any
  assert.equal(item?.appAction?.state, 'applied', 'successful create flags the timeline item applied')
  ok('successful app-control create persists applied id + flags item')
})()

// ---------------------------------------------------------------------------
// loadTerminals auto-restore: a custom CLI's persisted `command` is an arbitrary string
// typed character-for-character into a real login shell on restore — only known built-ins
// may be auto-restored without confirmation. A session with no `builtin` flag at all
// (older persisted state) must be treated the same as a custom CLI: not auto-restored.
// ---------------------------------------------------------------------------
console.log('\nterminal auto-restore tests:')

await (async () => {
  const restoredIds: string[] = []
  api.listTerminals = (async () => []) as typeof api.listTerminals
  api.restorableTerminals = (async () => ([
    { id: 'sess-codex', projectId: 'proj-restore-test', cliId: 'codex', title: 'Codex', status: 'idle', createdAt: 1, builtin: true },
    { id: 'sess-custom', projectId: 'proj-restore-test', cliId: 'custom-1', title: 'my tool', status: 'idle', createdAt: 1, builtin: false, command: 'echo hi; rm -rf /tmp/x' },
    { id: 'sess-legacy', projectId: 'proj-restore-test', cliId: 'pi', title: 'pi', status: 'idle', createdAt: 1 }, // no builtin field (pre-existing persisted state)
  ])) as typeof api.restorableTerminals
  api.restoreTerminal = (async (id: string) => { restoredIds.push(id); return {} as any }) as typeof api.restoreTerminal

  await useStore.getState().loadTerminals('proj-restore-test')
  await new Promise((r) => setTimeout(r, 0)) // let the fire-and-forget restoreTerminal calls dispatch

  assert.deepEqual(restoredIds, ['sess-codex'], 'only the known-builtin session auto-restores; custom and builtin-less sessions do not')
  ok('loadTerminals auto-restores built-ins only, never a custom CLI command or a pre-existing session with no builtin flag')
})()

await (async () => {
  const projectId = 'proj-remote-terminal'
  const remoteSession = { id: 'sess-remote', projectId, cliId: 'shell', title: 'Shell', status: 'idle', createdAt: 1, hostLocation: 'orchestrator', managedBy: 'local' } as any
  const calls: string[] = []

  useStore.setState({
    settings: { ...settings, remoteOrchestrators: [{ id: 'mini', name: 'Mini', enabled: true }] } as any,
    projects: [{ id: projectId, name: 'Mini project', cwd: '/srv/app', executionTarget: 'orchestrator', orchestratorId: 'mini', createdAt: 0, updatedAt: 0 } as any],
    terminals: {},
    activeTerminalId: {},
  })
  api.listTerminals = (async () => { throw new Error('local list should not be called') }) as typeof api.listTerminals
  api.listClis = (async () => { throw new Error('local CLI catalog should not be called') }) as typeof api.listClis
  api.startTerminal = (async () => { throw new Error('local start should not be called') }) as typeof api.startTerminal
  api.killTerminal = (async () => { throw new Error('local kill should not be called') }) as typeof api.killTerminal
  api.renameTerminal = (async () => { throw new Error('local rename should not be called') }) as typeof api.renameTerminal
  api.remoteListClis = (async (hostId: string | null | undefined) => {
    calls.push(`clis:${hostId}`)
    return [{ id: 'codex', name: 'Codex', short: 'codex', command: 'codex', resumeCommand: 'codex resume --last', color: '#2FB9B0', icon: 'code', builtin: true } as any]
  }) as typeof api.remoteListClis
  api.remoteTerminalList = (async (hostId: string | null | undefined, pid: string) => {
    calls.push(`list:${hostId}:${pid}`)
    return [remoteSession]
  }) as typeof api.remoteTerminalList
  api.remoteTerminalStart = (async (hostId: string | null | undefined, input: any) => {
    calls.push(`start:${hostId}:${input.projectId}:${input.cliId}`)
    return { ...remoteSession, id: 'sess-started', cliId: input.cliId }
  }) as typeof api.remoteTerminalStart
  api.remoteTerminalKill = (async (hostId: string | null | undefined, sessionId: string) => {
    calls.push(`kill:${hostId}:${sessionId}`)
  }) as typeof api.remoteTerminalKill
  api.remoteTerminalRename = (async (hostId: string | null | undefined, sessionId: string, title: string) => {
    calls.push(`rename:${hostId}:${sessionId}:${title}`)
  }) as typeof api.remoteTerminalRename
  api.approve = (async () => { throw new Error('local approval should not be called') }) as typeof api.approve
  api.approveRemoteOrchestrator = (async (hostId: string | null | undefined, callId: string, decision: 'allow' | 'allow_always' | 'deny') => {
    calls.push(`approve:${hostId}:${callId}:${decision}`)
  }) as typeof api.approveRemoteOrchestrator
  api.deleteTicket = (async () => { throw new Error('local ticket delete should not be called') }) as typeof api.deleteTicket
  api.deleteRemoteOrchestratorTicket = (async (hostId: string | null | undefined, ticketId: string) => {
    calls.push(`delete-ticket:${hostId}:${ticketId}`)
    return { id: ticketId, projectId, archivedAt: Date.now(), orchestratorHostId: hostId } as any
  }) as typeof api.deleteRemoteOrchestratorTicket

  const clis = await useStore.getState().listTerminalClis(projectId)
  assert.deepEqual(clis.map((cli) => cli.id), ['codex'])

  await useStore.getState().loadTerminals(projectId)
  assert.equal(useStore.getState().terminals[projectId]?.[0]?.id, 'sess-remote')

  const started = await useStore.getState().startTerminal(projectId, 'shell')
  assert.equal(started?.id, 'sess-started')
  assert.equal(useStore.getState().activeTerminalId[projectId], 'orchestrator:mini:sess-started')

  const custom = await useStore.getState().startTerminal(projectId, 'custom:gemini --yolo', { command: 'gemini --yolo', label: 'Gemini custom' })
  assert.equal(custom, null, 'remote custom terminal commands are blocked before calling the Orchestrator')
  assert.equal(useStore.getState().activeTerminalId[projectId], 'orchestrator:mini:sess-started')

  await useStore.getState().renameTerminal(projectId, 'sess-started', '  Remote shell  ')
  assert.equal(useStore.getState().terminals[projectId]?.find((session) => session.id === 'sess-started')?.title, 'Remote shell')

  await useStore.getState().killTerminal(projectId, 'sess-started')
  useStore.setState({
    threads: [{ id: 'thread-remote-approval', title: 'Remote approval', kind: 'task', projectId, cwd: '/srv/app', workspace: null, orchestratorHostId: 'mini', providerId: null, providerAccountId: null, model: null, createdAt: 0, updatedAt: 0, status: 'running', apiMessages: [], timeline: [] } as any],
    approvals: { 'thread-remote-approval': [{ callId: 'call-remote', itemId: 'item-1' }] },
    tickets: [{ id: 'ticket-remote-delete', projectId, orchestratorHostId: 'mini', title: 'Remote ticket', body: '', type: 'task', status: 'todo', priority: 'medium', effort: 'small', tags: [], parentId: null, assignee: null, threadId: null, runState: null, order: 1, createdAt: 0, updatedAt: 0 } as any],
  })
  await useStore.getState().approve('thread-remote-approval', 'call-remote', 'allow')
  assert.deepEqual(useStore.getState().approvals['thread-remote-approval'], [])
  await useStore.getState().deleteTicket('ticket-remote-delete')
  assert.equal(useStore.getState().tickets.some((ticket) => ticket.id === 'ticket-remote-delete'), false)
  assert.ok(calls.includes('clis:mini'), 'remote CLI catalog uses the selected Orchestrator')
  assert.ok(calls.includes(`list:mini:${projectId}`), 'remote list uses the selected Orchestrator')
  assert.ok(calls.includes(`start:mini:${projectId}:shell`), 'remote start uses the selected Orchestrator')
  assert.ok(!calls.some((call) => call.includes('custom')), 'remote custom CLI start is never forwarded to the Orchestrator')
  assert.ok(calls.includes('rename:mini:sess-started:Remote shell'), 'remote rename uses the selected Orchestrator')
  assert.ok(calls.includes('kill:mini:sess-started'), 'remote kill uses the selected Orchestrator')
  assert.ok(calls.includes('approve:mini:call-remote:allow'), 'remote approvals resolve on the selected Orchestrator')
  assert.ok(calls.includes('delete-ticket:mini:ticket-remote-delete'), 'remote ticket deletion uses the selected Orchestrator')
  ok('Orchestrator project terminal, approval, and ticket actions route through remote APIs')
})()

// ---------------------------------------------------------------------------
// Remote project/room lifecycle: client-side project proxies must forward edits
// and removals to the selected Orchestrator host before updating local state.
// ---------------------------------------------------------------------------
console.log('\nremote lifecycle routing tests:')

await (async () => {
  useStore.setState({
    settings: { ...settings, remoteOrchestrators: [] } as any,
    projects: [],
    newProjectOpen: true,
    view: { name: 'home' },
  })
  api.createProject = (async (project: any) => ({ id: 'proj-tagged-local', ...project, createdAt: project.createdAt ?? 0, updatedAt: project.updatedAt ?? 1 })) as typeof api.createProject

  await useStore.getState().newProject({ name: 'Tagged local', cwd: '/repo/local', mode: 'solo', tags: ['work', 'client a'] })

  assert.deepEqual(useStore.getState().projects[0]?.tags, ['work', 'client a'], 'local project creation persists selected tags')
  ok('new local project preserves creation tags')
})()

await (async () => {
  const projectId = 'proj-remote-new'
  useStore.setState({
    settings: { ...settings, remoteOrchestrators: [{ id: 'mini', name: 'Mini', enabled: true }] } as any,
    projects: [],
    newProjectOpen: true,
    view: { name: 'home' },
  })
  api.createRemoteOrchestratorProject = (async (hostId: string | null | undefined, project: any) => ({
    id: projectId,
    name: project.name,
    cwd: project.cwd,
    tags: project.tags,
    executionTarget: 'orchestrator',
    orchestratorId: hostId,
    createdAt: Date.now(),
    updatedAt: Date.now(),
  } as any)) as typeof api.createRemoteOrchestratorProject
  api.createProject = (async (project: any) => ({ ...project, createdAt: project.createdAt ?? Date.now(), updatedAt: Date.now() })) as typeof api.createProject

  await useStore.getState().newProject({ name: 'Remote new', cwd: '/srv/new', mode: 'solo', executionTarget: 'orchestrator', orchestratorId: 'mini', tags: ['personal'] })

  assert.deepEqual(useStore.getState().view, { name: 'project', id: projectId, tab: 'terminals' }, 'remote projects land on the terminal tab for immediate CLI launch')
  assert.deepEqual(useStore.getState().projects[0]?.tags, ['personal'], 'remote project proxy preserves selected tags')
  ok('new Orchestrator project opens directly to the Terminal tab')
})()

await (async () => {
  useStore.setState({
    settings: { ...settings, remoteOrchestrators: [] } as any,
    projects: [],
    newProjectOpen: true,
    view: { name: 'home' },
  })
  api.createRemoteOrchestratorProject = (async () => { throw new Error('remote create should not be called without a host') }) as typeof api.createRemoteOrchestratorProject

  await assert.rejects(
    () => useStore.getState().newProject({ name: 'No host', cwd: '/srv/new', mode: 'solo', executionTarget: 'orchestrator' }),
    /Choose an Orchestrator/,
  )

  useStore.setState({
    settings: { ...settings, remoteOrchestrators: [{ id: 'mini', name: 'Mini', enabled: true }] } as any,
  })
  await assert.rejects(
    () => useStore.getState().newProject({ name: 'No folder', cwd: '', mode: 'solo', executionTarget: 'orchestrator', orchestratorId: 'mini' }),
    /Choose a folder/,
  )
  ok('remote project creation blocks missing Orchestrator and missing host folder')
})()

await (async () => {
  const calls: string[] = []
  useStore.setState({
    settings: {
      ...settings,
      remoteOrchestrators: [
        { id: 'disabled', name: 'Disabled', enabled: false },
        { id: 'mini', name: 'Mini', enabled: true },
      ],
    } as any,
    projects: [],
    newProjectOpen: true,
    view: { name: 'home' },
  })
  api.createRemoteOrchestratorProject = (async (hostId: string | null | undefined, project: any) => {
    calls.push(`remote:create:${hostId}:${project.cwd}`)
    return { id: 'proj-default-host', ...project, executionTarget: 'orchestrator', orchestratorId: hostId, createdAt: 0, updatedAt: 1 } as any
  }) as typeof api.createRemoteOrchestratorProject
  api.createProject = (async (project: any) => ({ ...project, createdAt: project.createdAt ?? 0, updatedAt: project.updatedAt ?? 1 })) as typeof api.createProject

  await useStore.getState().newProject({ name: 'Default host', cwd: '/srv/default', mode: 'solo', executionTarget: 'orchestrator' })

  assert.ok(calls.includes('remote:create:mini:/srv/default'), 'remote creation falls back to first enabled Orchestrator')
  assert.equal(useStore.getState().projects[0]?.orchestratorId, 'mini')
  ok('remote project creation defaults to the first enabled Orchestrator')
})()

await (async () => {
  const projectId = 'proj-remote-life'
  const roomId = 'room-remote-life'
  const calls: string[] = []
  const localProxyPatches: Array<{ kind: 'project' | 'room'; patch: any }> = []
  useStore.setState({
    settings: { ...settings, remoteOrchestrators: [{ id: 'mini', name: 'Mini', enabled: true }] } as any,
    projects: [{ id: projectId, name: 'Remote', cwd: '/srv/app', executionTarget: 'orchestrator', orchestratorId: 'mini', createdAt: 0, updatedAt: 0 } as any],
    rooms: [{ id: roomId, projectId, orchestratorHostId: 'mini', name: 'Remote room', seats: [], state: 'created', events: [], createdAt: 0, updatedAt: 0 } as any],
    roomStatus: { [roomId]: 'running' } as any,
    roomLive: { [roomId]: { seat: 'Builder', text: 'Working' } },
  })
  api.updateProject = (async (id: string, patch: any) => {
    localProxyPatches.push({ kind: 'project', patch })
    return { ...useStore.getState().projects.find((project) => project.id === id), ...patch, updatedAt: Date.now() }
  }) as typeof api.updateProject
  api.setProjectExecutionTarget = (async (id: string, executionTarget: any, orchestratorId?: string | null, configPatch: any = {}) => {
    calls.push(`project:route:${id}:${executionTarget}:${orchestratorId ?? ''}`)
    localProxyPatches.push({ kind: 'project', patch: configPatch })
    return { ...useStore.getState().projects.find((project) => project.id === id), ...configPatch, executionTarget, orchestratorId: executionTarget === 'orchestrator' ? orchestratorId ?? null : null, updatedAt: Date.now() } as any
  }) as typeof api.setProjectExecutionTarget
  api.deleteProject = (async (_id: string) => {}) as typeof api.deleteProject
  api.updateRoom = (async (_id: string, patch: any) => {
    localProxyPatches.push({ kind: 'room', patch })
    return { ...useStore.getState().rooms.find((room) => room.id === _id), ...patch, updatedAt: Date.now() }
  }) as typeof api.updateRoom
  api.renameRoom = (async () => { throw new Error('local rename should not be called') }) as typeof api.renameRoom
  api.deleteRoom = (async (_id: string) => {}) as typeof api.deleteRoom
  api.updateRemoteOrchestratorProject = (async (hostId: string | null | undefined, id: string, patch: any) => {
    calls.push(`project:update:${hostId}:${id}:${patch.name ?? ''}`)
    return { ...useStore.getState().projects.find((project) => project.id === id), ...patch, executionTarget: 'orchestrator', orchestratorId: hostId, updatedAt: Date.now() } as any
  }) as typeof api.updateRemoteOrchestratorProject
  api.deleteRemoteOrchestratorProject = (async (hostId: string | null | undefined, id: string) => {
    calls.push(`project:delete:${hostId}:${id}`)
    return { id, archivedAt: Date.now() } as any
  }) as typeof api.deleteRemoteOrchestratorProject
  api.updateRemoteOrchestratorRoom = (async (hostId: string | null | undefined, id: string, patch: any) => {
    calls.push(`room:update:${hostId}:${id}:${patch.name ?? patch.seats?.[0]?.name ?? ''}`)
    return { ...useStore.getState().rooms.find((room) => room.id === id), ...patch, updatedAt: Date.now() } as any
  }) as typeof api.updateRemoteOrchestratorRoom
  api.renameRemoteOrchestratorRoom = (async (hostId: string | null | undefined, id: string, name: string) => {
    calls.push(`room:rename:${hostId}:${id}:${name}`)
    return { ...useStore.getState().rooms.find((room) => room.id === id), name, updatedAt: Date.now() } as any
  }) as typeof api.renameRemoteOrchestratorRoom
  api.deleteRemoteOrchestratorRoom = (async (hostId: string | null | undefined, id: string) => {
    calls.push(`room:delete:${hostId}:${id}`)
    return { id, projectId, archivedAt: Date.now() } as any
  }) as typeof api.deleteRemoteOrchestratorRoom

  await useStore.getState().updateProject(projectId, { name: 'Remote renamed' })
  await useStore.getState().renameRoom(roomId, 'Room renamed')
  await useStore.getState().setRoomSeats(roomId, [{ id: 'seat1', name: 'Builder', role: 'Engineer', providerId: 'local', model: 'm' } as any])
  await useStore.getState().deleteRoom(roomId)
  assert.equal(useStore.getState().roomStatus[roomId], undefined, 'room delete clears stale room status')
  assert.equal(useStore.getState().roomLive[roomId], undefined, 'room delete clears stale live room text')
  await useStore.getState().deleteProject(projectId)

  assert.ok(calls.includes(`project:update:mini:${projectId}:Remote renamed`), 'project update forwards to Orchestrator')
  assert.ok(calls.includes(`room:rename:mini:${roomId}:Room renamed`), 'room rename forwards to the dedicated Orchestrator rename endpoint')
  assert.ok(calls.includes(`room:update:mini:${roomId}:Builder`), 'room seat changes forward to Orchestrator update')
  assert.ok(calls.includes(`room:delete:mini:${roomId}`), 'room delete forwards to Orchestrator')
  assert.ok(calls.includes(`project:delete:mini:${projectId}`), 'project delete forwards to Orchestrator')
  for (const { kind, patch } of localProxyPatches) {
    assert.equal('id' in patch, false, `${kind} proxy persistence never patches identity`)
    assert.equal('archivedAt' in patch, false, `${kind} proxy persistence never patches archive state`)
    assert.equal('createdAt' in patch, false, `${kind} proxy persistence never patches creation time`)
    assert.equal('updatedAt' in patch, false, `${kind} proxy persistence lets the host stamp update time`)
    if (kind === 'project') {
      assert.equal('cwd' in patch, false, 'project proxy persistence cannot retarget the local execution root')
      assert.equal('workspace' in patch, false, 'project proxy persistence cannot forge workspace identity')
      assert.equal('executionTarget' in patch, false, 'project route uses its dedicated host command')
      assert.equal('orchestratorId' in patch, false, 'project route host id uses its dedicated command')
      assert.equal('mode' in patch, false, 'unsupported project snapshot fields never poison the strict config patch')
      assert.ok(Object.keys(patch).every((key) => ['name', 'tags', 'kanban', 'localOnly', 'defaultProviderId', 'defaultProviderAccountId', 'defaultModel', 'memory', 'team'].includes(key)), 'project proxy patch contains only strict config keys')
    } else {
      assert.equal('orchestratorHostId' in patch, false, 'room proxy persistence cannot forge origin routing')
      assert.equal('seatHeartbeats' in patch, false, 'room proxy persistence cannot overwrite host liveness')
    }
  }
  assert.ok(localProxyPatches.some((entry) => entry.kind === 'project'), 'remote project state is persisted through an admissible local patch')
  assert.equal(localProxyPatches.some((entry) => entry.kind === 'room'), false, 'remote room config stays on its Orchestrator and is not replayed through the local room patch boundary')
  ok('remote project and room lifecycle actions route through the selected Orchestrator')
})()

await (async () => {
  const roomId = 'room-error-liveness'
  useStore.setState({
    rooms: [{ id: roomId, projectId: 'proj-room-error', name: 'Room errors', seats: [], state: 'running', events: [], createdAt: 0, updatedAt: 0 } as any],
    roomStatus: { [roomId]: 'running' } as any,
    roomLive: {},
  })
  useStore.getState().applyRoomEvent({ roomId, type: 'error', message: 'worker warning' })
  assert.equal(useStore.getState().rooms.find((room) => room.id === roomId)?.state, 'running', 'nonfatal room errors do not fail the room')
  assert.equal(useStore.getState().roomStatus[roomId], 'running', 'nonfatal room errors keep status running')
  assert.equal(useStore.getState().rooms.find((room) => room.id === roomId)?.events.at(-1)?.kind, 'error', 'nonfatal room errors still append transcript evidence')
  useStore.getState().applyRoomEvent({ roomId, type: 'error', message: 'room crashed', fatal: true })
  assert.equal(useStore.getState().rooms.find((room) => room.id === roomId)?.state, 'failed', 'fatal room errors fail the room')
  assert.equal(useStore.getState().roomStatus[roomId], 'failed', 'fatal room errors set failed status')
  ok('room error events distinguish nonfatal warnings from fatal run failures')
})()

await (async () => {
  const projectId = 'proj-promote-remote'
  const calls: string[] = []
  useStore.setState({
    settings: { ...settings, remoteOrchestrators: [{ id: 'mini', name: 'Mini', enabled: true }] } as any,
    projects: [{ id: projectId, name: 'Local first', cwd: '/srv/local-first', executionTarget: 'local', orchestratorId: null, createdAt: 0, updatedAt: 0 } as any],
  })
  api.createRemoteOrchestratorProject = (async (hostId: string | null | undefined, project: any) => {
    calls.push(`remote:create:${hostId}:${project.id}:${project.cwd}`)
    return { ...project, executionTarget: 'orchestrator', orchestratorId: hostId, updatedAt: Date.now() } as any
  }) as typeof api.createRemoteOrchestratorProject
  api.updateRemoteOrchestratorProject = (async () => { throw new Error('remote update should not be used before host project exists') }) as typeof api.updateRemoteOrchestratorProject
  api.updateProject = (async (id: string, patch: any) => ({ ...useStore.getState().projects.find((project) => project.id === id), ...patch, updatedAt: Date.now() })) as typeof api.updateProject
  api.setProjectExecutionTarget = (async (id: string, executionTarget: any, orchestratorId?: string | null) => {
    calls.push(`local:route:${id}:${executionTarget}:${orchestratorId ?? ''}`)
    return { ...useStore.getState().projects.find((project) => project.id === id), executionTarget, orchestratorId: executionTarget === 'orchestrator' ? orchestratorId ?? null : null, updatedAt: Date.now() } as any
  }) as typeof api.setProjectExecutionTarget

  await useStore.getState().updateProject(projectId, { executionTarget: 'orchestrator', orchestratorId: 'mini' })

  assert.ok(calls.includes(`remote:create:mini:${projectId}:/srv/local-first`), 'switching a local project to Orchestrator creates the host-side project record')
  assert.ok(calls.includes(`local:route:${projectId}:orchestrator:mini`), 'the validated dedicated command persists the new route')
  assert.equal(useStore.getState().projects.find((project) => project.id === projectId)?.executionTarget, 'orchestrator')
  ok('switching an existing project to Orchestrator upserts it on the selected host')
})()

await (async () => {
  const projectId = 'proj-switch-host'
  const calls: string[] = []
  useStore.setState({
    settings: {
      ...settings,
      remoteOrchestrators: [
        { id: 'old', name: 'Old', enabled: true },
        { id: 'new', name: 'New', enabled: true },
      ],
    } as any,
    projects: [{ id: projectId, name: 'Switch host', cwd: '/srv/switch', executionTarget: 'orchestrator', orchestratorId: 'old', createdAt: 0, updatedAt: 0 } as any],
  })
  api.createRemoteOrchestratorProject = (async (hostId: string | null | undefined, project: any) => {
    calls.push(`remote:create:${hostId}:${project.id}`)
    return { ...project, executionTarget: 'orchestrator', orchestratorId: hostId, updatedAt: Date.now() } as any
  }) as typeof api.createRemoteOrchestratorProject
  api.updateRemoteOrchestratorProject = (async (hostId: string | null | undefined, id: string) => {
    calls.push(`remote:update:${hostId}:${id}`)
    return { ...useStore.getState().projects.find((project) => project.id === id), executionTarget: 'orchestrator', orchestratorId: hostId, updatedAt: Date.now() } as any
  }) as typeof api.updateRemoteOrchestratorProject
  api.deleteRemoteOrchestratorProject = (async (hostId: string | null | undefined, id: string) => {
    calls.push(`remote:delete:${hostId}:${id}`)
    return { id, archivedAt: Date.now() } as any
  }) as typeof api.deleteRemoteOrchestratorProject
  api.updateProject = (async (id: string, patch: any) => {
    calls.push(`local:update:${id}:${patch.executionTarget ?? ''}:${patch.orchestratorId ?? ''}`)
    return { ...useStore.getState().projects.find((project) => project.id === id), ...patch, updatedAt: Date.now() } as any
  }) as typeof api.updateProject
  api.setProjectExecutionTarget = (async (id: string, executionTarget: any, orchestratorId?: string | null) => {
    calls.push(`local:route:${id}:${executionTarget}:${orchestratorId ?? ''}`)
    return { ...useStore.getState().projects.find((project) => project.id === id), executionTarget, orchestratorId: executionTarget === 'orchestrator' ? orchestratorId ?? null : null, updatedAt: Date.now() } as any
  }) as typeof api.setProjectExecutionTarget

  await useStore.getState().updateProject(projectId, { orchestratorId: 'new' })
  await useStore.getState().updateProject(projectId, { executionTarget: 'local', orchestratorId: null })

  assert.ok(calls.includes(`remote:create:new:${projectId}`), 'switching hosts creates/upserts the project on the new host')
  assert.ok(calls.includes(`remote:delete:old:${projectId}`), 'switching hosts deletes the stale project proxy on the old host')
  assert.ok(calls.includes(`remote:delete:new:${projectId}`), 'switching back to local deletes the selected host project')
  assert.ok(calls.includes(`local:route:${projectId}:orchestrator:new`), 'switching hosts persists the validated new route')
  assert.ok(calls.includes(`local:route:${projectId}:local:`), 'switching back to local persists through the dedicated route command')
  assert.equal(useStore.getState().projects.find((project) => project.id === projectId)?.executionTarget, 'local')
  ok('project run-location switches cleanly between Orchestrators and back to Local')
})()

await (async () => {
  const projectId = 'proj-serialized-updates'
  useStore.setState({
    projects: [{ id: projectId, name: 'Serialized', cwd: '/tmp', memory: 'old', team: 'old-team', executionTarget: 'local', createdAt: 0, updatedAt: 0 } as any],
  })
  const calls: string[] = []
  let releaseFirst!: () => void
  const firstGate = new Promise<void>((resolve) => { releaseFirst = resolve })
  api.updateProject = (async (id: string, patch: any) => {
    calls.push(String(patch.memory ?? patch.team ?? ''))
    if (patch.memory === 'first') await firstGate
    return { ...useStore.getState().projects.find((project) => project.id === id), ...patch, updatedAt: Date.now() } as any
  }) as typeof api.updateProject

  const first = useStore.getState().updateProject(projectId, { memory: 'first' })
  const second = useStore.getState().updateProject(projectId, { memory: 'second' })
  assert.equal(useStore.getState().projects[0].memory, 'second', 'the latest keystroke is optimistic immediately')
  await new Promise<void>((resolve) => setImmediate(resolve))
  assert.deepEqual(calls, ['first'], 'same-project persistence is serialized')
  releaseFirst()
  await Promise.all([first, second])
  assert.deepEqual(calls, ['first', 'second'])
  assert.equal(useStore.getState().projects[0].memory, 'second', 'a slower stale response cannot overwrite the later edit')

  api.updateProject = (async (id: string, patch: any) => {
    if (patch.memory === 'fails') throw new Error('persist failed')
    return { ...useStore.getState().projects.find((project) => project.id === id), ...patch, updatedAt: Date.now() } as any
  }) as typeof api.updateProject
  const failed = useStore.getState().updateProject(projectId, { memory: 'fails' }).catch((error) => error)
  const independentField = useStore.getState().updateProject(projectId, { team: 'new-team' })
  assert.equal(useStore.getState().projects[0].team, 'new-team')
  assert.match(String((await failed)?.message), /persist failed/)
  await independentField
  assert.equal(useStore.getState().projects[0].memory, 'second', 'failed field rolls back to its prior value')
  assert.equal(useStore.getState().projects[0].team, 'new-team', 'failure cannot clobber a newer independent edit')

  api.updateProject = (async () => { throw new Error('offline') }) as typeof api.updateProject
  const failA = useStore.getState().updateProject(projectId, { memory: 'phantom-a' }).catch((error) => error)
  const failB = useStore.getState().updateProject(projectId, { memory: 'phantom-b' }).catch((error) => error)
  await Promise.all([failA, failB])
  assert.equal(useStore.getState().projects[0].memory, 'second', 'consecutive failures restore the last durable baseline, not the penultimate optimistic value')
  ok('project updates serialize per project and use field-scoped stale-response/rollback guards')
})()

await (async () => {
  const projectId = 'proj-remote-tombstone'
  const otherProjectId = 'proj-other'
  const threadId = 'thread-remote-owned'
  const otherThreadId = 'thread-other'
  const roomId = 'room-remote-owned'
  const otherRoomId = 'room-other'
  useStore.setState({
    projects: [
      { id: projectId, name: 'Remote', cwd: '/srv/app', executionTarget: 'orchestrator', orchestratorId: 'mini', createdAt: 0, updatedAt: 1 } as any,
      { id: otherProjectId, name: 'Other', cwd: '/srv/other', executionTarget: 'local', createdAt: 0, updatedAt: 1 } as any,
    ],
    threads: [
      { id: threadId, title: 'Owned', cwd: '', kind: 'task', projectId, timeline: [], apiMessages: [], createdAt: 0, updatedAt: 1, orchestratorHostId: 'mini' } as any,
      { id: otherThreadId, title: 'Other', cwd: '', kind: 'task', projectId: otherProjectId, timeline: [], apiMessages: [], createdAt: 0, updatedAt: 1 } as any,
    ],
    rooms: [
      { id: roomId, projectId, name: 'Owned room', seats: [], state: 'running', events: [], createdAt: 0, updatedAt: 1, orchestratorHostId: 'mini' } as any,
      { id: otherRoomId, projectId: otherProjectId, name: 'Other room', seats: [], state: 'ready', events: [], createdAt: 0, updatedAt: 1 } as any,
    ],
    tickets: [
      { id: 'ticket-owned', projectId, title: 'Owned ticket', status: 'todo', updatedAt: 1, orchestratorHostId: 'mini' } as any,
      { id: 'ticket-other', projectId: otherProjectId, title: 'Other ticket', status: 'todo', updatedAt: 1 } as any,
    ],
    status: { [threadId]: 'thinking', [otherThreadId]: 'thinking' } as any,
    approvals: { [threadId]: [{ callId: 'owned-call', itemId: 'owned-item' }], [otherThreadId]: [{ callId: 'other-call', itemId: 'other-item' }] },
    roomStatus: { [roomId]: 'running', [otherRoomId]: 'ready' } as any,
    roomLive: { [roomId]: { seat: 'Builder', text: 'working' }, [otherRoomId]: { seat: 'Other', text: 'idle' } },
    terminals: { [projectId]: [{ id: 'term-owned' } as any], [otherProjectId]: [{ id: 'term-other' } as any] },
    activeTerminalId: { [projectId]: 'term-owned', [otherProjectId]: 'term-other' },
    view: { name: 'thread', id: threadId },
  })

  useStore.getState().applyProjectUpdate({ id: projectId, name: 'Remote', cwd: '/srv/app', executionTarget: 'orchestrator', orchestratorId: 'mini', archivedAt: Date.now(), updatedAt: Date.now() } as any)

  const state = useStore.getState()
  assert.deepEqual(state.projects.map((project) => project.id), [otherProjectId], 'project tombstone removes the remote project proxy')
  assert.deepEqual(state.threads.map((thread) => thread.id), [otherThreadId], 'project tombstone removes project-owned threads')
  assert.deepEqual(state.rooms.map((room) => room.id), [otherRoomId], 'project tombstone removes project-owned rooms')
  assert.deepEqual(state.tickets.map((ticket) => ticket.id), ['ticket-other'], 'project tombstone removes project-owned tickets')
  assert.equal(state.status[threadId], undefined, 'project tombstone clears owned thread status')
  assert.equal(state.approvals[threadId], undefined, 'project tombstone clears owned thread approvals')
  assert.equal(state.roomStatus[roomId], undefined, 'project tombstone clears owned room status')
  assert.equal(state.roomLive[roomId], undefined, 'project tombstone clears owned room live text')
  assert.equal(state.terminals[projectId], undefined, 'project tombstone clears owned terminal cache')
  assert.equal(state.activeTerminalId[projectId], undefined, 'project tombstone clears owned active terminal')
  assert.deepEqual(state.view, { name: 'projects' }, 'view falls back when the selected thread belonged to the deleted project')
  ok('project tombstones remove owned threads, rooms, tickets, terminals, and run state from the client proxy')
})()

// ---------------------------------------------------------------------------
// deleteThread: must clean up the per-thread status/approvals maps, not just
// st.threads, or both maps leak orphaned entries keyed to dead thread ids
// over a long-running session (#29).
// ---------------------------------------------------------------------------
console.log('\ndeleteThread cleanup tests:')

await (async () => {
  const threadId = 'th-del'
  useStore.setState({
    threads: [{ id: threadId, title: 't', cwd: '', kind: 'task', projectId: null, timeline: [], createdAt: 0, updatedAt: 0 } as any],
    status: { ...useStore.getState().status, [threadId]: 'thinking' },
    approvals: { ...useStore.getState().approvals, [threadId]: [{ callId: 'c1', itemId: 'i1' }] },
  })
  assert.equal(useStore.getState().status[threadId], 'thinking', 'precondition: status set for thread')
  assert.equal(useStore.getState().approvals[threadId]?.length, 1, 'precondition: approvals set for thread')

  await useStore.getState().deleteThread(threadId)

  assert.equal(useStore.getState().threads.find((t) => t.id === threadId), undefined, 'thread removed from threads')
  assert.equal(threadId in useStore.getState().status, false, 'status entry removed for deleted thread')
  assert.equal(threadId in useStore.getState().approvals, false, 'approvals entry removed for deleted thread')
  ok('deleteThread removes the thread\'s status + approvals entries, not just the thread itself')
})()

// ---- global orchestrator mode routing (VST-041) ----
console.log('\nglobal orchestrator mode routing:')
await (async () => {
  const hosts = [{ id: 'mini', name: 'Mac mini', serverUrl: 'http://127.0.0.1:9100', authToken: 't', enabled: true, lastConnectedAt: null, lastError: null, createdAt: 1, updatedAt: 1 }]
  useStore.setState({
    settings: { ...baseSettings, remoteOrchestrators: hosts, defaultExecutionTarget: 'orchestrator', defaultOrchestratorId: 'mini' },
    projects: [],
    threads: [],
  })
  const remoteCalls: Array<{ hostId: string | null | undefined; input: any }> = []
  api.createRemoteOrchestratorThread = (async (hostId: any, input: any) => {
    remoteCalls.push({ hostId, input })
    return { id: 'rt1', title: 'New task', kind: input.kind ?? 'task', projectId: input.projectId ?? null, cwd: input.cwd ?? '', providerId: null, model: null, createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: [], timeline: [] }
  }) as typeof api.createRemoteOrchestratorThread
  api.createThread = (async () => { throw new Error('must not create locally while global orchestrator mode is on') }) as typeof api.createThread

  await useStore.getState().newThread(null)
  assert.equal(remoteCalls.length, 1, 'a project-less new task routes to the orchestrator host')
  assert.equal(remoteCalls[0].hostId, 'mini')
  const mirrored = useStore.getState().threads.find((t) => t.id === 'rt1')
  assert.equal(mirrored?.orchestratorHostId, 'mini', 'the created thread carries its origin stamp for later runs/guides/cancels')
  ok('global mode: project-less work runs on the configured host, origin-stamped')

  // Changing a creation default must not reinterpret ids that already live locally.
  let localStarts = 0
  api.startRun = (async () => { localStarts++; return { runId: 'local-run' } }) as typeof api.startRun
  api.startRemoteOrchestratorThread = (async () => { throw new Error('an unstamped local thread must not be sent to the default host') }) as typeof api.startRemoteOrchestratorThread
  const existingLocal = { id: 'existing-local', title: 'Existing', kind: 'task', projectId: null, cwd: '', createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: [], timeline: [] } as any
  useStore.setState({ threads: [existingLocal, ...useStore.getState().threads], view: { name: 'thread', id: existingLocal.id }, status: {} })
  await useStore.getState().send('stay local')
  assert.equal(localStarts, 1, 'an existing unstamped local thread remains locally owned under global Orchestrator mode')

  const localProject = { id: 'existing-project', name: 'Existing local', cwd: '/local', memory: '', team: '', executionTarget: 'local', orchestratorId: null, createdAt: 1, updatedAt: 1 } as any
  useStore.setState({ projects: [localProject] })
  let localProjectUpdates = 0
  api.updateProject = (async (_id: string, patch: any) => { localProjectUpdates++; return { ...localProject, ...patch, updatedAt: 2 } }) as typeof api.updateProject
  api.updateRemoteOrchestratorProject = (async () => { throw new Error('a global default does not own an existing local Project record') }) as typeof api.updateRemoteOrchestratorProject
  await useStore.getState().updateProject(localProject.id, { memory: 'local edit' })
  assert.equal(localProjectUpdates, 1, 'existing local project config persists locally even when new work defaults remote')
  ok('global/project routing defaults never teleport existing unstamped records')

  // A localOnly project is a hard pin: creation stays local even with the mode on.
  api.createThread = (async (cwd: any, kind: any, projectId: any) => ({ id: 'lt1', title: 'New task', kind, projectId, cwd, providerId: null, model: null, createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: [], timeline: [] })) as typeof api.createThread
  useStore.setState({ projects: [{ id: 'sec', name: 'Secure', cwd: '/sec', mode: 'solo', localOnly: true, executionTarget: 'local', orchestratorId: null, createdAt: 1, updatedAt: 1 } as any] })
  await useStore.getState().newThread('sec')
  assert.equal(remoteCalls.length, 1, 'the localOnly project did not route remote')
  assert.ok(useStore.getState().threads.find((t) => t.id === 'lt1'), 'the localOnly project created its thread locally')
  ok('global mode: a localOnly project stays a hard local pin')
})()

// ---- authoritative conversation replacement + remote ownership integrity ----
console.log('\nprojection integrity:')
await (async () => {
  const original = {
    id: 'revert-thread', title: 'Revert me', kind: 'task', projectId: null, cwd: '',
    providerId: null, model: null, createdAt: 1, updatedAt: 100, status: 'idle', conversationRevision: 0,
    apiMessages: [{ role: 'user', content: 'keep' }, { role: 'assistant', content: 'remove' }],
    timeline: [
      { id: 'keep', kind: 'user', text: 'keep', createdAt: 1 },
      { id: 'remove', kind: 'assistant', text: 'remove', createdAt: 2 },
    ],
  } as any
  useStore.setState({ threads: [original], status: {}, approvals: {} })
  useStore.getState().applyThreadUpdate({
    ...original,
    updatedAt: 200,
    conversationRevision: 1,
    apiMessages: [original.apiMessages[0]],
    timeline: [original.timeline[0]],
  })
  let projected = useStore.getState().threads.find((thread) => thread.id === original.id)!
  assert.deepEqual(projected.timeline.map((item) => item.id), ['keep'], 'a newer conversation epoch accepts authoritative timeline truncation')
  assert.equal(projected.apiMessages.length, 1, 'a newer conversation epoch accepts authoritative message truncation')
  useStore.getState().applyThreadUpdate({ ...original, updatedAt: 250 })
  projected = useStore.getState().threads.find((thread) => thread.id === original.id)!
  assert.deepEqual(projected.timeline.map((item) => item.id), ['keep'], 'an older pre-revert push cannot union removed items back in')
  assert.equal(projected.apiMessages.length, 1, 'an older pre-revert push cannot restore removed API messages')

  useStore.setState({
    threads: [{
      ...projected,
      updatedAt: 300,
      timeline: [...projected.timeline, { id: 'live', kind: 'assistant', text: 'streamed', createdAt: 3 } as any],
      apiMessages: [...projected.apiMessages, { role: 'assistant', content: 'streamed' } as any],
    }],
  })
  useStore.getState().applyThreadUpdate({ ...projected, updatedAt: 250 })
  projected = useStore.getState().threads.find((thread) => thread.id === original.id)!
  assert.ok(projected.timeline.some((item) => item.id === 'live'), 'same-epoch persistence snapshot cannot erase a live stream item')
  assert.equal(projected.apiMessages.length, 2, 'same-epoch persistence snapshot cannot shrink live API messages')
  ok('checkpoint revert truncates authoritatively while same-epoch stream races still merge')
})()

await (async () => {
  useStore.setState({
    rooms: [{ id: 'gone-room', projectId: 'p', state: 'running', updatedAt: 1 } as any],
    roomStatus: { 'gone-room': 'running' } as any,
    roomLive: { 'gone-room': { seat: 'Builder', text: 'working' } },
    tickets: [{ id: 'gone-ticket', projectId: 'p', parentId: null, updatedAt: 1 } as any],
    jobs: [{ id: 'gone-job', updatedAt: 1 } as any],
    automations: [{ id: 'gone-auto', updatedAt: 1 } as any],
    view: { name: 'room', id: 'gone-room' },
  })
  useStore.getState().applyRoomUpdate({ id: 'gone-room', projectId: 'p', archivedAt: 2, updatedAt: 2 } as any)
  useStore.getState().applyTicketUpdate({ id: 'gone-ticket', archivedAt: 2, updatedAt: 2 } as any)
  useStore.getState().applyJobUpdate({ id: 'gone-job', archivedAt: 2, updatedAt: 2 } as any)
  useStore.getState().applyAutomationUpdate({ id: 'gone-auto', archivedAt: 2, updatedAt: 2 } as any)
  const state = useStore.getState()
  assert.equal(state.rooms.some((room) => room.id === 'gone-room'), false)
  assert.equal(state.roomStatus['gone-room'], undefined)
  assert.equal(state.roomLive['gone-room'], undefined)
  assert.equal(state.tickets.some((ticket) => ticket.id === 'gone-ticket'), false)
  assert.equal(state.jobs.some((job) => job.id === 'gone-job'), false)
  assert.equal(state.automations.some((automation) => automation.id === 'gone-auto'), false)
  assert.deepEqual(state.view, { name: 'project', id: 'p' })
  ok('snapshot tombstones remove stale Zustand rows and dependent room state')
})()

await (async () => {
  const projectId = 'remote-owner-project'
  const ticketId = 'remote-owner-ticket'
  const threadId = 'remote-owner-thread'
  const calls: Array<{ kind: string; hostId?: string; patch?: any }> = []
  useStore.setState({
    settings: { ...baseSettings, remoteOrchestrators: [{ id: 'mini', name: 'Mini', enabled: true }] } as any,
    projects: [{ id: projectId, name: 'Remote', cwd: '/srv/remote', executionTarget: 'orchestrator', orchestratorId: 'mini', createdAt: 1, updatedAt: 1 } as any],
    tickets: [{
      id: ticketId, projectId, title: 'Remote work', body: '', type: 'task', status: 'ready', priority: 'medium', effort: 'small', tags: [],
      parentId: null, assignee: null, threadId: null, runState: null, order: 1, createdAt: 1, updatedAt: 1, orchestratorHostId: 'mini',
    } as any],
    threads: [], status: {}, approvals: {},
  })
  api.startRemoteOrchestratorTicketExecution = (async (hostId: string | null | undefined, id: string) => {
    calls.push({ kind: 'start-ticket', hostId: hostId ?? undefined })
    const current = useStore.getState().tickets.find((ticket) => ticket.id === id)!
    const worker = { id: threadId, title: 'Remote work', kind: 'task', projectId, cwd: '/srv/remote', orchestratorHostId: hostId, providerId: null, model: null, createdAt: 1, updatedAt: 1, status: 'running', apiMessages: [], timeline: [] } as any
    const ticket = { ...current, threadId, status: 'in_progress', runState: 'running', execution: { target: 'remote', hostId, threadId, runId: 'remote-run', timelineBaseline: 0, startedAt: 1, finishedAt: null, checks: [], blockers: [], reviewVerdict: null }, updatedAt: 2 } as any
    return { ticket, thread: worker, runId: 'remote-run' }
  }) as typeof api.startRemoteOrchestratorTicketExecution
  api.transitionRemoteOrchestratorTicket = (async (hostId: string | null | undefined, id: string, command: any) => {
    calls.push({ kind: `transition-${command.action}`, hostId: hostId ?? undefined, patch: command })
    const current = useStore.getState().tickets.find((ticket) => ticket.id === id)!
    if (command.action === 'cancel') return { ...current, status: 'needs_clarification', runState: 'cancelled', updatedAt: current.updatedAt + 1 } as any
    if (command.action === 'sync' && current.runState !== 'cancelled') return { ...current, status: 'in_progress', runState: 'running', updatedAt: current.updatedAt + 1 } as any
    return current
  }) as typeof api.transitionRemoteOrchestratorTicket
  api.updateTicket = (async () => { throw new Error('remote ticket lifecycle must not write the local host') }) as typeof api.updateTicket
  api.cancelRun = (async () => { throw new Error('remote cancellation must not hit the local host') }) as typeof api.cancelRun

  await useStore.getState().workTicketRemote(ticketId)
  assert.equal(useStore.getState().tickets[0].execution?.hostId, 'mini')
  useStore.getState().applyThreadUpdate({
    ...useStore.getState().threads.find((thread) => thread.id === threadId)!,
    status: 'running', updatedAt: 2,
  })
  await new Promise((resolve) => setTimeout(resolve, 0))
  assert.ok(calls.some((call) => call.kind === 'transition-sync' && call.hostId === 'mini'), 'thread lifecycle settles on the owning host')

  await useStore.getState().cancel(threadId)
  const cancelled = useStore.getState().tickets.find((ticket) => ticket.id === ticketId)!
  assert.equal(cancelled.runState, 'cancelled')
  assert.equal(cancelled.status, 'needs_clarification')
  const syncCount = calls.filter((call) => call.kind === 'transition-sync').length
  useStore.getState().applyThreadUpdate({
    ...useStore.getState().threads.find((thread) => thread.id === threadId)!,
    status: 'idle', updatedAt: 3,
    timeline: [{ id: 'partial', kind: 'assistant', text: 'partial output', createdAt: 2 } as any],
  })
  await new Promise((resolve) => setTimeout(resolve, 0))
  assert.equal(useStore.getState().tickets.find((ticket) => ticket.id === ticketId)?.runState, 'cancelled', 'idle thread after cancel stays cancelled')
  assert.equal(calls.filter((call) => call.kind === 'transition-sync').length, syncCount + 1, 'idle mirror may request an idempotent host sync after cancellation')
  assert.equal(useStore.getState().tickets.find((ticket) => ticket.id === ticketId)?.runState, 'cancelled', 'host sync preserves cancellation')
  assert.ok(calls.every((call) => !call.hostId || call.hostId === 'mini'), 'every execution/lifecycle/cancel action uses the owning host')
  ok('remote ticket assignment, lifecycle, and cancellation stay on the owner and never become review/done')
})()

await (async () => {
  const projectId = 'archive-closure-project'
  const roomId = 'archive-closure-room'
  const ticketId = 'archive-closure-ticket'
  const otherTicketId = 'archive-other-ticket'
  let archived = false
  const project = { id: projectId, name: 'Remote archive', cwd: '/srv/archive', executionTarget: 'orchestrator', orchestratorId: 'mini', createdAt: 1, updatedAt: 1 } as any
  const room = { id: roomId, projectId, name: 'Recoverable room', state: 'ready', events: [], seats: [], createdAt: 1, updatedAt: 1, orchestratorHostId: 'mini' } as any
  const ticket = { id: ticketId, projectId, roomId, title: 'Room child', status: 'todo', parentId: null, createdAt: 1, updatedAt: 1, orchestratorHostId: 'mini' } as any
  const otherTicket = { id: otherTicketId, projectId, roomId: null, title: 'Unrelated', status: 'todo', parentId: null, createdAt: 1, updatedAt: 1, orchestratorHostId: 'mini' } as any

  useStore.setState({
    settings: { ...baseSettings, remoteOrchestrators: [{ id: 'mini', name: 'Mini', enabled: true, serverUrl: 'http://127.0.0.1:8765', authToken: 'token' }] } as any,
    projects: [project], rooms: [room], tickets: [ticket, otherTicket], threads: [],
    archives: { projects: [], threads: [], rooms: [], tickets: [] },
    roomStatus: { [roomId]: 'ready' } as any, roomLive: {}, notice: null,
  })
  api.archiveItem = (async () => { throw new Error('remote-only room must not mutate a same-id local record') }) as typeof api.archiveItem
  api.restoreItem = (async () => { throw new Error('remote-only room must not restore a same-id local record') }) as typeof api.restoreItem
  api.getArchives = (async () => ({ projects: [], threads: [], rooms: [], tickets: [] })) as typeof api.getArchives
  api.getRemoteOrchestratorArchives = (async () => ({
    projects: [], threads: [], tickets: [],
    rooms: archived ? [{ ...room, archivedAt: 10, updatedAt: 10 }] : [],
  })) as typeof api.getRemoteOrchestratorArchives
  api.deleteRemoteOrchestratorRoom = (async () => {
    archived = true
    return { ...room, archivedAt: 10, updatedAt: 10 }
  }) as typeof api.deleteRemoteOrchestratorRoom
  api.restoreRemoteOrchestratorRoom = (async () => {
    archived = false
    return { ...room, archivedAt: null, updatedAt: 11 }
  }) as typeof api.restoreRemoteOrchestratorRoom
  api.getRemoteSnapshot = (async () => ({
    ok: true, hostId: 'mini', name: 'Mini', serverUrl: 'http://127.0.0.1:8765',
    rooms: [{ ...room, archivedAt: null, updatedAt: 11 }],
    tickets: [{ ...ticket, archivedAt: null, updatedAt: 11 }],
    threads: [], projects: [project], jobs: [],
  })) as typeof api.getRemoteSnapshot
  api.getProjects = (async () => [project]) as typeof api.getProjects
  api.getThreads = (async () => []) as typeof api.getThreads
  api.getRooms = (async () => []) as typeof api.getRooms
  api.getTickets = (async () => [otherTicket]) as typeof api.getTickets

  await useStore.getState().deleteRoom(roomId)
  assert.equal(useStore.getState().rooms.some((item) => item.id === roomId), false)
  assert.equal(useStore.getState().tickets.some((item) => item.id === ticketId), false, 'room archive removes cascaded tickets without waiting for SSE')
  assert.equal(useStore.getState().tickets.some((item) => item.id === otherTicketId), true, 'room archive preserves unrelated tickets')
  assert.equal(useStore.getState().archives?.rooms[0]?.orchestratorHostId, 'mini')

  await useStore.getState().restoreItem('room', roomId, 'mini')
  assert.equal(useStore.getState().rooms.find((item) => item.id === roomId)?.orchestratorHostId, 'mini')
  assert.equal(useStore.getState().tickets.find((item) => item.id === ticketId)?.orchestratorHostId, 'mini', 'room restore repopulates cascaded tickets from the authenticated snapshot without SSE')

  useStore.setState({ tickets: [
    { id: 'archived-parent', projectId, title: 'Parent', parentId: null, updatedAt: 1 } as any,
    { id: 'retained-child', projectId, title: 'Child', parentId: 'archived-parent', updatedAt: 1 } as any,
  ] })
  useStore.getState().applyTicketUpdate({ id: 'archived-parent', projectId, archivedAt: 12, updatedAt: 12 } as any)
  assert.equal(useStore.getState().tickets.find((item) => item.id === 'retained-child')?.parentId, 'archived-parent', 'recoverable ticket archive preserves subticket ownership')
  ok('remote room archive/restore stays coherent without SSE and ticket archives preserve relationships')
})()

await (async () => {
  const sharedId = 'same-id-on-two-hosts'
  useStore.setState({
    settings: {
      ...baseSettings,
      remoteOrchestrators: [
        { id: 'offline', name: 'Offline', enabled: true, serverUrl: 'http://127.0.0.1:8001', authToken: 'a' },
        { id: 'online', name: 'Online', enabled: true, serverUrl: 'http://127.0.0.1:8002', authToken: 'b' },
      ],
    } as any,
    archives: {
      projects: [], rooms: [], tickets: [],
      threads: [
        { id: sharedId, title: 'Last known offline archive', archivedAt: 5, updatedAt: 5, orchestratorHostId: 'offline' } as any,
        { id: sharedId, title: 'Stale online archive', archivedAt: 6, updatedAt: 6, orchestratorHostId: 'online' } as any,
      ],
    },
  })
  api.getArchives = (async () => ({
    projects: [{ id: 'local-archive', name: 'Local', archivedAt: 2, updatedAt: 2 } as any],
    threads: [], rooms: [], tickets: [],
  })) as typeof api.getArchives
  api.getRemoteOrchestratorArchives = (async (hostId: string | null | undefined) => {
    if (hostId === 'offline') throw new Error('host is offline')
    return {
      projects: [], rooms: [], tickets: [],
      threads: [{ id: sharedId, title: 'Fresh online archive', archivedAt: 9, updatedAt: 9, orchestratorHostId: 'online' } as any],
    }
  }) as typeof api.getRemoteOrchestratorArchives

  await useStore.getState().refreshArchives()
  const archives = useStore.getState().archives!
  assert.equal(archives.projects.some((item) => item.id === 'local-archive'), true)
  assert.equal(archives.threads.length, 2, 'same-id archives remain distinct by origin host')
  assert.equal(archives.threads.find((item) => item.orchestratorHostId === 'offline')?.title, 'Last known offline archive', 'a transient host failure retains its last known archive rows')
  assert.equal(archives.threads.find((item) => item.orchestratorHostId === 'online')?.title, 'Fresh online archive')
  ok('archive refresh preserves multi-host identity and last-known rows for an offline host')
})()

await (async () => {
  const roomId = 'serialized-room-config'
  let serverRoom = {
    id: roomId, projectId: 'serialized-project', name: 'Room', seats: [], mode: 'implement', collaborationMode: 'balanced', deliberationEffort: 'normal',
    workerBudget: { maxActive: 2, maxStarts: 4 }, memory: [], memorySummary: '', problem: '', state: 'ready', events: [], finalText: '', createdAt: 1, updatedAt: 1,
  } as any
  useStore.setState({ rooms: [serverRoom], roomStatus: { [roomId]: 'ready' } })
  const pending: Array<{ patch: any; resolve: (room: any) => void }> = []
  api.updateRoom = (async (_id: string, patch: any) => new Promise((resolve) => pending.push({ patch, resolve }))) as typeof api.updateRoom

  const first = useStore.getState().updateRoom(roomId, { workerBudget: { maxActive: 3, maxStarts: 4 } })
  const second = useStore.getState().updateRoom(roomId, { workerBudget: { maxActive: 2, maxStarts: 8 } })
  await new Promise((resolve) => setTimeout(resolve, 0))
  assert.equal(pending.length, 1, 'only one room config mutation is in flight')
  serverRoom = { ...serverRoom, workerBudget: pending[0].patch.workerBudget, updatedAt: 2 }
  pending[0].resolve(serverRoom)
  await first
  await new Promise((resolve) => setTimeout(resolve, 0))
  assert.equal(pending.length, 2, 'the next room edit starts after the first commits')
  assert.deepEqual(pending[1].patch.workerBudget, { maxActive: 3, maxStarts: 8 }, 'nested budget edits rebase on the latest committed room')
  serverRoom = { ...serverRoom, workerBudget: pending[1].patch.workerBudget, updatedAt: 3 }
  pending[1].resolve(serverRoom)
  await second
  assert.deepEqual(useStore.getState().rooms[0].workerBudget, { maxActive: 3, maxStarts: 8 })

  let releaseSeats!: (room: any) => void
  api.updateRoom = (async (_id: string, patch: any) => new Promise((resolve) => {
    releaseSeats = () => { serverRoom = { ...serverRoom, ...patch, state: 'ready', updatedAt: 4 }; resolve(serverRoom) }
  })) as typeof api.updateRoom
  let starts = 0
  api.startRoom = (async () => { starts++; serverRoom = { ...serverRoom, state: 'queued', updatedAt: 5 }; return { ok: true, room: serverRoom } }) as typeof api.startRoom
  const configure = useStore.getState().updateRoom(roomId, { seats: [{ id: 'seat', name: 'Builder', role: 'Build', providerId: 'p', model: 'm' }] })
  const start = useStore.getState().startRoom(roomId, 'Use the configured seat')
  await new Promise((resolve) => setTimeout(resolve, 0))
  assert.equal(starts, 0, 'room start waits behind pending configuration')
  releaseSeats(serverRoom)
  await configure
  await start
  assert.equal(starts, 1)
  ok('room configuration serializes, rebases nested budgets, and commits before start')
})()

await (async () => {
  const ticketId = 'serialized-ticket'
  let serverTicket = {
    id: ticketId, projectId: 'p', title: 'Ticket', body: '', type: 'task', status: 'todo', priority: 'medium', effort: 'small', tags: [],
    parentId: null, assignee: null, threadId: null, runState: null, order: 1, createdAt: 1, updatedAt: 1,
  } as any
  useStore.setState({ tickets: [serverTicket] })
  const pending: Array<{ patch: any; resolve: (ticket: any) => void; reject: (error: Error) => void }> = []
  api.updateTicket = (async (_id: string, patch: any) => new Promise((resolve, reject) => pending.push({ patch, resolve, reject }))) as typeof api.updateTicket
  const first = useStore.getState().updateTicket(ticketId, { status: 'ready' }).catch(() => undefined)
  const second = useStore.getState().updateTicket(ticketId, { priority: 'high' })
  await new Promise((resolve) => setTimeout(resolve, 0))
  assert.equal(pending.length, 1)
  assert.equal(useStore.getState().tickets[0].status, 'todo', 'ticket edits are pessimistic and cannot leave rejected phantom state')
  pending[0].reject(new Error('first write failed'))
  await first
  await new Promise((resolve) => setTimeout(resolve, 0))
  assert.equal(pending.length, 2, 'a failed edit does not poison the per-ticket queue')
  serverTicket = { ...serverTicket, ...pending[1].patch, updatedAt: 2 }
  pending[1].resolve(serverTicket)
  await second
  assert.equal(useStore.getState().tickets[0].status, 'todo')
  assert.equal(useStore.getState().tickets[0].priority, 'high')
  ok('ticket edits serialize and a failed mutation cannot clobber the next committed edit')
})()

await (async () => {
  const roomId = 'steer-after-config'
  const room = { id: roomId, projectId: 'p', name: 'Room', seats: [], state: 'running', events: [], workerBudget: { maxActive: 2, maxStarts: 4 }, createdAt: 1, updatedAt: 1 } as any
  useStore.setState({ rooms: [room], roomStatus: { [roomId]: 'running' } })
  let releaseConfig!: () => void
  const order: string[] = []
  api.updateRoom = (async (_id: string, patch: any) => new Promise((resolve) => {
    releaseConfig = () => { order.push('config'); resolve({ ...room, ...patch, updatedAt: 2 }) }
  })) as typeof api.updateRoom
  api.steerRoom = (async (_id: string, text: string) => {
    order.push('steer')
    return { id: 'steer-event', tick: 1, kind: 'user', seat: 'you', target: 'all', text, createdAt: 3 }
  }) as typeof api.steerRoom
  const config = useStore.getState().updateRoom(roomId, { workerBudget: { maxActive: 3, maxStarts: 5 } })
  const steer = useStore.getState().steerRoom(roomId, 'Use the new budget')
  await new Promise((resolve) => setTimeout(resolve, 0))
  assert.deepEqual(order, [], 'steering cannot overtake pending room configuration')
  releaseConfig()
  await Promise.all([config, steer])
  assert.deepEqual(order, ['config', 'steer'])
  ok('room steering waits behind pending room mutation')
})()

await (async () => {
  const projectId = 'dispatch-queue-project'
  const roomId = 'dispatch-queue-room'
  const ticketId = 'dispatch-queue-ticket'
  const project = { id: projectId, name: 'Project', cwd: '/tmp', createdAt: 1, updatedAt: 1 } as any
  const room = { id: roomId, projectId, name: 'Room', seats: [{ id: 'old-seat', name: 'Old', role: 'Build', providerId: 'p', model: 'm' }], state: 'ready', events: [], workerBudget: { maxActive: 2, maxStarts: 4 }, createdAt: 1, updatedAt: 1 } as any
  const ticket = { id: ticketId, projectId, roomId, title: 'Queued dispatch', body: 'old body', type: 'task', status: 'todo', priority: 'medium', effort: 'small', tags: [], parentId: null, assignee: null, threadId: null, runState: null, order: 1, createdAt: 1, updatedAt: 1 } as any
  useStore.setState({ projects: [project], rooms: [room], tickets: [ticket], roomStatus: { [roomId]: 'ready' }, settings: baseSettings as any })
  let releaseTicket!: () => void
  let releaseRoom!: () => void
  let ticketWrites = 0
  let startedProblem = ''
  api.updateTicket = (async (_id: string, patch: any) => {
    ticketWrites++
    if (ticketWrites === 1) return new Promise((resolve) => {
      releaseTicket = () => resolve({ ...ticket, ...patch, updatedAt: 2 })
    })
    const current = useStore.getState().tickets.find((candidate) => candidate.id === ticketId)!
    return { ...current, ...patch, updatedAt: 3 } as any
  }) as typeof api.updateTicket
  api.updateRoom = (async (_id: string, patch: any) => new Promise((resolve) => {
    releaseRoom = () => resolve({ ...room, ...patch, updatedAt: 2 })
  })) as typeof api.updateRoom
  api.startRoom = (async (_id: string, problem: string) => {
    startedProblem = problem
    const current = useStore.getState().rooms.find((candidate) => candidate.id === roomId)!
    return { ok: true, room: { ...current, problem, state: 'queued', updatedAt: 4 } }
  }) as typeof api.startRoom

  const ticketEdit = useStore.getState().updateTicket(ticketId, { body: 'new body from pending edit' })
  const roomEdit = useStore.getState().updateRoom(roomId, { seats: [{ id: 'new-seat', name: 'New', role: 'Review', providerId: 'p', model: 'm' }] })
  const dispatch = useStore.getState().startRoomFromTicket(ticketId, undefined)
  await new Promise((resolve) => setTimeout(resolve, 0))
  assert.equal(ticketWrites, 1)
  assert.equal(startedProblem, '', 'dispatch cannot overtake either entity queue')
  releaseTicket(); releaseRoom()
  await Promise.all([ticketEdit, roomEdit, dispatch])
  assert.match(startedProblem, /new body from pending edit/, 'dispatch prompt is built from committed ticket config')
  assert.equal(useStore.getState().rooms.find((candidate) => candidate.id === roomId)?.seats[0]?.id, 'new-seat')
  ok('ticket-to-room dispatch atomically waits for ticket and room queues')
})()

await (async () => {
  const localId = 'work-after-config-local'
  const remoteId = 'work-after-config-remote'
  const local = { id: localId, projectId: 'p', title: 'Local', body: 'old', type: 'task', status: 'ready', priority: 'medium', effort: 'small', tags: [], parentId: null, threadId: null, runState: null, order: 1, createdAt: 1, updatedAt: 1 } as any
  const remote = { ...local, id: remoteId, title: 'Remote', orchestratorHostId: 'mini' } as any
  useStore.setState({ tickets: [local, remote], threads: [] })
  const releases = new Map<string, () => void>()
  api.updateTicket = (async (id: string, patch: any) => new Promise((resolve) => releases.set(id, () => resolve({ ...local, id, ...patch, updatedAt: 2 })))) as typeof api.updateTicket
  api.updateRemoteOrchestratorTicket = (async (_host: string | null | undefined, id: string, patch: any) => new Promise((resolve) => releases.set(id, () => resolve({ ...remote, id, ...patch, updatedAt: 2 })))) as typeof api.updateRemoteOrchestratorTicket
  const calls: string[] = []
  api.workTicket = (async () => { calls.push('local-work'); return { id: 'local-thread', title: 'Local', kind: 'task', projectId: 'p', cwd: '', status: 'idle', apiMessages: [], timeline: [], createdAt: 1, updatedAt: 1 } as any }) as typeof api.workTicket
  api.startRemoteOrchestratorTicketExecution = (async (_host, _id, prompt) => {
    calls.push(prompt)
    return { ticket: { ...remote, body: 'new remote config', runState: 'running' }, thread: { id: 'remote-thread', title: 'Remote', kind: 'task', projectId: 'p', cwd: '', status: 'running', apiMessages: [], timeline: [], createdAt: 1, updatedAt: 1 }, runId: 'run' } as any
  }) as typeof api.startRemoteOrchestratorTicketExecution
  const localEdit = useStore.getState().updateTicket(localId, { body: 'new local config' })
  const remoteEdit = useStore.getState().updateTicket(remoteId, { body: 'new remote config' })
  const localWork = useStore.getState().workTicket(localId)
  const remoteWork = useStore.getState().workTicketRemote(remoteId)
  await new Promise((resolve) => setTimeout(resolve, 0))
  assert.equal(calls.length, 0)
  releases.get(localId)!()
  await localEdit
  await new Promise((resolve) => setTimeout(resolve, 0))
  releases.get(remoteId)!()
  await Promise.all([remoteEdit, localWork, remoteWork])
  assert.equal(calls[0], 'local-work')
  assert.ok(calls.some((call) => call.includes('new remote config')), 'remote prompt uses committed ticket config')
  ok('local and remote ticket work wait behind pending ticket configuration')
})()

await (async () => {
  const projectId = 'archive-after-mutation-project'
  const roomId = 'archive-after-mutation-room'
  const ticketId = 'archive-after-mutation-ticket'
  const project = { id: projectId, name: 'Project', cwd: '/tmp', createdAt: 1, updatedAt: 1 } as any
  const room = { id: roomId, projectId, name: 'Room', seats: [], state: 'ready', events: [], workerBudget: { maxActive: 2, maxStarts: 4 }, createdAt: 1, updatedAt: 1 } as any
  const ticket = { id: ticketId, projectId, roomId, title: 'Ticket', body: '', type: 'task', status: 'todo', priority: 'medium', effort: 'small', tags: [], parentId: null, threadId: null, runState: null, order: 1, createdAt: 1, updatedAt: 1 } as any
  useStore.setState({ projects: [project], rooms: [room], tickets: [ticket], settings: baseSettings as any, archives: { projects: [], threads: [], rooms: [], tickets: [] } })
  let releaseRoom!: () => void
  let archives = 0
  api.updateRoom = (async (_id: string, patch: any) => new Promise((resolve) => {
    releaseRoom = () => resolve({ ...room, ...patch, updatedAt: 2 })
  })) as typeof api.updateRoom
  api.archiveItem = (async (kind: any, id: string) => {
    archives++
    return {
      projects: kind === 'project' ? [{ ...project, id, archivedAt: 3 }] : [], threads: [],
      rooms: kind === 'room' ? [{ ...room, id, archivedAt: 3 }] : [],
      tickets: kind === 'ticket' ? [{ ...ticket, id, archivedAt: 3 }] : [],
    }
  }) as typeof api.archiveItem
  api.getArchives = (async () => ({ projects: [], threads: [], rooms: [], tickets: [] })) as typeof api.getArchives
  const mutation = useStore.getState().updateRoom(roomId, { workerBudget: { maxActive: 3, maxStarts: 6 } })
  const archive = useStore.getState().archiveItem('room', roomId)
  await new Promise((resolve) => setTimeout(resolve, 0))
  assert.equal(archives, 0, 'room archive waits for its pending mutation')
  releaseRoom()
  await Promise.all([mutation, archive])
  assert.equal(useStore.getState().rooms.some((candidate) => candidate.id === roomId), false)
  assert.equal(useStore.getState().tickets.some((candidate) => candidate.id === ticketId), false, 'room cascade remains archived after the prior edit settles')

  useStore.setState({ projects: [project], rooms: [], tickets: [ticket], archives: { projects: [], threads: [], rooms: [], tickets: [] } })
  let releaseProject!: () => void
  archives = 0
  api.updateProject = (async (_id: string, patch: any) => new Promise((resolve) => {
    releaseProject = () => resolve({ ...project, ...patch, updatedAt: 2 })
  })) as typeof api.updateProject
  const projectMutation = useStore.getState().updateProject(projectId, { name: 'Pending rename' })
  const projectArchive = useStore.getState().archiveItem('project', projectId)
  await new Promise((resolve) => setTimeout(resolve, 0))
  assert.equal(archives, 0, 'project archive waits for pending project config')
  releaseProject()
  await Promise.all([projectMutation, projectArchive])
  assert.equal(useStore.getState().projects.some((candidate) => candidate.id === projectId), false)
  ok('archive operations wait for pending entity mutations and cannot be followed by resurrection')
})()

await (async () => {
  const id = 'cross-host-collision'
  const localProject = { id, name: 'Local project', cwd: '/local', createdAt: 1, updatedAt: 1 } as any
  const localThread = { id, title: 'Local thread', kind: 'task', projectId: id, cwd: '/local', status: 'idle', apiMessages: [], timeline: [], createdAt: 1, updatedAt: 1 } as any
  const localRoom = { id, projectId: id, name: 'Local room', seats: [], state: 'ready', events: [], createdAt: 1, updatedAt: 1 } as any
  const localTicket = { id, projectId: id, title: 'Local ticket', status: 'todo', parentId: null, createdAt: 1, updatedAt: 1 } as any
  useStore.setState({ projects: [localProject], threads: [localThread], rooms: [localRoom], tickets: [localTicket], status: {}, approvals: {}, roomStatus: { [id]: 'ready' }, roomLive: {} })

  useStore.getState().applyProjectUpdate({ ...localProject, name: 'Host A project', executionTarget: 'orchestrator', orchestratorId: 'host-a', updatedAt: 2 })
  useStore.getState().applyThreadUpdate({ ...localThread, title: 'Host A thread', orchestratorHostId: 'host-a', updatedAt: 2 })
  useStore.getState().applyRoomUpdate({ ...localRoom, name: 'Host A room', orchestratorHostId: 'host-a', updatedAt: 2 })
  useStore.getState().applyTicketUpdate({ ...localTicket, title: 'Host A ticket', orchestratorHostId: 'host-a', updatedAt: 2 })
  useStore.getState().applyEvent({ runId: 'a', threadId: id, type: 'title', title: 'Host A event title', orchestratorHostId: 'host-a' })
  useStore.getState().applyRoomEvent({ roomId: id, type: 'final', text: 'Host A final', orchestratorHostId: 'host-a' })
  assert.equal(useStore.getState().projects[0].name, 'Local project')
  assert.equal(useStore.getState().threads[0].title, 'Local thread')
  assert.equal(useStore.getState().rooms[0].name, 'Local room')
  assert.equal(useStore.getState().rooms[0].finalText, undefined)
  assert.equal(useStore.getState().tickets[0].title, 'Local ticket')

  useStore.getState().applyThreadUpdate({ ...localThread, archivedAt: 3, orchestratorHostId: 'host-a' })
  useStore.getState().applyRoomUpdate({ ...localRoom, archivedAt: 3, orchestratorHostId: 'host-a' })
  useStore.getState().applyTicketUpdate({ ...localTicket, archivedAt: 3, orchestratorHostId: 'host-a' })
  useStore.getState().applyProjectUpdate({ ...localProject, archivedAt: 3, executionTarget: 'orchestrator', orchestratorId: 'host-a' })
  assert.equal(useStore.getState().threads.length, 1, 'foreign tombstone cannot delete local thread')
  assert.equal(useStore.getState().rooms.length, 1, 'foreign tombstone cannot delete local room')
  assert.equal(useStore.getState().tickets.length, 1, 'foreign tombstone cannot delete local ticket')
  assert.equal(useStore.getState().projects.length, 1, 'foreign tombstone cannot delete local project')

  const remoteProject = { ...localProject, name: 'A project', executionTarget: 'orchestrator', orchestratorId: 'host-a' } as any
  const remoteThread = { ...localThread, title: 'A thread', orchestratorHostId: 'host-a' } as any
  const remoteRoom = { ...localRoom, name: 'A room', orchestratorHostId: 'host-a' } as any
  const remoteTicket = { ...localTicket, title: 'A ticket', orchestratorHostId: 'host-a' } as any
  useStore.setState({ projects: [remoteProject], threads: [remoteThread], rooms: [remoteRoom], tickets: [remoteTicket], status: {}, approvals: {}, roomStatus: { [id]: 'ready' } })
  useStore.getState().applyThreadUpdate({ ...remoteThread, title: 'B thread', orchestratorHostId: 'host-b', updatedAt: 4 })
  useStore.getState().applyRoomUpdate({ ...remoteRoom, name: 'B room', orchestratorHostId: 'host-b', updatedAt: 4 })
  useStore.getState().applyTicketUpdate({ ...remoteTicket, title: 'B ticket', orchestratorHostId: 'host-b', updatedAt: 4 })
  useStore.getState().applyProjectUpdate({ ...remoteProject, name: 'B project', orchestratorId: 'host-b', updatedAt: 4 })
  useStore.getState().applyEvent({ runId: 'b', threadId: id, type: 'title', title: 'B event', orchestratorHostId: 'host-b' })
  useStore.getState().applyRoomEvent({ roomId: id, type: 'final', text: 'B final', orchestratorHostId: 'host-b' })
  assert.equal(useStore.getState().projects[0].name, 'A project')
  assert.equal(useStore.getState().threads[0].title, 'A thread')
  assert.equal(useStore.getState().rooms[0].name, 'A room')
  assert.equal(useStore.getState().tickets[0].title, 'A ticket')

  useStore.getState().applyEvent({ runId: 'a', threadId: id, type: 'title', title: 'A event accepted', orchestratorHostId: 'host-a' })
  useStore.getState().applyRoomEvent({ roomId: id, type: 'final', text: 'A final accepted', orchestratorHostId: 'host-a' })
  assert.equal(useStore.getState().threads[0].title, 'A event accepted')
  assert.equal(useStore.getState().rooms[0].finalText, 'A final accepted')
  ok('local and host A/B projection collisions quarantine records, tombstones, and live events')
})()

await (async () => {
  const id = 'restore-origin-collision'
  const local = { id, projectId: 'p', title: 'Local active', body: '', type: 'task', status: 'todo', priority: 'medium', effort: 'small', tags: [], parentId: null, threadId: null, runState: null, order: 1, createdAt: 1, updatedAt: 1 } as any
  const remoteArchived = { ...local, title: 'Host A archived', orchestratorHostId: 'host-a', archivedAt: 5, updatedAt: 5 } as any
  useStore.setState({ tickets: [local], projects: [], rooms: [], threads: [], archives: { projects: [], rooms: [], threads: [], tickets: [remoteArchived] }, settings: baseSettings as any })
  let releaseEdit!: () => void
  let restores = 0
  api.updateTicket = (async (_id: string, patch: any) => new Promise((resolve) => {
    releaseEdit = () => resolve({ ...local, ...patch, updatedAt: 2 })
  })) as typeof api.updateTicket
  api.restoreRemoteOrchestratorTicket = (async () => { restores++; return { ...remoteArchived, archivedAt: null, updatedAt: 6 } }) as typeof api.restoreRemoteOrchestratorTicket
  api.getProjects = (async () => []) as typeof api.getProjects
  api.getThreads = (async () => []) as typeof api.getThreads
  api.getRooms = (async () => []) as typeof api.getRooms
  api.getTickets = (async () => useStore.getState().tickets) as typeof api.getTickets
  api.getArchives = (async () => ({ projects: [], rooms: [], threads: [], tickets: [] })) as typeof api.getArchives

  const edit = useStore.getState().updateTicket(id, { title: 'Local committed' })
  const restore = useStore.getState().restoreItem('ticket', id, 'host-a')
  await new Promise((resolve) => setTimeout(resolve, 0))
  assert.equal(restores, 0, 'restore waits behind pending same-id ticket edits')
  releaseEdit()
  await Promise.all([edit, restore])
  assert.equal(restores, 1)
  assert.equal(useStore.getState().tickets.find((ticket) => ticket.id === id)?.title, 'Local committed', 'restoring host A cannot replace active local same-id ticket')
  ok('restore is queue-ordered and origin-aware')
})()

console.log(`\n${passed} passed`)
