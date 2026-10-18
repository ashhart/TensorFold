import assert from 'node:assert/strict'
import { canUseWorkerProject, isLocalWorker, launchLocalWorker, makeWorkerDraft, workerState } from './workers'

const globals = globalThis as Record<string, unknown>
globals.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} }
globals.matchMedia = () => ({ matches: false, addEventListener: () => {}, removeEventListener: () => {} })
globals.addEventListener = () => {}
globals.removeEventListener = () => {}
globals.window = globals

await (await import('../../lib/browser-demo-bootstrap')).prepareBrowserDemo()
const { api } = await import('../../lib/ipc')
const { useStore } = await import('../../lib/store')
const settings = await api.getSettings()
const projects = await api.getProjects()
const project = projects.find(canUseWorkerProject)!
assert.ok(project, 'Browser demo supplies a local project')
useStore.setState({ settings, projects, threads: [], status: {}, approvals: {}, view: { name: 'models' } })
const unsubscribe = api.onAgentEvent((event) => useStore.getState().applyEvent(event))
const model = (await api.engineSnapshot()).installed[0].repo
await api.engineLoad(model)

const launch = (title: string, identity: string) => launchLocalWorker(makeWorkerDraft(project, title, 'Update the repository documentation and check its examples.', model, identity), {
  api,
  getState: useStore.getState,
  onThread: (thread) => useStore.setState((state) => ({ threads: [thread, ...state.threads.filter((t) => t.id !== thread.id)] })),
  onStarting: (id) => useStore.setState((state) => ({ status: { ...state.status, [id]: 'thinking' } })),
  onStartFailed: (id) => useStore.setState((state) => ({ status: { ...state.status, [id]: 'error' } })),
})

try {
  const first = await launch('Update docs', '11111111-1111-4111-8111-111111111111')
  const second = await launch('Check examples', '22222222-2222-4222-8222-222222222222')
  assert.notEqual(first.draft.path, second.draft.path, 'Independent workers never share their worktree')
  assert.equal(useStore.getState().view.name, 'models', 'Launch keeps the integrated workspace open')
  const threads = useStore.getState().threads.filter(isLocalWorker)
  assert.equal(threads.length, 2)
  assert.equal(threads.find((thread) => thread.id === first.threadId)?.title, 'Update docs')
  assert.ok(threads.every((thread) => thread.providerId === 'tensorfold' && thread.model === model))
  assert.ok(threads.every((thread) => workerState(thread, useStore.getState().status[thread.id]) === 'running'))
  const worktrees = await api.getWorktrees({ cwd: project.cwd })
  assert.ok(worktrees.worktrees.some((row) => row.path === first.draft.path && row.branch === first.draft.branch))
  assert.ok(worktrees.worktrees.some((row) => row.path === second.draft.path && row.branch === second.draft.branch))
  assert.equal((await api.createWorktree({ cwd: project.cwd, name: first.draft.branch, mode: 'worktree' })).ok, false, 'Demo rejects duplicate branch creation')

  await useStore.getState().cancel(first.threadId)
  const stopped = useStore.getState().threads.find((thread) => thread.id === first.threadId)!
  assert.equal(workerState(stopped, useStore.getState().status[first.threadId]), 'stopped')
  assert.equal(useStore.getState().status[first.threadId], 'idle')
  const stoppedTimelineLength = stopped.timeline.length
  await new Promise((resolve) => setTimeout(resolve, 350))
  assert.equal(useStore.getState().status[first.threadId], 'idle', 'Stopped simulations must not resume on their next timer')
  assert.equal(useStore.getState().threads.find((thread) => thread.id === first.threadId)!.timeline.length, stoppedTimelineLength)
  assert.equal(workerState(useStore.getState().threads.find((thread) => thread.id === second.threadId)!, useStore.getState().status[second.threadId]), 'running', 'Stopping one worker leaves its peer running')
  assert.ok((await api.getWorktrees({ cwd: project.cwd })).worktrees.some((row) => row.path === first.draft.path), 'Stop preserves worktree changes for review')

  useStore.getState().openThread(first.threadId)
  useStore.getState().openWorkPane('changes')
  assert.deepEqual(useStore.getState().view, { name: 'thread', id: first.threadId })
  assert.equal(useStore.getState().workPaneTabs[first.threadId], 'changes')
  assert.equal(useStore.getState().rightPanelHidden, false)
} finally {
  for (const thread of useStore.getState().threads.filter(isLocalWorker)) await useStore.getState().cancel(thread.id)
  unsubscribe()
}

console.log('Browser demo launches independent local workers, preserves navigation and worktrees, stops only the selected worker, and opens Changes for review.')
