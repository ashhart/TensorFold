import assert from 'node:assert/strict'
import { defaultSettings } from '../../electron/store/defaults'
import type { RunExportBundle } from '../lib/types'
import { find, harness, jsx, load, settle, text } from './reading/reading-test-harness'

const globals = globalThis as Record<string, unknown>
const events = new EventTarget()
globals.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} }
globals.matchMedia = () => ({ matches: false, addEventListener: () => {}, removeEventListener: () => {} })
globals.window = globals
globals.addEventListener = events.addEventListener.bind(events)
globals.removeEventListener = events.removeEventListener.bind(events)

await (await import('../lib/browser-demo-bootstrap')).prepareBrowserDemo()
const { api } = await import('../lib/ipc')
const { useStore } = await import('../lib/store')
useStore.setState({
  settings: { ...defaultSettings(), activeProviderId: 'fixture-provider', activeModel: 'fixture-model' },
  saveSettings: async (settings) => { useStore.setState({ settings }) },
  threads: [], projects: [],
})

let prompt = 'Explain why four is even.'
const created: string[] = []
const starts: { threadId: string; userText: string }[] = []
let admit = async () => ({ runId: 'fixture-run' })
api.importBundle = async () => ({
  bundle: { mode: 'full', thread: { title: 'Synthetic replay source' }, messages: [{ role: 'user', content: prompt }] } as RunExportBundle,
})
api.createThread = async (cwd, kind = 'task', projectId = null) => {
  const id = `fixture-task-${created.length + 1}`
  created.push(id)
  return {
    id, cwd, kind, projectId, workspace: null, title: id, status: 'idle', createdAt: 1, updatedAt: 1,
    providerId: null, providerAccountId: null, model: null, timeline: [], apiMessages: [],
  }
}
api.configureThread = async () => {}
api.startRun = async ({ threadId, userText }) => {
  starts.push({ threadId, userText })
  return admit()
}

const hooks = harness()
const { ReplayPanel } = load('src/components/ReplayPanel.tsx', {
  react: hooks.hooks, 'react/jsx-runtime': jsx,
  clsx: (...values: unknown[]) => values.filter(Boolean).join(' '),
  '../lib/store': { useStore }, '../lib/ipc': { api }, '../lib/icons': { Icon: () => null },
})
const render = () => hooks.render(() => ReplayPanel())
const button = (label: string) => {
  const node = find(render(), (item) => item.type === 'button' && text(item) === label)
  assert.ok(node, label)
  return node
}
async function openBundle() {
  events.dispatchEvent(new Event('vontra:replay'))
  await button('Choose bundle…').props.onClick()
}
function holdAdmission() {
  let complete!: (value: { runId: string }) => void
  const pending = new Promise<{ runId: string }>((resolve) => { complete = resolve })
  admit = () => pending
  return () => complete({ runId: 'fixture-run' })
}

render(); hooks.flush()
await openBundle()
const complete = holdAdmission()
const replay = button('Replay as new task')
const first = replay.props.onClick()
const second = replay.props.onClick()
await settle()
assert.deepEqual(starts, [{ threadId: 'fixture-task-1', userText: prompt }], 'rapid clicks before rerender only admit one task')
assert.deepEqual(created, ['fixture-task-1'])
assert.equal(button('Starting…').props.disabled, true, 'the replay button is disabled while admission is pending')
complete()
await Promise.all([first, second])
assert.equal(render(), null, 'successful admission closes this replay')

await openBundle()
api.configureThread = async () => { throw new Error('Synthetic configuration failed') }
await button('Replay as new task').props.onClick()
assert.match(text(render()), /Synthetic configuration failed/)
assert.equal(starts.length, 1, 'configuration failure occurs before admission')
assert.equal(button('Replay as new task').props.disabled, false, 'an early rejection can be retried')
api.configureThread = async () => {}
admit = async () => { throw new Error('Synthetic admission failed') }
await button('Replay as new task').props.onClick()
assert.equal(starts.length, 2)
assert.match(text(render()), /Synthetic admission failed/)
assert.equal(button('Replay as new task').props.disabled, false, 'a rejected admission can be retried')
admit = async () => ({ runId: 'retry-run' })
await button('Replay as new task').props.onClick()
assert.equal(starts.length, 3)
assert.equal(render(), null, 'a successful retry closes the panel')

await openBundle()
const completeOld = holdAdmission()
const oldLaunch = button('Replay as new task').props.onClick()
await settle()
assert.equal(starts.length, 4)
button('Close').props.onClick()
assert.equal(render(), null)
prompt = 'Explain why five is odd.'
await openBundle()
completeOld()
await oldLaunch
assert.match(text(render()), /Explain why five is odd/)
assert.equal(button('Replay as new task').props.disabled, false, 'an old success leaves the reopened replay usable')
hooks.cleanup()
console.log('Replay starts one task, permits retries after failures, and leaves a reopened panel usable')
