import assert from 'node:assert/strict'
import { harness, jsx, load, settle } from '../reading/reading-test-harness'
import * as execution from '../../lib/execution-target'
import * as workspace from '../../lib/thread-workspace'
import * as format from '../../lib/format'
import { terminalSessionKey } from '../../lib/terminal-sessions'

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => { resolve = done })
  return { promise, resolve }
}
const projects = [{ id: 'alpha', name: 'Alpha', cwd: '/synthetic/alpha' }, { id: 'beta', name: 'Beta', cwd: '/synthetic/beta' }]
const state: any = { projects, terminals: {}, activeTerminalId: {}, sshHosts: [], sshStatus: {}, homeComposerWorkspace: null }
const settings: any = { defaultCwd: '/synthetic/default', remoteOrchestrators: [], defaultExecutionTarget: 'local' }
const useStore = Object.assign((select: any) => select(state), { getState: () => state, setState: (value: any) => Object.assign(state, value) })
let picker = deferred<string | null>()
const branches: { cwd: string; result: ReturnType<typeof deferred<string | null>> }[] = []
const api = {
  pickDirectory: () => picker.promise,
  getBranch(cwd: string) { const result = deferred<string | null>(); branches.push({ cwd, result }); return result.promise },
}
function mount() {
  const hooks = harness()
  const { useComposerWorkspace } = load('src/components/composer/useComposerWorkspace.tsx', {
    react: hooks.hooks, 'react/jsx-runtime': jsx,
    '../../lib/store': { useStore }, '../../lib/ipc': { api }, '../../lib/format': format,
    '../../lib/execution-target': execution, '../../lib/thread-workspace': workspace,
    '../../lib/terminal-sessions': { terminalSessionKey }, '../../lib/icons': { Icon: 'icon' },
    '../ui': { Button: 'button', MenuItem: 'menu-item' },
  })
  const render = () => { const value = hooks.render(() => useComposerWorkspace({ mode: 'home', settings })); hooks.flush(); return value }
  return { render, unmount: () => hooks.cleanup() }
}

let view = mount(), work = view.render()
work.overflowWorkspace.onSelectProject('alpha')
work = view.render()
assert.equal(work.workCwd, '/synthetic/alpha')
assert.equal(work.activeProjectId, 'alpha')
assert.equal(state.homeComposerWorkspace.projectId, 'alpha')

const stalePick = work.pickFolder()
work.overflowWorkspace.onSelectProject('beta')
work = view.render()
picker.resolve('/synthetic/stale-folder')
await stalePick
work = view.render()
assert.equal(work.workCwd, '/synthetic/beta', 'a late native folder picker cannot move the newly selected project')
branches.find(row => row.cwd === '/synthetic/alpha')!.result.resolve('stale-alpha-branch')
await settle()
work = view.render()
assert.equal(work.overflowWorkspace.branch, null, 'a late branch read cannot claim the current project branch')
branches.find(row => row.cwd === '/synthetic/beta')!.result.resolve('beta-branch')
await settle()
work = view.render()
assert.equal(work.overflowWorkspace.branch, 'beta-branch')

picker = deferred()
const unmountedPick = work.chooseDir()
view.unmount()
view = mount()
work = view.render()
picker.resolve('/synthetic/old-view')
await unmountedPick
work = view.render()
assert.equal(work.activeProjectId, 'beta', 'project selection survives leaving and returning to New task')
assert.equal(work.workCwd, '/synthetic/beta', 'folder selection survives remount without an old view overriding it')

picker = deferred()
const first = work.pickFolder(), firstPicker = picker
picker = deferred()
const second = work.pickFolder()
picker.resolve('/synthetic/newest')
await second
firstPicker.resolve('/synthetic/older')
await first
work = view.render()
assert.equal(work.workCwd, '/synthetic/newest', 'latest directory operation wins even before a rerender')

const hostA = { id: 'host-a', label: 'Host A' }, hostB = { id: 'host-b', label: 'Host B' }
state.sshHosts = [hostA, hostB]
state.connectSshHost = () => {}
work = view.render()
work.overflowWorkspace.onHost(hostA, '/synthetic/shared-name')
work = view.render()
picker = deferred()
const oldHostPick = work.pickFolder()
work.overflowWorkspace.onHost(hostB, '/synthetic/shared-name')
work = view.render()
picker.resolve('/synthetic/late-local-choice')
await oldHostPick
work = view.render()
assert.equal(work.workspace.hostId, 'host-b', 'a same-path host switch still changes ownership of pending operations')
assert.equal(work.workCwd, '/synthetic/shared-name')
view.unmount()
console.log('Composer project/folder retention, branch/SSH-host/unmount fencing and latest-directory selection pass')
