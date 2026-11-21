import assert from 'node:assert/strict'
import * as executionTarget from '../../lib/execution-target'
import * as terminalLayout from '../../lib/terminal-layout'
import * as terminalSessions from '../../lib/terminal-sessions'
import { find, harness, jsx, load, settle, text } from '../reading/reading-test-harness'

const hooks = harness()
const Menu = () => null
const projectId = 'fixture-project'
const codex = { id: 'codex', name: 'Codex', command: '/fixture/bin/codex', icon: 'terminal' }
const shell = { id: 'shell', name: 'Shell', command: null, icon: 'terminal' }
const hosts = [{ id: 'alpha', name: 'Alpha', enabled: true }, { id: 'bravo', name: 'Bravo', enabled: true }]
const project = { id: projectId, executionTarget: 'orchestrator', orchestratorId: 'alpha' }
const catalogs: (string | null)[] = []
const launches: unknown[][] = []
const state = {
  settings: { remoteOrchestrators: hosts, defaultExecutionTarget: 'local', defaultOrchestratorId: 'bravo' },
  projects: [project],
  terminals: {} as Record<string, object[]>,
  activeTerminalId: {},
  loadTerminals: async () => {},
  killTerminal: async () => {},
  listTerminalClis: async () => {
    const target = executionTarget.resolveExecutionTarget(state.settings as any, state.projects[0] as any)
    catalogs.push(target.hostId)
    return target.hostId === 'bravo' ? [shell] : [codex, shell]
  },
  startTerminal: async (...args: unknown[]) => { launches.push(args) },
}
const useStore = Object.assign((select: (current: typeof state) => unknown) => select(state), { getState: () => state })
const { TerminalsTab } = load('src/components/terminals/TerminalsTab.tsx', {
  react: hooks.hooks, 'react/jsx-runtime': jsx,
  '../../lib/preferences/useAuthoredText': { useAuthoredText: () => (value: string) => value },
  '../../lib/store': { useStore }, '../../lib/execution-target': executionTarget,
  '../../lib/icons': { Icon: 'icon' }, '../../lib/terminal-sessions': terminalSessions,
  '../../lib/terminal-layout': terminalLayout, '../ui': { Menu }, './TerminalPane': { TerminalPane: 'terminal' },
})
const render = () => hooks.render(() => TerminalsTab({ projectId }))
function picker(tree: any) {
  const split = find(tree, (node) => node.type === Menu)
  if (split) return split.props.children(() => {})
  const empty = find(tree, (node) => node.type?.name === 'EmptyState')
  assert.ok(empty)
  return empty.type(empty.props)
}
const button = (tree: any, name: string) => find(tree, (node) => node.type === 'button' && text(node) === name)
render(); hooks.flush(); await settle()
button(picker(render()), 'Codex').props.onClick()
assert.deepEqual(launches, [[projectId, 'codex', undefined]], 'remote empty picker sends only the registered CLI ID')
state.terminals[projectId] = [{ id: 'shell-fixture', projectId, cliId: 'shell', title: 'Shell' }]
const alphaClick = button(picker(render()), 'Codex').props.onClick
alphaClick()
assert.deepEqual(launches[1], [projectId, 'codex', undefined], 'remote Split sends only the registered CLI ID')

state.projects = [{ ...project, orchestratorId: 'bravo' }]
alphaClick()
assert.equal(launches.length, 2, 'a stale Alpha click cannot start a CLI on Bravo')
assert.equal(button(picker(render()), 'Codex'), undefined, 'Alpha catalog disappears before Bravo responds')
hooks.flush(); await settle()
const bravoPicker = picker(render())
assert.deepEqual(catalogs, ['alpha', 'bravo'])
assert.equal(button(bravoPicker, 'Codex'), undefined)
const bravoClick = button(bravoPicker, 'Shell').props.onClick
state.projects = []
state.settings.defaultExecutionTarget = 'orchestrator'
bravoClick()
assert.equal(launches.length, 2, 'a deleted project cannot inherit the global host for a stale click')

state.projects = [{ ...project, orchestratorId: 'bravo' }]
state.settings.remoteOrchestrators = [hosts[0], { ...hosts[1], enabled: false }]
bravoClick()
assert.equal(launches.length, 2, 'an unavailable remote host cannot route a stale launch locally')
state.projects = [{ ...project, executionTarget: 'local' }]
state.settings.defaultExecutionTarget = 'local'
render(); hooks.flush(); await settle()
button(picker(render()), 'Codex').props.onClick()
assert.deepEqual(launches[2], [projectId, 'codex', { command: codex.command, label: codex.name }])
hooks.cleanup()
console.log('Terminal launch picker preserves named remote IDs, local options and host ownership')
