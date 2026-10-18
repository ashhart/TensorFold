import assert from 'node:assert/strict'
import { commonProfiles, discoveryEndpoint, newClusterCommand } from './display'
import { find, harness, jsx, load, nodes, text, settle } from '../reading/reading-test-harness'
import type { ClusterHost } from '../../../shared/cluster-workspace'
const host = (id: string, implemented = true): ClusterHost => ({ id, revision: 2, label: `Synthetic ${id}`, role: 'headless', connection: { kind: 'endpoint', url: 'https://example.invalid' }, nodeId: `node-${id}`, hasCredential: true, state: 'ready', lastError: null, createdAt: 1, updatedAt: 2,
  snapshot: { protocol: 1, nodeId: `node-${id}`, version: 'test', platform: 'darwin', arch: 'arm64', observedAt: 1, totalMemoryBytes: 100, freeMemoryBytes: 80,
    engine: { available: true, version: 'test', reason: implemented ? null : 'Generic distributed launch is unsupported.', profiles: implemented ? [{ id: 'native-test', title: 'Synthetic implemented profile', mode: 'single', platform: 'darwin', arch: 'arm64', ranks: 1, transports: ['tcp'], modelTypes: ['synthetic'], poolsWeights: false, requiresFabric: false, engineVersion: 'test' }] : [] }, models: [], links: [], active: null } })
const hooks = harness(), saved: any[] = [], cancelled: any[] = []
let busy = false, succeeds = false, release: ((value: boolean) => void) | null = null
const { ClusterHostEditor } = load('src/components/clusters/ClusterHostEditor.tsx', { react: hooks.hooks, 'react/jsx-runtime': jsx, '../ui': { Button: 'button', Input: 'input' } })
const render = () => hooks.render(() => ClusterHostEditor({ sshHosts: [], busy, onCancel: () => cancelled.push(true), onSave: (value: any) => { saved.push(value); return release === null ? Promise.resolve(succeeds) : new Promise(resolve => { release = resolve }) } }))
const field = (tree: any, label: string) => find(tree, node => node.type === 'input' && node.props['aria-label'] === label)
let tree = render(); field(tree, 'Name').props.onChange({ target: { value: 'Original host' } }); field(tree, 'Management URL').props.onChange({ target: { value: 'https://example.invalid' } }); field(tree, 'Management token').props.onChange({ target: { value: 'synthetic-private-token' } }); tree = render()
assert.equal(field(tree, 'Management token').props.type, 'password')
await tree.props.onSubmit({ preventDefault() {} }); tree = render(); assert.equal(cancelled.length, 0, 'A failed save retains the editor for review')
assert.equal(field(tree, 'Name').props.value, 'Original host'); assert.equal(field(tree, 'Management token').props.value, 'synthetic-private-token')
succeeds = true; await tree.props.onSubmit({ preventDefault() {} }); tree = render(); assert.equal(cancelled.length, 1); assert.equal(field(tree, 'Management token').props.value, '')
assert.equal(saved[0].expectedRevision, 0); assert.equal(saved[0].role, 'workstation'); assert.deepEqual(saved[0].connection, { kind: 'endpoint', url: 'https://example.invalid' })

const sshEditorHooks = harness(), sshSaves: any[] = []
const sshEditor = load('src/components/clusters/ClusterHostEditor.tsx', { react: sshEditorHooks.hooks, 'react/jsx-runtime': jsx, '../ui': { Button: 'button', Input: 'input' } }).ClusterHostEditor
const renderSsh = () => sshEditorHooks.render(() => sshEditor({ initialKind: 'ssh', initialPort: 8767, initialSshHostId: 'saved-ssh', sshHosts: [{ id: 'saved-ssh', label: 'Synthetic SSH' }], busy: false, onCancel() {}, onSave: async (input: any) => { sshSaves.push(input); return true } }))
let sshTree = renderSsh(); assert.equal(field(sshTree, 'Management URL'), undefined, 'An SSH-only suggestion never invents an HTTPS endpoint')
assert.equal(field(sshTree, 'Management token').props.type, 'password', 'Manual SSH enrollment has the same protected management-token field')
field(sshTree, 'Name').props.onChange({ target: { value: 'Synthetic SSH enrollment' } }); field(sshTree, 'Management token').props.onChange({ target: { value: 'a'.repeat(64) } }); sshTree = renderSsh(); await sshTree.props.onSubmit({ preventDefault() {} })
assert.deepEqual(sshSaves[0].connection, { kind: 'ssh', sshHostId: 'saved-ssh', port: 8767 }); assert.equal(sshSaves[0].token, 'a'.repeat(64))

const plannerHooks = harness(), plans: any[] = []
const { ClusterPlanEditor } = load('src/components/clusters/ClusterPlanEditor.tsx', { react: plannerHooks.hooks, 'react/jsx-runtime': jsx, '../ui': { Button: 'button', Input: 'input' }, '../OriginalDiagnostic': { OriginalDiagnostic: ({ children }: any) => children }, './display': { commonProfiles } })
let members = [host('one', false)]
const planner = () => plannerHooks.render(() => ClusterPlanEditor({ snapshot: { revision: 1, hosts: members, plans: [], jobs: [], deployments: [] }, busy: false, onCancel() {}, onPlan: async (input: any) => { plans.push(input); return false } }))
tree = planner(); const check = find(tree, node => node.type === 'input' && node.props.type === 'checkbox'); check.props.onChange({ target: { checked: true } }); tree = planner()
assert.match(text(tree), /No implemented execution profile/); assert.match(text(tree), /unsupported/)
assert.equal(find(tree, node => node.type === 'button' && text(node) === 'Review plan').props.disabled, true)
await tree.props.onSubmit({ preventDefault() {} }); assert.equal(plans.length, 0, 'Generic engine clustering cannot queue an unsupported model job')
members = [host('one')]; tree = planner()
field(tree, 'Name').props.onChange({ target: { value: 'Original plan' } }); field(tree, 'Model repository').props.onChange({ target: { value: 'synthetic/model' } }); field(tree, 'Immutable model revision').props.onChange({ target: { value: 'a'.repeat(40) } })
find(tree, node => node.type === 'select' && node.props['aria-label'] === 'Execution profile').props.onChange({ target: { value: 'native-test' } }); tree = planner()
assert.equal(find(tree, node => node.type === 'button' && text(node) === 'Review plan').props.disabled, false)
await tree.props.onSubmit({ preventDefault() {} }); assert.equal(plans.length, 1)
assert.deepEqual(plans[0].members, [{ hostId: 'one', hostRevision: 2, nodeId: 'node-one', rank: 0 }]); assert.equal(plans[0].model.revision, 'a'.repeat(40))
assert.equal(commonProfiles([host('one', false)]).length, 0)
assert.equal(discoveryEndpoint('::1', 8766), 'https://[::1]:8766/')
for (const unsafe of ['-oProxyCommand=x', 'https://example.invalid', 'host/path', 'host@other', 'host?x']) assert.equal(discoveryEndpoint(unsafe, 8766), '', 'Untrusted discovery strings are not executable or blindly accepted as URL components')
console.log('Actual host/plan forms preserve failed edits and password handling, require explicit save, refuse unsupported generic profiles and bind immutable model/member revisions; discovery suggestions remain untrusted')
