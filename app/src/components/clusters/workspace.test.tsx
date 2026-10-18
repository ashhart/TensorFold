import assert from 'node:assert/strict'
import { createClusterClient } from './client'
import { discoveryEndpoint, newClusterCommand } from './display'
import { find, harness, jsx, load, nodes, text, settle } from '../reading/reading-test-harness'
const calls: string[] = [], blocked: boolean[] = []
let discoveryVia: 'ssh' | 'https' = 'https'
let snapshot: any = { revision: 1, hosts: [], plans: [], deployments: [], jobs: [{ id: 'old-uncertain', phase: 'needs-review', updatedAt: 1 }, ...Array.from({ length: 40 }, (_, index) => ({ id: `finished-${index}`, phase: 'completed', updatedAt: index + 2 }))] }
const api: any = { platform: 'browser', clusterWorkspace: async () => { calls.push('workspace'); return snapshot }, clusterDiscover: async () => { calls.push('discover'); return [{ id: 'candidate', label: 'Synthetic suggestion', host: 'worker.example.invalid', port: 8766, expiresAt: Date.now() + 10000, protocol: 1, trusted: false, via: discoveryVia }] }, clusterSaveHost: async (input: any) => { calls.push('save'); snapshot = { ...snapshot, revision: 2, hosts: [{ ...input, id: 'saved-host' }] }; return snapshot } }
function stableHarness() { const h = harness(); h.hooks.useMemo = (build: () => unknown, deps: unknown[]) => { const memo = h.hooks.useRef(null); if (!memo.current || deps.some((value, index) => value !== memo.current.deps[index])) memo.current = { deps, value: build() }; return memo.current.value }; return h }
const h = stableHarness()
const imports = { react: h.hooks, 'react/jsx-runtime': jsx, '../../lib/ipc': { api }, '../../lib/store': { useStore: (select: any) => select({ sshHosts: [] }) }, '../ui': { Button: 'button' }, './client': { createClusterClient }, './display': { discoveryEndpoint, newClusterCommand },
  './ClusterHostEditor': { ClusterHostEditor: 'host-editor' }, './ClusterPlanEditor': { ClusterPlanEditor: 'plan-editor' }, './ClusterSshSetup': { ClusterSshSetup: 'ssh-setup' }, './ClusterCards': { ClusterHostCard: 'host-card', ClusterPlanCard: 'plan-card', ClusterDeploymentCard: 'deployment-card', ClusterJobCard: 'job-card' } }
const { ClusterWorkspace } = load('src/components/clusters/ClusterWorkspace.tsx', imports)
const onCloseBlockedChange = (value: boolean) => blocked.push(value)
const render = () => h.render(() => ClusterWorkspace({ onCloseBlockedChange }))
let tree = render(); h.flush(); await settle(); tree = render()
assert.deepEqual(calls, ['workspace'], 'Opening workspace only reads stored state; it never executes SSH, installs, stages or starts a model')
const jobs = nodes(tree).filter(node => node.type === 'job-card')
assert.equal(jobs.length, 30); assert.equal(jobs[0].props.job.id, 'old-uncertain', 'An old uncertain receipt remains visible before recent terminal history')
const button = (value: any, label: string) => find(value, node => node.type === 'button' && text(node) === label)
await button(tree, 'Discover hosts').props.onClick(); tree = render()
assert.deepEqual(calls, ['workspace','discover'])
button(tree, 'Review').props.onClick(); tree = render(); h.flush()
const editor = find(tree, node => node.type === 'host-editor')
assert.equal(editor.props.initialUrl, 'https://worker.example.invalid:8766/')
assert.equal(calls.includes('save'), false, 'Reviewing an untrusted discovery suggestion never saves or trusts it automatically')
assert.equal(blocked.at(-1), true, 'A host editor blocks accidental drawer dismissal until Save/Cancel')
assert.equal(await editor.props.onSave({ expectedRevision: 0, label: 'Synthetic saved host', role: 'workstation', connection: { kind: 'endpoint', url: editor.props.initialUrl } }), true)
editor.props.onCancel(); tree = render(); h.flush(); assert.equal(blocked.at(-1), false)
assert.equal(nodes(tree).filter(node => node.type === 'host-card').length, 1)
discoveryVia = 'ssh'; await button(tree, 'Discover hosts').props.onClick(); tree = render(); button(tree, 'Review').props.onClick(); tree = render()
const sshEditor = find(tree, node => node.type === 'host-editor'); assert.equal(sshEditor.props.initialKind, 'ssh'); assert.equal(sshEditor.props.initialPort, 8766); assert.equal(sshEditor.props.initialUrl, undefined, 'An SSH-advertised loopback service is never changed into a made-up HTTPS URL'); assert.equal(calls.filter(value => value === 'save').length, 1, 'Discovery review still performs no implicit enrollment or trust')
sshEditor.props.onCancel(); tree = render(); h.flush()

find(tree, node => node.type === 'ssh-setup').props.onEditingChange(true); tree = render(); h.flush(); assert.equal(blocked.at(-1), true)
h.cleanup()
const remote = stableHarness(); api.platform = 'web-remote'
const remoteModule = load('src/components/clusters/ClusterWorkspace.tsx', { ...imports, react: remote.hooks })
const before = calls.length; const remoteTree = remote.render(() => remoteModule.ClusterWorkspace({})); remote.flush(); await settle()
assert.equal(calls.length, before); assert.match(text(remoteTree), /local desktop app/); remote.cleanup()
console.log('Actual cluster workspace reads only at startup, paginates terminal history while surfacing old uncertain receipts, requires explicit discovery review/save, protects dirty setup and refuses paired-remote management')
