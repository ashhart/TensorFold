import assert from 'node:assert/strict'
import { createClusterClient } from './client'
import { clusterMembersMatch, clusterPhaseLabel, newClusterCommand, shortDigest } from './display'
import { find, harness, jsx, load, nodes, text, settle } from '../reading/reading-test-harness'
const empty = { revision: 1, hosts: [], plans: [], jobs: [], deployments: [] }
const client = createClusterClient({ clusterWorkspace: async () => empty })
const calls: any[] = [], notices: any[] = []
let resolvePreview!: (value: any) => void
const api: any = { clusterBootstrapPreview: (input: any) => { calls.push(['preview', input]); return new Promise(resolve => { resolvePreview = resolve }) },
  clusterBootstrapApply: async (input: any) => { calls.push(['apply', input]); throw Error('Synthetic enrollment failure') }, clusterSshKey: async (input: any) => { calls.push(['key', input]); return { available: true, fingerprint: 'SHA256:synthetic', publicKey: 'ssh-ed25519 synthetic-public', message: 'Synthetic public identity' } }, writeClipboard: async (value: string) => { calls.push(['clipboard', value]); return { ok: true } },
  clusterEnsureModel: async (input: any) => { calls.push(['stage', input]); return empty }, clusterActivate: async (input: any) => { calls.push(['activate', input]); return empty }, clusterReconcile: async (input: any) => { calls.push(['reconcile', input]); return empty }, clusterStop: async (input: any) => { calls.push(['stop', input]); return empty }, clusterEndpointCredentials: async () => { calls.push(['credentials']); return { baseUrl: 'https://example.invalid/v1', model: 'synthetic', apiKey: 'synthetic-private-token' } }, clusterUseEndpoint: async (input: any) => { calls.push(['provider', input]); return { providerId: 'synthetic-remote' } } }
const hooks = harness()
const { ClusterSshSetup } = load('src/components/clusters/ClusterSshSetup.tsx', { react: hooks.hooks, 'react/jsx-runtime': jsx, '../ui': { Button: 'button', Input: 'input' }, '../../lib/ipc': { api }, './display': { newClusterCommand }, './ClusterSshHostForm': { ClusterSshHostForm: 'ssh-form' } })
const sshHosts: any[] = [{ id: 'ssh-one', label: 'Synthetic one' }, { id: 'ssh-two', label: 'Synthetic two' }]
const render = () => hooks.render(() => ClusterSshSetup({ hosts: sshHosts, client, busy: false }))
const button = (tree: any, label: string) => find(tree, node => node.type === 'button' && text(node) === label)
let tree = render(); assert.deepEqual(calls, [], 'Opening setup never generates keys or executes SSH')
const choose = find(tree, node => node.type === 'select'); choose.props.onChange({ target: { value: 'ssh-one' } }); tree = render()
const first = button(tree, 'Review setup').props.onClick(); tree = render(); choose.props.onChange({ target: { value: 'ssh-two' } }); render()
resolvePreview({ id: 'stale', digest: 'old', sshHostId: 'ssh-one', platform: 'linux', nodeAvailable: false, engineAvailable: false, steps: ['Synthetic review'], blockers: [], expiresAt: Date.now() + 1000 }); await first; tree = render()
assert.equal(button(tree, 'Apply reviewed setup'), undefined, 'A late preview cannot own a different selected SSH host')
const next = button(tree, 'Review setup').props.onClick(); resolvePreview({ id: 'review', digest: 'digest', sshHostId: 'ssh-two', platform: 'linux', nodeAvailable: false, engineAvailable: false, steps: ['Synthetic reviewed action'], blockers: [], expiresAt: Date.now() + 10000 }); await next; tree = render()
assert.equal(calls.filter(call => call[0] === 'apply').length, 0, 'Review alone performs no installation')
await button(tree, 'Apply reviewed setup').props.onClick(); tree = render(); assert.match(client.snapshot().error, /enrollment failure/); assert.ok(button(tree, 'Apply reviewed setup'), 'A failed setup retains the reviewed context for recovery rather than hiding the error')
assert.equal(calls.filter(call => call[0] === 'apply').length, 1)

const cardsHooks = harness()
const cards = load('src/components/clusters/ClusterCards.tsx', { react: cardsHooks.hooks, 'react/jsx-runtime': jsx, '../ui': { Button: 'button' }, '../../lib/ipc': { api }, '../../lib/engine': { formatBytes: (value: number) => `${value} bytes` }, './display': { clusterMembersMatch, clusterPhaseLabel, newClusterCommand, shortDigest } })
const host: any = { id: 'host', revision: 1, nodeId: 'node', state: 'ready' }
const plan: any = { id: 'plan', digest: 'a'.repeat(64), name: 'Synthetic plan', model: { repo: 'synthetic/model', revision: 'a'.repeat(40) }, profileId: 'single', transport: 'tcp', generation: { contextTokens: 8192, maxOutputTokens: 1024, temperature: 0.7 }, poolsWeights: false, members: [{ hostId: 'host', hostRevision: 1, nodeId: 'node', rank: 0 }], blockers: ['Model not cached'], warnings: [], admission: [], stageAllowed: true, expiresAt: Date.now() + 10000 }
const snapshot: any = { ...empty, hosts: [host] }
tree = cardsHooks.render(() => cards.ClusterPlanCard({ plan, snapshot, client, busy: false }))
assert.equal(button(tree, 'Stage model').props.disabled, false); assert.equal(button(tree, 'Start deployment').props.disabled, true, 'Missing cached weights may permit staging while still refusing model loading')
await button(tree, 'Stage model').props.onClick(); assert.equal(calls.filter(call => call[0] === 'stage').length, 1)
plan.blockers = []; plan.expiresAt = Date.now() - 1; tree = cardsHooks.render(() => cards.ClusterPlanCard({ plan, snapshot, client, busy: false })); assert.equal(button(tree, 'Start deployment').props.disabled, true)
const job: any = { id: 'restored-job', commandId: 'existing-command', kind: 'bootstrap', phase: 'needs-review', hostIds: ['host'], step: 'Awaiting receipt', message: 'Synthetic lost acknowledgement', cancelRequested: false }
tree = cardsHooks.render(() => cards.ClusterJobCard({ job, hosts: [host], client, busy: false })); assert.match(text(tree), /Reconcile before retrying/); await button(tree, 'Reconcile').props.onClick(); assert.deepEqual(calls.at(-1), ['reconcile', { jobId: 'restored-job' }])
const deployment: any = { id: 'deployment', name: 'Synthetic deployment', state: 'needs-review', endpoint: { baseUrl: 'https://example.invalid/v1', model: 'synthetic', ready: false, hasKey: true }, error: 'Synthetic uncertain receipt' }
const renderDeployment = () => cardsHooks.render(() => cards.ClusterDeploymentCard({ deployment, client, busy: false }))
tree = renderDeployment(); assert.equal(button(tree, 'Use as provider'), undefined); assert.equal(button(tree, 'Copy API key'), undefined, 'An unverified restored deployment cannot expose credentials or claim readiness')
deployment.state = 'ready'; deployment.endpoint.ready = true; tree = renderDeployment(); button(tree, 'Copy API key').props.onClick(); tree = renderDeployment()
assert.equal(calls.some(call => call[0] === 'credentials'), false, 'Showing a copy confirmation never retrieves the private credential')
await button(tree, 'Copy').props.onClick(); assert.deepEqual(calls.at(-1), ['clipboard', 'synthetic-private-token'])
assert.equal(text(renderDeployment()).includes('synthetic-private-token'), false, 'A copied credential is never kept in rendered text')
console.log('Actual cluster SSH/cards pass explicit setup review/apply, stale host preview and enrollment errors, stage-vs-load admission, expiring plans, restored standalone receipt reconciliation and gated secret-safe remote endpoints')
