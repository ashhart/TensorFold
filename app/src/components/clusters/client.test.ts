import assert from 'node:assert/strict'
import { createClusterClient } from './client'
import type { ClusterWorkspaceSnapshot } from '../../../shared/cluster-workspace'
const snapshot = (revision: number): ClusterWorkspaceSnapshot => ({ revision, hosts: [], plans: [], jobs: [], deployments: [] })
const pending: Array<(value: ClusterWorkspaceSnapshot) => void> = []
const client = createClusterClient({ clusterWorkspace: () => new Promise(resolve => pending.push(resolve)) })
const first = client.refresh(), second = client.refresh()
pending[1](snapshot(2)); await second; pending[0](snapshot(1)); await first
assert.equal(client.snapshot().snapshot?.revision, 2, 'A late older poll cannot replace a newer request')
const oldRead = client.refresh(); let actions = 0, release!: (value: ClusterWorkspaceSnapshot) => void
const action = client.execute('Inspect', () => { actions++; return new Promise<ClusterWorkspaceSnapshot>(resolve => { release = resolve }) })
assert.equal(await client.execute('Duplicate', async () => { actions++; return snapshot(4) }), undefined)
assert.equal(actions, 1, 'A duplicate intent is synchronously refused before another API call')
release(snapshot(4)); await action; pending[2](snapshot(3)); await oldRead
assert.equal(client.snapshot().snapshot?.revision, 4, 'An in-flight read cannot replace a completed mutation snapshot')
await client.execute('Review', async () => { throw Error('Synthetic enrollment refused') })
assert.match(client.snapshot().error, /enrollment refused/); assert.equal(client.snapshot().pending, null)
const invalid = createClusterClient({ clusterWorkspace: async () => ({ revision: NaN } as ClusterWorkspaceSnapshot) })
await invalid.refresh(); assert.match(invalid.snapshot().error, /Invalid cluster/)
let late!: (value: ClusterWorkspaceSnapshot) => void
const lateAction = client.execute('Install', () => new Promise<ClusterWorkspaceSnapshot>(resolve => { late = resolve }))
client.dispose(); late(snapshot(5)); assert.equal(await lateAction, undefined, 'A disposed workspace cannot accept a late setup response')
assert.equal(client.snapshot().snapshot?.revision, 4)
client.activateView()
const reattached = client.refresh(); pending[3](snapshot(6)); await reattached; assert.equal(client.snapshot().snapshot?.revision, 6, 'StrictMode effect cleanup/remount can reattach without freezing the workspace')
let oldRelease!: (value: ClusterWorkspaceSnapshot) => void
const oldOwner = client.execute('Old view', () => new Promise<ClusterWorkspaceSnapshot>(resolve => { oldRelease = resolve }))
client.dispose(); client.activateView(); const newRead = client.refresh(); pending[4](snapshot(8)); await newRead; oldRelease(snapshot(9)); assert.equal(await oldOwner, undefined); assert.equal(client.snapshot().snapshot?.revision, 8, 'An old view response cannot gain authority again after a new mount')
console.log('Cluster client passes out-of-order poll/mutation fences, duplicate intent, enrollment errors, malformed snapshot refusal and disposed-owner guards without retrying remote commands')
