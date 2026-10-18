import assert from 'node:assert/strict'
import { createClusterClient } from './client'
import { find, harness, jsx, load, nodes, text, settle } from '../reading/reading-test-harness'
const calls: any[] = [], changes: any[] = [], done: any[] = []
let fail = true
const api = { saveSshHost: async (value: any) => { calls.push(value); if (fail) throw Error('Synthetic SSH metadata refusal'); return [value] } }
const client = createClusterClient({ clusterWorkspace: async () => ({ revision: 1, hosts: [], plans: [], jobs: [], deployments: [] }) })
const h = harness()
const { ClusterSshHostForm } = load('src/components/clusters/ClusterSshHostForm.tsx', { react: h.hooks, 'react/jsx-runtime': jsx, '../ui': { Button: 'button', Input: 'input' }, '../../lib/ipc': { api }, '../../lib/store': { useStore: { setState: (value: any) => changes.push(value) } } })
const render = () => h.render(() => ClusterSshHostForm({ client, busy: false, onDone: (id?: string) => done.push(id) }))
let tree = render(); const field = (label: string) => find(tree, node => node.type === 'input' && node.props['aria-label'] === label)
field('Name').props.onChange({ target: { value: 'Synthetic worker' } }); field('Hostname').props.onChange({ target: { value: 'worker.example.invalid' } }); field('Username').props.onChange({ target: { value: 'example-user' } }); field('Identity file').props.onChange({ target: { value: '/synthetic/identity' } }); tree = render()
await tree.props.onSubmit({ preventDefault() {} }); tree = render(); assert.match(client.snapshot().error, /metadata refusal/); assert.deepEqual(changes, []); assert.deepEqual(done, []); assert.equal(field('Hostname').props.value, 'worker.example.invalid')
fail = false; await tree.props.onSubmit({ preventDefault() {} })
assert.equal(calls.length, 2); assert.equal(calls[1].origin, 'manual'); assert.deepEqual(calls[1].identityFiles, ['/synthetic/identity']); assert.equal(calls[1].port, 22)
assert.deepEqual(changes, [{ sshHosts: [calls[1]] }]); assert.deepEqual(done, [calls[1].id]); assert.equal(nodes(tree).some(node => node.type === 'input' && node.props.type === 'password'), false, 'Saved SSH metadata form never collects or stores a login password')
console.log('Actual saved-SSH form uses the existing metadata API, keeps failed edits, refreshes only accepted records and never runs trust, login, key installation or model actions')
