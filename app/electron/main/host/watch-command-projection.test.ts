import assert from 'node:assert/strict'
import {readFileSync} from 'node:fs'
import {store} from '../state'
import {createWatchCommandProjection} from './watch-command-projection'
import type {WatchSnapshot} from '../../../shared/watch-relay'
import type {Thread} from '../../agent/types'

const original = {...store}
const thread = (id = 'shared-task'): Thread => ({id, title: 'Current task', kind: 'task', cwd: '', providerId: 'example', model: 'example', createdAt: 1, updatedAt: 200, status: 'idle', apiMessages: [], timeline: [{id: 'current-assistant', kind: 'assistant', text: 'CURRENT_TRANSCRIPT'}]})
const input: WatchSnapshot = {version: 1, session: 'paired-session', generatedAt: 10, emittedAt: 10, connected: true, serverName: 'Earlier host', jobs: [{id: 'shared-task', title: 'OLD_TITLE', state: 'finished', updatedAt: 1, canReply: true, latestReply: 'OLD_TRANSCRIPT', latestReplyId: 'old-assistant'}], running: 9, queued: 9, approvals: 9, transcriptReview: true, error: null}
let role = 'owner', active: string[] = []
const project = createWatchCommandProjection({now: () => 250, activeThreadIds: () => active, environment: () => ({VONTRA_SECURITY_ROLE: role})})
try {
  store.settings = {remoteControl: {serverName: 'Current host'}} as any
  store.threads = [thread()]; store.jobs = []; store.connectorEvents = []; store.missions = []
  const first = project(input, 'paired-session')
  assert.equal(first.generatedAt, 250); assert.equal(first.emittedAt, 250)
  assert.equal(first.serverName, 'Current host'); assert.equal(first.jobs[0].title, 'Current task')
  assert.equal(first.jobs[0].latestReplyId, 'current-assistant'); assert.equal(first.jobs[0].updatedAt, 200)
  assert.equal(first.jobs[0].latestReply, null); assert.equal(first.transcriptReview, false, 'paired status never gains owner transcript scope')
  assert.equal(first.jobs[0].canReply, true, 'null engine in role clamp does not erase already verified readiness')
  role = 'viewer'; assert.equal(project(input, 'paired-session').jobs[0].canReply, false)
  role = 'operator'; active = ['shared-task']; assert.equal(project(input, 'paired-session').jobs[0].state, 'running')
  store.threads[0].timeline.push({id: 'approval-tool', kind: 'tool', state: 'awaiting'} as any)
  assert.equal(project(input, 'paired-session').jobs[0].state, 'waiting')
  for (const patch of [{incognito: true}, {phoneLocalOnly: true}, {archivedAt: 1}, {kind: 'chat'}]) {
    store.threads = [{...thread(), ...patch} as Thread]
    assert.equal(project(input, 'paired-session').jobs.length, 0, JSON.stringify(patch))
  }
  store.threads = []; assert.equal(project(input, 'paired-session').jobs.length, 0, 'deleted task disappears')
  store.threads = [thread(), {...thread('private-task'), phoneLocalOnly: true}]
  store.jobs = [
    {id: 'shared-work', kind: 'thread', owner: 'shared-task', threadId: 'shared-task', currentThreadId: null, status: 'running', pendingApprovals: [{callId: 'shared-approval'}]},
    {id: 'private-work', kind: 'thread', owner: 'private-task', threadId: 'private-task', currentThreadId: null, status: 'queued', pendingApprovals: [{callId: 'private-approval'}]},
    {id: 'unknown-work', kind: 'thread', owner: 'unknown', threadId: 'unknown', currentThreadId: null, status: 'queued', pendingApprovals: [{callId: 'unknown-approval'}]},
  ] as any
  const counts = project(input, 'paired-session')
  assert.equal(counts.running, 1); assert.equal(counts.queued, 0); assert.equal(counts.approvals, 1)
  assert.throws(() => project(input, 'other-session'), /session/)
  const main = readFileSync(new URL('../composition/connections.ts', import.meta.url), 'utf8')
  assert.match(main, /projectWatchSnapshot:\s*createWatchCommandProjection\(\{\s*activeThreadIds:\s*\(\)\s*=>\s*\[\.\.\.aborters\.keys\(\)\]\s*\}\)/)
  const runtime = readFileSync(new URL('./mobile-bridge-runtime.ts', import.meta.url), 'utf8')
  assert.match(runtime, /projectWatchSnapshot:\s*deps\.projectWatchSnapshot/)
  console.log('Final synchronous watch projection uses current privacy, current role, current assistant/state and shared work counts; production bootstrap wires guard')
} finally { Object.assign(store, original) }
