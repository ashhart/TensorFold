import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const remoteHostApi = readFileSync(new URL('./remote-host-api.ts', import.meta.url), 'utf8')
const remoteControl = readFileSync(new URL('./ipc/remote-control-ipc.ts', import.meta.url), 'utf8')
const store = readFileSync(new URL('../../../src/lib/store/slices/ticketsSlice.ts', import.meta.url), 'utf8')
const workspace = readFileSync(new URL('./workspace-runtime-ipc.ts', import.meta.url), 'utf8')

const start = remoteHostApi.slice(
  remoteHostApi.indexOf('async function startRemoteTicketExecutionOnThisHost'),
  remoteHostApi.indexOf('\n  async function archiveTicketOnThisHost'),
)
assert.match(start, /ticket\.runState === 'running'.*existingExecution\.runId/s, 'retry returns the existing active binding')
assert.match(start, /ticket\.runState === 'queued'.*existingThread\.timeline\.length === \(existingExecution\.timelineBaseline \?\? 0\)/s, 'retry resumes only its fresh bound thread')
assert.match(start, /createThreadOnThisHost/, 'host creates the execution thread')
assert.match(start, /thread\.timeline\.length \|\| thread\.apiMessages\.length/, 'historical thread output is rejected')
assert.match(start, /timelineBaseline: thread\.timeline\.length/, 'begin freezes a host-owned evidence baseline')
assert.match(start, /runId: started\.runId/, 'host-issued run id is persisted before success')

const work = store.slice(store.indexOf('async workTicketRemote'), store.indexOf('\n  applyTicketUpdate', store.indexOf('async workTicketRemote')))
assert.match(work, /startRemoteOrchestratorTicketExecution/, 'renderer uses the atomic host operation')
assert.doesNotMatch(work, /createRemoteOrchestratorThread|transitionRemoteOrchestratorTicket\([^)]*begin|startRemoteOrchestratorThread/, 'renderer cannot recreate the split create-bind-start window')

const coreStart = remoteControl.slice(remoteControl.indexOf("handle('remote:startTicketExecution'"), remoteControl.indexOf("handle('remote:updateTicket'"))
assert.match(coreStart, /hostId: host\.id/, 'controller ignores caller host identity and stamps the selected profile')

const localCreate = workspace.slice(workspace.indexOf("handle('tickets:create'"), workspace.indexOf("handle('tickets:update'"))
const remoteCreate = remoteHostApi.slice(remoteHostApi.indexOf('async function createTicketOnThisHost'), remoteHostApi.indexOf('async function updateTicketOnThisHost'))
assert.match(localCreate, /pickTicketCreateInput/, 'local public creation strips runtime-owned ticket fields')
assert.match(remoteCreate, /pickTicketCreateInput/, 'remote public creation strips runtime-owned ticket fields')

console.log('remote-ticket-start.test.ts passed')
