import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

function source(name: string): string {
  return readFileSync(fileURLToPath(new URL(name, import.meta.url)), 'utf8')
}

function handler(body: string, channel: string, next: string): string {
  const start = body.indexOf(`handle('${channel}'`)
  const end = body.indexOf(`handle('${next}'`, start)
  assert.ok(start >= 0 && end > start, `${channel} handler remains discoverable`)
  return body.slice(start, end)
}

const core = source('./core-ipc.ts')
const remoteControl = source('./ipc/remote-control-ipc.ts')
const workspaceIpc = source('./ipc/workspace-ipc.ts')
const remoteProjectCreate = handler(remoteControl, 'remote:createProject', 'remote:updateProject')
assert.match(remoteProjectCreate, /createdAt: _createdAt/, 'controller timestamps are stripped before remote project creation')
assert.match(remoteProjectCreate, /executionTarget: 'local', orchestratorId: null/, 'remote hosts persist projects as local-owned rather than controller-profile-owned')
const threadDelete = handler(workspaceIpc, 'threads:delete', 'threads:rename')
assert.match(threadDelete, /await archiveEntity\('thread'/, 'legacy thread delete delegates to recoverable archive')
assert.doesNotMatch(threadDelete, /store\.threads\s*=|fs\.rm|journalPath/, 'thread delete neither splices primary state nor removes its transcript journal')

const workspace = source('./workspace-runtime-ipc.ts')
const projectCreate = handler(workspace, 'projects:create', 'projects:update')
assert.match(projectCreate, /sanitizeProjectCreateInput/, 'new local projects pass through the strict creation boundary')
assert.match(projectCreate, /validateProjectExecutionTargetInput/, 'new local project routing validates the target and enabled host')
assert.match(projectCreate, /if \(existing\)[\s\S]*return existing/, 'project creation is idempotent for an existing identity')
assert.doesNotMatch(projectCreate, /Object\.assign\(existing/, 'projects:create cannot bypass strict project update validation by upserting an existing id')
for (const [channel, next, kind] of [
  ['projects:delete', 'projects:rename', 'project'],
  ['rooms:delete', 'rooms:rename', 'room'],
  ['tickets:delete', 'tickets:generate', 'ticket'],
] as const) {
  const body = handler(workspace, channel, next)
  assert.match(body, new RegExp(`await archiveEntity\\('${kind}'`), `${channel} delegates to recoverable archive`)
  assert.doesNotMatch(body, /store\.(?:projects|threads|rooms|tickets)\s*=.*filter/, `${channel} does not splice primary records`)
}
assert.match(workspace, /handle\('tickets:get'[\s\S]*?store\.tickets\.filter\(isActive\)/, 'archived tickets stay persisted but leave the active board')

const projectUpdate = handler(workspace, 'projects:update', 'projects:delete')
assert.match(projectUpdate, /p\.archivedAt/, 'archived projects reject ordinary updates')
assert.match(projectUpdate, /sanitizeProjectUpdatePatch\(patch\)/, 'renderer project patches use a strict own-key allowlist for config-only fields')
const projectRoute = handler(workspace, 'projects:setExecutionTarget', 'projects:delete')
assert.match(projectRoute, /validateProjectExecutionTargetInput/, 'dedicated project routing validates target enum and enabled host through the tested policy')
const roomCreate = handler(workspace, 'rooms:create', 'rooms:update')
assert.match(roomCreate, /!project \|\| project\.archivedAt/, 'room creation requires an active parent project')
const roomUpdate = handler(workspace, 'rooms:update', 'rooms:delete')
assert.match(roomUpdate, /r\.archivedAt/, 'archived rooms reject updates')
assert.match(roomUpdate, /pickRoomConfigPatch\(requestedPatch\)/, 'renderer room patches use a strict config-only own-key allowlist')
assert.match(roomUpdate, /assertRoomConfigUpdateAllowed/, 'room config cannot change during a live room run')
const ticketCreate = handler(workspace, 'tickets:create', 'tickets:update')
assert.match(ticketCreate, /assertTicketRelations\(store, t\.projectId/, 'ticket creation enforces active same-project relationship ownership')
const ticketUpdate = handler(workspace, 'tickets:update', 'tickets:delete')
assert.match(ticketUpdate, /t\.archivedAt/, 'archived tickets reject updates')
assert.match(ticketUpdate, /assertTicketRelations\(store, next\.projectId/, 'ticket moves enforce active same-project relationship ownership')
assert.match(ticketUpdate, /pickTicketUserPatch\(requestedPatch\)/, 'renderer ticket patches use a strict user-field own-key allowlist')
assert.match(ticketUpdate, /assertTicketUserUpdateAllowed/, 'ticket configuration cannot change during a live worker run')
const ticketGenerate = handler(workspace, 'tickets:generate', 'ticket:work')
assert.match(ticketGenerate, /!project \|\| project\.archivedAt/, 'ticket generation requires an active project')

const threadCreate = handler(workspaceIpc, 'threads:create', 'threads:delete')
assert.match(threadCreate, /projectId && \(!project \|\| project\.archivedAt\)/, 'thread creation requires an active parent project')
const threadPatch = handler(workspaceIpc, 'threads:patch', 'threads:setWorkspace')
assert.match(threadPatch, /t\?\.archivedAt/, 'archived threads reject renderer planning patches')
const threadNotice = handler(workspaceIpc, 'threads:appendNotice', 'thread:compact')
assert.match(threadNotice, /t\.archivedAt/, 'archived threads reject timeline annotations')
const threadCompact = handler(workspaceIpc, 'thread:compact', 'threads:setProject')
assert.match(threadCompact, /thread\.archivedAt/, 'archived threads reject transcript compaction')

const projectRename = handler(workspace, 'projects:rename', 'rooms:get')
assert.match(projectRename, /p\?\.archivedAt/, 'archived projects reject rename mutations')
const roomRenameStart = workspace.indexOf("handle('rooms:rename'")
const roomRenameEnd = workspace.indexOf('\nfunction cancelRoomRun', roomRenameStart)
assert.ok(roomRenameStart >= 0 && roomRenameEnd > roomRenameStart)
assert.match(workspace.slice(roomRenameStart, roomRenameEnd), /r\?\.archivedAt/, 'archived rooms reject rename mutations')

const handoffIpc = source('./handoff-ipc.ts')
const handoffStart = handoffIpc.indexOf("handle('thread:handoff'")
const handoffEnd = handoffIpc.length
assert.ok(handoffStart >= 0, 'thread handoff handler remains discoverable')
const handoff = handoffIpc.slice(handoffStart, handoffEnd)
assert.match(handoff, /source\.archivedAt/, 'handoff requires an active source thread')
assert.match(handoff, /!sourceProject \|\| sourceProject\.archivedAt/, 'handoff requires an active parent project')
assert.match(handoff, /assertSshThreadTransportSupported\(sourceWorkspace, toProvider/, 'handoff rejects delegated local engines on SSH before persistence')
assert.ok(handoff.indexOf('assertSshThreadTransportSupported') < handoff.indexOf('store.threads.unshift'), 'handoff transport preflight precedes new-thread persistence')

const archive = source('./archive.ts')
const archiveState = source('./archive-state.ts')
assert.match(archive, /tickets:\s*store\.tickets\.filter/, 'archive snapshot exposes recoverable tickets')
assert.match(archiveState, /thread\.archivedAt == null/, 're-archiving a thread preserves its existing batch marker')
assert.match(archive, /thread\?\.projectId === ticket\.projectId\)[\s\S]{0,120}controls\.abortThread\?\.\(thread\.id\)/, 'ticket archive stops only its same-project active worker thread')
assert.match(archive, /controls\.publish\?\.\('ticket'/, 'archive publishes ticket tombstones to connected projections')
assert.match(archive, /Restore the active parent project before restoring linked tickets/, 'ticket restore cannot create an active orphan under archived parents')
assert.match(archive, /Restore the active linked thread in the same project/, 'ticket restore cannot reactivate against an archived or cross-project linked thread')

const threadRuntime = source('./thread-runtime.ts')
const threadStart = threadRuntime.slice(threadRuntime.indexOf('async function startThreadRun'), threadRuntime.indexOf('\n  function applyEventToThread'))
assert.match(threadStart, /thread\.archivedAt/, 'thread start rejects archived records')
assert.match(threadStart, /project\.archivedAt/, 'thread start rejects an archived parent project')
assert.ok(threadStart.indexOf('reserveStart') < threadStart.indexOf('aborters.set'), 'thread run lease is acquired before its aborter/start reservation')

const ticketWorker = source('./ticket-worker-runtime.ts')
const workerStart = ticketWorker.slice(ticketWorker.indexOf('async function startTicketWork'), ticketWorker.indexOf('\n  return { startTicketWork }'))
assert.match(ticketWorker, /ticket\.archivedAt/, 'ticket worker rejects archived tickets')
assert.ok(workerStart.lastIndexOf('inspectGitRepository') < workerStart.indexOf('ticket.status ='), 'all Git inspection awaits finish before ticket running state is committed')
assert.ok(workerStart.indexOf('runLease.addWorkspaceKeys') < workerStart.indexOf('store.threads.unshift'), 'final-key lease conflicts fail before a new thread is persisted')

const roomStart = workspace.slice(workspace.indexOf('async function startRoomRun'), workspace.indexOf("handle('room:cancel'"))
assert.match(roomStart, /Local room execution does not support SSH workspaces/, 'local room startup fails closed before SSH paths reach local Git')
assert.ok(roomStart.indexOf('reserveStart') < roomStart.indexOf('ensureRoomWorktree'), 'room startup reserves its workspace before asynchronous worktree setup')
assert.match(roomStart, /if \(!startupComplete\)[\s\S]*runLease\.release/, 'room prelaunch failures release the startup reservation')

const instructionWriter = workspace.slice(workspace.indexOf('async function withProjectInstructionLease'), workspace.indexOf("handle('projects:get'"))
assert.ok(instructionWriter.indexOf('workspaceMutationGuard.reserveStart') < instructionWriter.indexOf('writeIfMissing'), 'project guidance writes reserve their local workspace before touching AGENTS.md')
assert.match(instructionWriter, /canonicalLocalWorkspaceKey\(project\.cwd\)/, 'project guidance adds the canonical workspace identity before writing')
assert.match(instructionWriter, /finally[\s\S]*lease\.release/, 'project guidance reservation releases on every write outcome')

const localPr = source('./local-pr-runtime.ts')
const createPr = localPr.slice(localPr.indexOf('async function createLocalPrLoop'), localPr.indexOf('function resolvePrLoopProvider'))
assert.ok(createPr.indexOf('workspaceMutationGuard.reserveStart') < createPr.indexOf('inspectGitRepository'), 'local PR creation reserves the source repo before Git inspection/worktree mutation')
assert.ok(createPr.indexOf('lexicalLocalWorkspaceKey(worktree)') < createPr.indexOf('worktree add'), 'local PR creation reserves the destination before git worktree add')
assert.match(createPr, /finally[\s\S]*worktreeLease\.release/, 'local PR worktree lease spans the complete durable creation flow')

console.log('local legacy deletes retain primary data and route through recoverable archives')
