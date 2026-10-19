import type { IncomingMessage, ServerResponse } from 'node:http'
import { canonicalHomeDirectory, resolvePairedWorkspace, type RemoteWorkspaceScope } from './local-workspace-policy'
import { normalizeAttachments } from './attachment-policy'
import { parseTerminalStart, parseTerminalWrite, parseTerminalResize, parseTerminalRename } from './terminal-forward'
import { readJson, writeJson, writeArchiveMutation } from './remote-control-http'
import type { RemoteControlServerDeps } from './remote-control-types'
import type { MissionStatus, Project, Room, RoomManualMemoryCommand, Ticket, TicketRemoteLifecycleCommand } from './types'

const MISSION_STATUSES = new Set<MissionStatus>(['draft', 'queued', 'running', 'waiting_approval', 'paused', 'stalled', 'completed', 'failed', 'cancelled'])

interface RemoteMutationContext {
  deps: RemoteControlServerDeps
  req: IncomingMessage
  res: ServerResponse
  url: URL
  ownerAuthed: boolean
  workspaceScope: RemoteWorkspaceScope
}

export async function isHomeConfinedPath(value: string): Promise<boolean> {
  try { await canonicalHomeDirectory(value.trim(), undefined, false); return true } catch { return false }
}

function isMissionStatus(input: string): input is MissionStatus {
  return MISSION_STATUSES.has(input as MissionStatus)
}

function rejectsArchiveStateMutation(body: Record<string, unknown>, update = false): string | null {
  if (Object.prototype.hasOwnProperty.call(body, 'archivedAt')) return 'archive_state_requires_archive_or_restore_route'
  if (update && Object.prototype.hasOwnProperty.call(body, 'id')) return 'record_id_is_immutable'
  return null
}


export async function handleRemoteMutation(context: RemoteMutationContext): Promise<void> {
  const { deps, req, res, url, ownerAuthed, workspaceScope } = context
  const body = await readJson(req)
  if (url.pathname === '/v1/projects') {
    const protectedFieldError = rejectsArchiveStateMutation(body)
    if (protectedFieldError) { writeJson(res, 400, { error: protectedFieldError }); return }
    const name = String(body.name || '').trim()
    if (!name) { writeJson(res, 400, { error: 'name_required' }); return }
    if (!ownerAuthed) {
      const requestedCwd = typeof body.cwd === 'string' ? body.cwd : ''
      try {
        const canonical = await resolvePairedWorkspace(requestedCwd, body.workspace as Project['workspace'])
        body.cwd = canonical
        if (body.workspace) body.workspace = { kind: 'local', path: canonical }
      } catch {
        writeJson(res, 403, { error: 'paired_sessions_are_limited_to_home_workspaces' })
        return
      }
    }
    writeJson(res, 201, await deps.createProject(body as Partial<Project> & { name?: string; cwd?: string }, workspaceScope))
    return
  }
  const projectUpdate = /^\/v1\/projects\/([^/]+)\/update$/.exec(url.pathname)
  if (projectUpdate) {
    const protectedFieldError = rejectsArchiveStateMutation(body, true)
    if (protectedFieldError) { writeJson(res, 400, { error: protectedFieldError }); return }
    writeJson(res, 202, await deps.updateProject(decodeURIComponent(projectUpdate[1]), body as Partial<Project>))
    return
  }
  const projectDelete = /^\/v1\/projects\/([^/]+)\/delete$/.exec(url.pathname)
  if (projectDelete) {
    await writeArchiveMutation(res, deps.deleteProject(decodeURIComponent(projectDelete[1])))
    return
  }
  const projectRestore = /^\/v1\/projects\/([^/]+)\/restore$/.exec(url.pathname)
  if (projectRestore) {
    await writeArchiveMutation(res, deps.restoreProject(decodeURIComponent(projectRestore[1])))
    return
  }
  if (url.pathname === '/v1/threads') {
    const protectedFieldError = rejectsArchiveStateMutation(body)
    if (protectedFieldError) { writeJson(res, 400, { error: protectedFieldError }); return }
    const kind = body.kind === 'chat' ? 'chat' : 'task'
    const requestedCwd = typeof body.cwd === 'string' ? body.cwd : ''
    if (!ownerAuthed && kind !== 'chat' && requestedCwd && !await isHomeConfinedPath(requestedCwd)) {
      writeJson(res, 403, { error: 'paired_sessions_are_limited_to_home_workspaces' })
      return
    }
    writeJson(res, 201, await deps.createThread({
      cwd: requestedCwd,
      kind,
      projectId: typeof body.projectId === 'string' ? body.projectId : null,
    }, workspaceScope))
    return
  }
  if (url.pathname === '/v1/rooms') {
    const protectedFieldError = rejectsArchiveStateMutation(body)
    if (protectedFieldError) { writeJson(res, 400, { error: protectedFieldError }); return }
    const projectId = String(body.projectId || '').trim()
    if (!projectId) { writeJson(res, 400, { error: 'projectId_required' }); return }
    writeJson(res, 201, await deps.createRoom({ projectId, name: typeof body.name === 'string' ? body.name : undefined }))
    return
  }
  if (url.pathname === '/v1/tickets') {
    const protectedFieldError = rejectsArchiveStateMutation(body)
    if (protectedFieldError) { writeJson(res, 400, { error: protectedFieldError }); return }
    const projectId = String(body.projectId || '').trim()
    if (!projectId) { writeJson(res, 400, { error: 'projectId_required' }); return }
    writeJson(res, 201, await deps.createTicket(body as Partial<Ticket> & { projectId: string }))
    return
  }
  if (url.pathname === '/v1/terminals') {
    if (!ownerAuthed) { writeJson(res, 403, { error: 'owner_token_required' }); return }
    if (!deps.startTerminal) { writeJson(res, 501, { error: 'terminals_unavailable' }); return }
    const parsed = parseTerminalStart(body)
    if ('error' in parsed) { writeJson(res, 400, { error: parsed.error }); return }
    if (parsed.command || parsed.cliId.startsWith('custom:')) { writeJson(res, 403, { error: 'custom_terminal_commands_unavailable' }); return }
    writeJson(res, 201, deps.startTerminal(parsed))
    return
  }
  const terminalWrite = /^\/v1\/terminals\/([^/]+)\/write$/.exec(url.pathname)
  if (terminalWrite) {
    if (!ownerAuthed) { writeJson(res, 403, { error: 'owner_token_required' }); return }
    if (!deps.writeTerminal) { writeJson(res, 501, { error: 'terminals_unavailable' }); return }
    const parsed = parseTerminalWrite(body)
    if ('error' in parsed) { writeJson(res, 400, { error: parsed.error }); return }
    deps.writeTerminal(decodeURIComponent(terminalWrite[1]), parsed.data)
    writeJson(res, 202, { ok: true })
    return
  }
  const terminalResize = /^\/v1\/terminals\/([^/]+)\/resize$/.exec(url.pathname)
  if (terminalResize) {
    if (!ownerAuthed) { writeJson(res, 403, { error: 'owner_token_required' }); return }
    if (!deps.resizeTerminal) { writeJson(res, 501, { error: 'terminals_unavailable' }); return }
    const parsed = parseTerminalResize(body)
    if ('error' in parsed) { writeJson(res, 400, { error: parsed.error }); return }
    deps.resizeTerminal(decodeURIComponent(terminalResize[1]), parsed.cols, parsed.rows)
    writeJson(res, 202, { ok: true })
    return
  }
  const terminalRename = /^\/v1\/terminals\/([^/]+)\/rename$/.exec(url.pathname)
  if (terminalRename) {
    if (!deps.renameTerminal) { writeJson(res, 501, { error: 'terminals_unavailable' }); return }
    const parsed = parseTerminalRename(body)
    if ('error' in parsed) { writeJson(res, 400, { error: parsed.error }); return }
    deps.renameTerminal(decodeURIComponent(terminalRename[1]), parsed.title)
    writeJson(res, 202, { ok: true })
    return
  }
  const terminalKill = /^\/v1\/terminals\/([^/]+)\/kill$/.exec(url.pathname)
  if (terminalKill) {
    if (!deps.killTerminal) { writeJson(res, 501, { error: 'terminals_unavailable' }); return }
    deps.killTerminal(decodeURIComponent(terminalKill[1]))
    writeJson(res, 202, { ok: true })
    return
  }
  const approvalResolve = /^\/v1\/approvals\/([^/]+)$/.exec(url.pathname)
  if (approvalResolve) {
    if (!deps.approve) { writeJson(res, 501, { error: 'approvals_unavailable' }); return }
    const decision = String(body.decision || '').trim()
    if (decision !== 'allow' && decision !== 'allow_always' && decision !== 'deny') { writeJson(res, 400, { error: 'decision_invalid' }); return }
    await deps.approve(decodeURIComponent(approvalResolve[1]), decision)
    writeJson(res, 202, { ok: true })
    return
  }
  const threadStart = /^\/v1\/threads\/([^/]+)\/start$/.exec(url.pathname)
  if (threadStart) {
    const userText = String(body.userText || '').trim()
    if (!userText) { writeJson(res, 400, { error: 'userText_required' }); return }
    const attachments = normalizeAttachments(body.attachments)
    writeJson(res, 202, await deps.startThread({
      threadId: decodeURIComponent(threadStart[1]),
      userText,
      attachments,
      sourceThreadKey: typeof body.sourceThreadKey === 'string' ? body.sourceThreadKey : undefined,
    }, workspaceScope))
    return
  }
  const threadGuide = /^\/v1\/threads\/([^/]+)\/guide$/.exec(url.pathname)
  if (threadGuide) {
    const userText = String(body.userText || '').trim()
    if (!userText) { writeJson(res, 400, { error: 'userText_required' }); return }
    const attachments = normalizeAttachments(body.attachments)
    writeJson(res, 202, await deps.guideThread({
      threadId: decodeURIComponent(threadGuide[1]),
      userText,
      attachments,
    }, workspaceScope))
    return
  }
  const threadCancel = /^\/v1\/threads\/([^/]+)\/cancel$/.exec(url.pathname)
  if (threadCancel) {
    await deps.cancelThread(decodeURIComponent(threadCancel[1]))
    writeJson(res, 202, { ok: true })
    return
  }
  const threadArchive = /^\/v1\/threads\/([^/]+)\/archive$/.exec(url.pathname)
  if (threadArchive) {
    await writeArchiveMutation(res, deps.deleteThread(decodeURIComponent(threadArchive[1])))
    return
  }
  const threadRestore = /^\/v1\/threads\/([^/]+)\/restore$/.exec(url.pathname)
  if (threadRestore) {
    await writeArchiveMutation(res, deps.restoreThread(decodeURIComponent(threadRestore[1])))
    return
  }
  const roomStart = /^\/v1\/rooms\/([^/]+)\/start$/.exec(url.pathname)
  if (roomStart) {
    const problem = String(body.problem || '').trim()
    if (!problem) { writeJson(res, 400, { error: 'problem_required' }); return }
    const attachments = normalizeAttachments(body.attachments)
    writeJson(res, 202, await deps.startRoom(decodeURIComponent(roomStart[1]), problem, attachments, workspaceScope))
    return
  }
  const roomSteer = /^\/v1\/rooms\/([^/]+)\/steer$/.exec(url.pathname)
  if (roomSteer) {
    const text = typeof body.text === 'string' ? body.text : ''
    if (!text.trim()) { writeJson(res, 400, { error: 'text_required' }); return }
    writeJson(res, 202, await deps.steerRoom(decodeURIComponent(roomSteer[1]), text, workspaceScope))
    return
  }
  const roomRename = /^\/v1\/rooms\/([^/]+)\/rename$/.exec(url.pathname)
  if (roomRename) {
    const name = typeof body.name === 'string' ? body.name.trim() : ''
    if (!name) { writeJson(res, 400, { error: 'name_required' }); return }
    writeJson(res, 202, await deps.renameRoom(decodeURIComponent(roomRename[1]), name))
    return
  }
  const roomMemory = /^\/v1\/rooms\/([^/]+)\/memory$/.exec(url.pathname)
  if (roomMemory) {
    writeJson(res, 202, await deps.mutateRoomMemory(decodeURIComponent(roomMemory[1]), body as RoomManualMemoryCommand))
    return
  }
  const roomUpdate = /^\/v1\/rooms\/([^/]+)\/update$/.exec(url.pathname)
  if (roomUpdate) {
    const protectedFieldError = rejectsArchiveStateMutation(body, true)
    if (protectedFieldError) { writeJson(res, 400, { error: protectedFieldError }); return }
    writeJson(res, 202, await deps.updateRoom(decodeURIComponent(roomUpdate[1]), body as Partial<Room>))
    return
  }
  const roomDelete = /^\/v1\/rooms\/([^/]+)\/delete$/.exec(url.pathname)
  if (roomDelete) {
    await writeArchiveMutation(res, deps.deleteRoom(decodeURIComponent(roomDelete[1])))
    return
  }
  const roomRestore = /^\/v1\/rooms\/([^/]+)\/restore$/.exec(url.pathname)
  if (roomRestore) {
    await writeArchiveMutation(res, deps.restoreRoom(decodeURIComponent(roomRestore[1])))
    return
  }
  const roomCancel = /^\/v1\/rooms\/([^/]+)\/cancel$/.exec(url.pathname)
  if (roomCancel) {
    await deps.cancelRoom(decodeURIComponent(roomCancel[1]))
    writeJson(res, 202, { ok: true })
    return
  }
  const missionStatus = /^\/v1\/missions\/([^/]+)\/status$/.exec(url.pathname)
  if (missionStatus) {
    const status = String(body.status || '')
    if (!isMissionStatus(status)) { writeJson(res, 400, { error: 'valid_status_required' }); return }
    writeJson(res, 202, await deps.setMissionStatus(decodeURIComponent(missionStatus[1]), status, typeof body.message === 'string' ? body.message : undefined))
    return
  }
  const missionHeartbeat = /^\/v1\/missions\/([^/]+)\/heartbeat$/.exec(url.pathname)
  if (missionHeartbeat) {
    writeJson(res, 202, await deps.recordMissionHeartbeat(decodeURIComponent(missionHeartbeat[1]), typeof body.message === 'string' ? body.message : 'Remote heartbeat.'))
    return
  }
  const ticketStart = /^\/v1\/tickets\/([^/]+)\/start$/.exec(url.pathname)
  if (ticketStart) {
    const userText = typeof body.userText === 'string' ? body.userText.trim() : ''
    if (!userText) { writeJson(res, 400, { error: 'userText_required' }); return }
    writeJson(res, 202, await deps.startTicketExecution(decodeURIComponent(ticketStart[1]), {
      hostId: typeof body.hostId === 'string' ? body.hostId : '',
      userText,
    }, workspaceScope))
    return
  }
  const ticketUpdate = /^\/v1\/tickets\/([^/]+)\/update$/.exec(url.pathname)
  if (ticketUpdate) {
    const protectedFieldError = rejectsArchiveStateMutation(body, true)
    if (protectedFieldError) { writeJson(res, 400, { error: protectedFieldError }); return }
    writeJson(res, 202, await deps.updateTicket(decodeURIComponent(ticketUpdate[1]), body as Partial<Ticket>))
    return
  }
  const ticketLifecycle = /^\/v1\/tickets\/([^/]+)\/lifecycle$/.exec(url.pathname)
  if (ticketLifecycle) {
    writeJson(res, 202, await deps.transitionTicket(decodeURIComponent(ticketLifecycle[1]), body as TicketRemoteLifecycleCommand))
    return
  }
  const ticketDelete = /^\/v1\/tickets\/([^/]+)\/delete$/.exec(url.pathname)
  if (ticketDelete) {
    await writeArchiveMutation(res, deps.deleteTicket(decodeURIComponent(ticketDelete[1])))
    return
  }
  const ticketRestore = /^\/v1\/tickets\/([^/]+)\/restore$/.exec(url.pathname)
  if (ticketRestore) {
    await writeArchiveMutation(res, deps.restoreTicket(decodeURIComponent(ticketRestore[1])))
    return
  }

  writeJson(res, 404, { error: 'not_found' })
}
