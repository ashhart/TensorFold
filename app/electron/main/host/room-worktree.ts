import path from 'node:path'
import { promises as fs } from 'node:fs'
import type { Project, Room } from '../../agent/types'
import { auditPreview } from '../../agent/security'
import { safeSegment, vontraPath, writeTensorFoldStudioProtectedJson } from '../../store'
import { auditEvent } from './audit'
import { inspectGitRepository, qq, sh, shStrict } from './workspace'

async function ensureRoomFolders(project: Project, room: Room): Promise<{ root: string; scratchpad: string; code: string }> {
  const projectDir = `${safeSegment(project.name)}-${project.id.slice(0, 8)}`
  const roomDir = `${safeSegment(room.name)}-${room.id.slice(0, 8)}`
  const root = room.rootCwd || vontraPath('rooms', projectDir, roomDir)
  const scratchpad = room.scratchpadCwd || path.join(root, 'scratchpad')
  const code = room.workspaceCwd || path.join(root, 'code')
  await fs.mkdir(scratchpad, { recursive: true })
  await fs.writeFile(
    path.join(scratchpad, 'README.md'),
    [
      `# ${room.name}`,
      '',
      `Project: ${project.name}`,
      `Room: ${room.id}`,
      `Created: ${new Date(room.createdAt).toISOString()}`,
      '',
      'This scratchpad is outside the code worktree. Use it for room notes, run summaries, and temporary agent artifacts.',
      '',
    ].join('\n'),
    { encoding: 'utf8', mode: 0o600, flag: 'wx' },
  ).catch(() => {})
  // This duplicates project paths, room prompts/memory, and seat metadata outside rooms.json.
  // Protect it too; writeTensorFoldStudioProtectedJson securely migrates a legacy plaintext JSON copy.
  await writeTensorFoldStudioProtectedJson(['rooms', projectDir, roomDir, 'room.json'], { project, room: { ...room, rootCwd: root, scratchpadCwd: scratchpad, workspaceCwd: code } }).catch(() => {})
  return { root, scratchpad, code }
}

export async function ensureRoomWorktree(project: Project, room: Room): Promise<string> {
  if (!project.cwd) throw new Error('Choose a project folder before starting a room. Rooms use Git worktrees to isolate agents.')
  const repo = await inspectGitRepository(project.cwd, 'project workspace')
  if (!repo.exists || !repo.topLevel) {
    throw new Error('This project folder is not initialized as a Git repo. Initialize Git first so TensorFold can create isolated room worktrees.')
  }
  if (!repo.hasCommits) {
    throw new Error('This Git repo has no commits yet. Make an initial commit before starting a multi-agent room so worktrees have a stable base.')
  }
  const folders = await ensureRoomFolders(project, room)
  room.rootCwd = folders.root
  room.scratchpadCwd = folders.scratchpad
  room.workspaceCwd = folders.code
  const workspace = folders.code
  if (room.workspaceCwd) {
    const existing = await inspectGitRepository(workspace, 'room workspace').catch(() => null)
    if (existing?.exists && path.resolve(existing.cwd) === path.resolve(workspace)) return existing.cwd
  }
  const stat = await fs.stat(workspace).catch(() => null)
  if (stat) {
    const existing = await inspectGitRepository(workspace, 'room workspace').catch(() => null)
    if (existing?.exists) return workspace
    const entries = stat.isDirectory() ? await fs.readdir(workspace).catch(() => []) : []
    if (entries.length) throw new Error(`Room worktree path is already occupied: ${workspace}`)
  }

  const branch = `vontra/room-${room.id.slice(0, 8)}`
  await auditEvent('room.worktree_create', { roomId: room.id, projectId: project.id, source: auditPreview(repo.topLevel), workspace: auditPreview(workspace), branch })
  const branchExists = (await sh(`git branch --list ${qq(branch)}`, repo.topLevel, 5000)).trim().length > 0
  const command = branchExists
    ? `git -C ${qq(repo.topLevel)} worktree add ${qq(workspace)} ${qq(branch)}`
    : `git -C ${qq(repo.topLevel)} worktree add -b ${qq(branch)} ${qq(workspace)} HEAD`
  await shStrict(command, repo.topLevel, 30_000)
  return workspace
}
