import type { Project, Thread, WorkspaceLocation } from './types'

export function effectiveThreadWorkspace(
  thread: Pick<Thread, 'projectId' | 'workspace'>,
  project: Pick<Project, 'id' | 'workspace'> | null | undefined,
): WorkspaceLocation | null {
  if (thread.workspace) return thread.workspace
  return project && thread.projectId === project.id ? project.workspace ?? null : null
}

export function effectiveThreadCwd(
  thread: Pick<Thread, 'cwd' | 'projectId' | 'workspace'>,
  project: Pick<Project, 'id' | 'cwd' | 'workspace'> | null | undefined,
): string {
  const workspace = effectiveThreadWorkspace(thread, project)
  if (workspace?.kind === 'ssh') return workspace.path
  return thread.cwd || workspace?.path || (project && thread.projectId === project.id ? project.cwd : '') || ''
}
