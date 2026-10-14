import type { Project, ResolvedInstructionGraph, Thread, WorkspaceLocation } from './types'
import { effectiveThreadCwd, effectiveThreadWorkspace } from './thread-workspace'

export interface ThreadInstructionPreview {
  graph: ResolvedInstructionGraph | null
  unavailableReason: string | null
}

/** Keep the preview on the same local/remote instruction boundary as runAgent. */
export async function resolveThreadInstructionPreview(
  thread: Pick<Thread, 'cwd' | 'providerId' | 'workspace' | 'projectId'>,
  resolve: (cwd: string, providerId?: string | null, workspace?: WorkspaceLocation | null) => Promise<ResolvedInstructionGraph>,
  project?: Pick<Project, 'id' | 'cwd' | 'workspace'> | null,
): Promise<ThreadInstructionPreview> {
  const workspace = effectiveThreadWorkspace(thread, project)
  if (workspace?.kind === 'ssh') {
    return { graph: null, unavailableReason: 'Remote instruction preview unavailable until instruction graphs can be read through the SSH workspace backend.' }
  }
  const cwd = effectiveThreadCwd(thread, project)
  // A chat with no folder has no instruction graph; resolving '' only raised a 'choose a folder' error toast.
  if (!cwd.trim() && !workspace) return { graph: null, unavailableReason: null }
  return { graph: await resolve(cwd, thread.providerId, workspace), unavailableReason: null }
}
