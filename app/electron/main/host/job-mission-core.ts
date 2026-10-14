import { updateJob } from '../../agent/jobs'
import type { JobPatch } from '../../agent/jobs'
import type { Job } from '../../agent/types'
import { abortMissionsForArchive } from './mission-runtime'
import { store } from '../state'

interface JobMissionCoreDeps {
  persistJobs(): void
  emitJob(job: Job): void
}

/**
 * The shared mutable substrate for mission + job execution. The `runningMissions`
 * set, the per-mission abort controllers, the approval-resolver bridge, and the
 * job patch/replace helpers all live here so the four runtimes that need them
 * (mission, room-job, job-control, security-purge) share a single identity.
 *
 * These collections MUST be created exactly once and passed by reference; the
 * archive flow awaits the `runningMissions` set draining, and the approval flow
 * resolves resolvers stashed in `missionApprovalResolvers`.
 */
export function createJobMissionCore(deps: JobMissionCoreDeps) {
  const runningMissions = new Set<string>() // ids with a live executor (no double-run)
  const missionAborters = new Map<string, AbortController>()
  // Bridges the agent-loop approval (keyed by callId) to the durable mission approval flow.
  const missionApprovalResolvers = new Map<string, (d: 'allow' | 'allow_always' | 'deny') => void>()

  async function abortAndWaitMissionsForArchive(target: { projectId?: string | null; threadId?: string | null }): Promise<void> {
    const ids = abortMissionsForArchive({ missions: store.missions, threads: store.threads }, missionAborters, target)
    if (!ids.some((id) => runningMissions.has(id))) return
    const deadline = Date.now() + 15_000
    while (ids.some((id) => runningMissions.has(id))) {
      if (Date.now() >= deadline) throw new Error('Mission cancellation did not settle; archive was stopped to prevent post-archive writes.')
      await new Promise<void>((resolve) => setTimeout(resolve, 10))
    }
  }

  function patchJob(job: Job | null, patch: JobPatch): Job | null {
    if (!job) return null
    const result = updateJob(store.jobs, job.id, patch)
    if (!result.job) return job
    store.jobs = result.jobs
    deps.persistJobs()
    deps.emitJob(result.job)
    return result.job
  }

  function replaceJobResult(result: { jobs: Job[]; job: Job | null }): Job {
    if (!result.job) throw new Error('Job cannot be changed from its current state.')
    store.jobs = result.jobs
    deps.persistJobs()
    deps.emitJob(result.job)
    return result.job
  }

  return { runningMissions, missionAborters, missionApprovalResolvers, patchJob, replaceJobResult, abortAndWaitMissionsForArchive }
}
