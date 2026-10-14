// Shared renderer/main plan math (todo statuses). Kept structural so both
// the renderer panel and the agent loop can use it without runtime imports across
// the type-only electron/renderer boundary.

export type PlanStepStatusLike = 'pending' | 'in_progress' | 'done' | 'blocked' | 'dropped'

export interface PlanStepLike {
  text: string
  done: boolean
  status?: PlanStepStatusLike
  depth?: number
  blocker?: string
}

export interface PlanProgress {
  total: number
  done: number
  open: number
  blocked: number
  inProgress: number
  /** done + half-credit for in_progress, over total — drives the progress bar. */
  fraction: number
}

export function planProgress(plan: PlanStepLike[]): PlanProgress {
  const total = plan.length
  let done = 0
  let credit = 0
  let blocked = 0
  let inProgress = 0
  for (const step of plan) {
    if (step.done || step.status === 'done') { done++; credit += 1; continue }
    if (step.status === 'blocked') { blocked++; continue }
    if (step.status === 'in_progress') { inProgress++; credit += 0.5; continue }
  }
  return { total, done, open: total - done, blocked, inProgress, fraction: total ? Math.min(1, credit / total) : 0 }
}

export function openPlanSteps<T extends PlanStepLike>(plan: T[]): T[] {
  return plan.filter((step) => !step.done && step.status !== 'done' && step.status !== 'dropped')
}
