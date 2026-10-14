import type { MissionStatus, TimelineItem } from './types'

export type InterruptedMissionSettlement =
  | { action: 'fail'; reason: string }
  | { action: 'pause'; reason: string }
  | { action: 'preserve'; reason: string }

export function interruptedMissionSettlement(stalled: boolean, status: MissionStatus): InterruptedMissionSettlement {
  if (stalled) return { action: 'fail', reason: 'Mission stalled — no agent activity within the heartbeat window.' }
  if (status === 'running' || status === 'waiting_approval' || status === 'stalled') {
    return { action: 'pause', reason: 'Mission run was interrupted.' }
  }
  return { action: 'preserve', reason: `Mission is already ${status}.` }
}

export function automationRunError(error: unknown, aborted: boolean): { error: string; jobOutcome: 'cancelled' | 'failed' } {
  if (aborted) return { error: 'Automation cancelled before completion.', jobOutcome: 'cancelled' }
  return { error: String((error as any)?.message ?? error), jobOutcome: 'failed' }
}

export function readOnlyChildRunError(timeline: TimelineItem[], aborted: boolean): string | null {
  if (aborted) return 'Sub-agent cancelled before completion.'
  const item = timeline.find((candidate) => candidate.kind === 'error' && candidate.text?.trim())
  return item?.text?.trim() ?? null
}
