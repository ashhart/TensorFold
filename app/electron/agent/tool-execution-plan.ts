import type { ToolCall } from './types'

export interface ToolExecutionBatch {
  mode: 'parallel' | 'ordered'
  calls: ToolCall[]
}

/**
 * Groups consecutive independent reads into bounded parallel batches. Every other
 * call is a one-item ordering barrier, so a write, approval-sensitive action, or
 * plugin tool always observes the results of every call before it.
 */
export function planToolExecutionBatches(
  calls: ToolCall[],
  canRunInParallel: (call: ToolCall) => boolean,
  maxParallel = 4,
): ToolExecutionBatch[] {
  const limit = Math.max(1, Math.floor(maxParallel) || 1)
  const batches: ToolExecutionBatch[] = []
  let reads: ToolCall[] = []

  const flushReads = () => {
    while (reads.length) {
      batches.push({ mode: 'parallel', calls: reads.slice(0, limit) })
      reads = reads.slice(limit)
    }
  }

  for (const call of calls) {
    if (canRunInParallel(call)) {
      reads.push(call)
      continue
    }
    flushReads()
    batches.push({ mode: 'ordered', calls: [call] })
  }
  flushReads()
  return batches
}

export async function executeToolExecutionBatches(
  calls: ToolCall[],
  canRunInParallel: (call: ToolCall) => boolean,
  execute: (call: ToolCall) => Promise<void>,
  commit: (call: ToolCall) => void,
  maxParallel = 4,
  shouldStop: () => boolean = () => false,
): Promise<void> {
  for (const batch of planToolExecutionBatches(calls, canRunInParallel, maxParallel)) {
    if (shouldStop()) break
    if (batch.mode === 'parallel') {
      let failed = false
      let firstError: unknown
      await Promise.allSettled(batch.calls.map(async (call) => {
        try { await execute(call) }
        catch (error) { if (!failed) { failed = true; firstError = error }; throw error }
      }))
      if (failed) throw firstError
    } else await execute(batch.calls[0])
    for (const call of batch.calls) commit(call)
  }
}
