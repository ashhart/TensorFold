import { createSerialTaskQueue } from '../../agent/serial-task-queue'

/**
 * One ordering boundary for webhook config. A global key is intentional: creates must serialize
 * with each other for path uniqueness, while saves/approvals must not interleave across DNS awaits.
 */
export function createWebhookMutationQueue() {
  const queue = createSerialTaskQueue()
  return {
    run<T>(task: () => Promise<T>): Promise<T> {
      return queue.run('webhook-config', task)
    },
    pending: () => queue.pendingKeys(),
  }
}
