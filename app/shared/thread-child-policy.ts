import type { Thread } from '../electron/agent/types'

/** Secondary runtimes do not yet inherit captured bot or phone privacy bounds. */
export function assertThreadChildSupported(thread?: Pick<Thread, 'bot' | 'phoneLocalOnly'> | null): void {
  if (thread && (thread.bot !== undefined || thread.phoneLocalOnly)) {
    throw new Error('Bots and phone-local conversations cannot launch child agents or linked missions yet; continue in this conversation or use a separate ordinary task.')
  }
}
