const transitions = new Map<string, number>()
const vmRuns = new Map<string, string>()

/** Reserve before the first startup await; one chat owns a bot VM until settlement. */
export function reserveBotVmRun(botId: string, threadId: string): () => void {
  if (vmRuns.has(botId)) throw new Error('This bot VM is already working in another chat; stop that run first.')
  vmRuns.set(botId, threadId)
  let released = false
  return () => {
    if (released) return
    released = true
    if (vmRuns.get(botId) === threadId) vmRuns.delete(botId)
  }
}

export function reserveBotLifecycleTransition(botId: string): () => void {
  transitions.set(botId, (transitions.get(botId) ?? 0) + 1)
  let released = false
  return () => {
    if (released) return
    released = true
    const remaining = (transitions.get(botId) ?? 1) - 1
    if (remaining) transitions.set(botId, remaining)
    else transitions.delete(botId)
  }
}

export function assertBotLifecycleIdle(botId: string): void {
  if (transitions.has(botId)) throw new Error('This bot is being paused or archived; wait for that change to finish.')
}
