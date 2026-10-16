/**
 * Small keyed async queue for main-process runtimes.
 *
 * Calls sharing a key run in issue order; different keys remain independent. Settled
 * tails are removed so a long-running desktop session does not retain every connector,
 * webhook, or project key it has ever seen.
 */
export function createSerialTaskQueue() {
  const tails = new Map<string, Promise<void>>()

  function run<T>(key: string, task: () => Promise<T>): Promise<T> {
    const previous = tails.get(key) ?? Promise.resolve()
    const result = previous.catch(() => {}).then(task)
    const tail = result.then(() => {}, () => {})
    tails.set(key, tail)
    void tail.finally(() => {
      if (tails.get(key) === tail) tails.delete(key)
    })
    return result
  }

  return {
    run,
    /** Diagnostics/tests only; callers must not use this for synchronization. */
    pendingKeys: () => tails.size,
  }
}
