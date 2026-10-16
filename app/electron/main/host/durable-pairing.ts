import type { PairingRegistry } from '../../agent/pairing-sessions'

const mutationTails = new WeakMap<PairingRegistry, Promise<void>>()

export interface DurablePairingMutationOptions<T> {
  /** Invalid/no-op results need no write (for example an unknown pairing credential). */
  changed?: (result: T) => boolean
  /** Revocation/purge callers may elect to stay fail-closed in memory if disk is unavailable. */
  rollbackOnFailure?: boolean
}

/** Commit a pairing mutation only when its protected registry snapshot is durable. */
export async function durablePairingMutation<T>(
  registry: PairingRegistry,
  mutate: () => T,
  persist: () => Promise<void>,
  options: DurablePairingMutationOptions<T> = {},
): Promise<T> {
  const prior = mutationTails.get(registry) ?? Promise.resolve()
  const run = prior.then(async () => {
    const before = registry.snapshot()
    try {
      const result = mutate()
      if (options.changed?.(result) === false) return result
      await persist()
      return result
    } catch (error) {
      if (options.rollbackOnFailure !== false) {
        registry.clearAll()
        registry.restore(before)
      }
      throw error
    }
  })
  mutationTails.set(registry, run.then(() => undefined, () => undefined))
  return run
}
