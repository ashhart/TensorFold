/** Serialize keychain writes and fence late completions across toggles and account changes. */
export function createWatchPreferenceQueue(deps: {
  identity(): string | null; disable(): void; write(identity: string, enabled: boolean): Promise<void>;
  apply(identity: string, enabled: boolean): void; error(): void;
}) {
  let intent = 0, tail: Promise<void> = Promise.resolve()
  return {
    invalidate() { intent++; deps.disable() },
    pending() { return tail },
    revision() { return intent },
    set(identity: string, enabled: boolean) {
      const mine = ++intent
      // Disable immediately, even while a previous Keychain write is still in flight.
      if (!enabled) deps.disable()
      const action = async () => {
        if (deps.identity() !== identity) return
        try {
          await deps.write(identity, enabled)
          if (mine === intent && deps.identity() === identity) deps.apply(identity, enabled)
        } catch {
          if (mine === intent && deps.identity() === identity) { deps.disable(); deps.error() }
        }
      }
      const result = tail.then(action, action)
      tail = result.catch(() => {})
      return result
    },
  }
}
