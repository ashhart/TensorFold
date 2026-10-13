// Incognito threads (composer toggle): memory-only conversations. Two seams keep
// them off disk — saveThreads strips them from threads.json, and the event-spine
// bridge drops their events before the journal sees them. Everything else treats
// them as ordinary threads while the app runs.

/** Threads that may be written to disk — incognito ones never persist. */
export function stripIncognitoThreads<T extends { incognito?: boolean }>(threads: T[]): T[] {
  return threads.filter((thread) => !thread.incognito)
}

/** Should this thread's agent events reach the persistent event spine? */
export function spineEligible(thread: { incognito?: boolean } | undefined | null): boolean {
  return !thread?.incognito
}
