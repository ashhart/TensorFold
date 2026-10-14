// Incognito threads (composer toggle): memory-only conversations. Two seams keep
// them off disk — protected snapshots, journals and automatic learning skip them.
// Cross-conversation retrieval cannot expose their live in-memory history either.

/** These tools create derived durable conversations, mailboxes or learned skills. */
export const INCOGNITO_PERSISTENT_TOOLS = new Set([
  'delegate_task', 'launch_subagent', 'launch_subagents', 'subagent_send', 'subagent_inbox',
  'suggest_skill', 'skill_curator', 'save_artifact',
])

/** Threads that may be written to disk — incognito ones never persist. */
export function stripIncognitoThreads<T extends { incognito?: boolean }>(threads: T[]): T[] {
  return threads.filter((thread) => !thread.incognito)
}

/** Should this thread's agent events reach the persistent event spine? */
export function spineEligible(thread: { incognito?: boolean } | undefined | null): boolean {
  return !thread?.incognito
}
