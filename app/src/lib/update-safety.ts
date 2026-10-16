// Component-local drafts are outside the main process's durable store; owners block restart here.
const blockers = new Set<() => string | null>()

export function registerUpdatePreparationBlocker(check: () => string | null): () => void {
  blockers.add(check)
  return () => { blockers.delete(check) }
}

export function rendererUpdateBlockReason(): string | null {
  for (const check of blockers) {
    const reason = check()
    if (reason) return reason
  }
  return null
}
