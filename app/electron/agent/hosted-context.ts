// Context windows hosted providers reported for their models (max_model_len, the field vLLM's
// /v1/models serves), by provider and model id. Electron-free so compaction limits and the context
// pressure panel can read them; a model the provider has not listed yet falls back to the guesses.

const windows = new Map<string, number>()

function key(providerId: string, model: string): string {
  return `${providerId}\u0000${model.trim()}`
}

export function recordHostedContext(providerId: string | null | undefined, model: string | null | undefined, tokens: number | null | undefined): void {
  if (!providerId || !model || typeof tokens !== 'number' || !Number.isSafeInteger(tokens) || tokens <= 0) return
  windows.set(key(providerId, model), tokens)
}

// Every listed model the provider's answer carried a window for; earlier entries for models the
// list no longer mentions stay, so a partial listing never erases a known window.
export function recordHostedContexts(providerId: string | null | undefined, models: readonly string[], tokens: Record<string, number>): void {
  if (!providerId) return
  for (const model of models) if (tokens[model] !== undefined) recordHostedContext(providerId, model, tokens[model])
}

export function hostedContextWindow(providerId: string | null | undefined, model: string | null | undefined): number | null {
  if (!providerId || !model) return null
  return windows.get(key(providerId, model)) ?? null
}

export function clearHostedContexts(): void {
  windows.clear()
}
