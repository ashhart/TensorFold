// Context windows the running engine reported (max_model_len), by repo. Electron-free so compaction
// limits and the context meter can read it; a model the engine has not served yet falls back to the
// configured local limit.

const windows = new Map<string, number>()

export function recordEngineContext(model: string | null | undefined, tokens: number | null | undefined): void {
  if (!model || typeof tokens !== 'number' || !Number.isSafeInteger(tokens) || tokens <= 0) return
  windows.set(model, tokens)
}

export function engineContextWindow(model: string | null | undefined): number | null {
  return model ? windows.get(model) ?? null : null
}

export function clearEngineContexts(): void {
  windows.clear()
}
