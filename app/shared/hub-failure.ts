// Explicit codes survive Electron's message-only error copy without sharing main-process classes.
export type HubFailureCategory = 'policy' | 'transport' | 'offline' | 'http' | 'check'
export interface HubFailure { category: HubFailureCategory; message: string; coded: boolean }
const CODE = /^\[TF_HUB_(POLICY|TRANSPORT|OFFLINE|HTTP|CHECK)\]\s*/

function errorMessage(error: unknown): string {
  if (typeof error === 'string') return error
  if (error && typeof error === 'object' && typeof (error as { message?: unknown }).message === 'string') return (error as Error).message
  return 'Model checks could not be completed.'
}

export function decodeHubFailure(error: unknown): HubFailure {
  const text = errorMessage(error).replace(/^Error invoking remote method '[^']*':\s*/, '').replace(/^(?:Error:\s*)+/, '')
  const match = CODE.exec(text)
  return { category: match ? match[1].toLowerCase() as HubFailureCategory : 'check', message: match ? text.slice(match[0].length) : text, coded: !!match }
}

export function hubFailureCause(error: unknown): HubFailure | null {
  let current = error
  for (let depth = 0; current && depth < 8; depth++) {
    const failure = decodeHubFailure(current)
    if (failure.coded) return failure
    current = typeof current === 'object' ? (current as { cause?: unknown }).cause : null
  }
  return null
}

export function hubError(category: HubFailureCategory, message: string, cause?: unknown): Error {
  return new Error(`[TF_HUB_${category.toUpperCase()}] ${message}`, { cause })
}

export function aggregateHubFailures(kinds: readonly HubFailureCategory[]): HubFailureCategory | null {
  return (['policy', 'offline', 'transport', 'http', 'check'] as const).find((kind) => kinds.includes(kind)) ?? null
}

export function hubFailureCopy(kind: HubFailureCategory) {
  switch (kind) {
    case 'policy': return { title: 'Model checks blocked by network policy', detail: 'The Hub address was refused. Check your DNS or proxy settings, then retry.' }
    case 'transport': return { title: 'Could not reach the Hub', detail: 'Check your connection or proxy settings, then retry.' }
    case 'offline': return { title: 'Model checks are offline', detail: 'HF_HUB_OFFLINE is set for this session. Turn it off to check models.' }
    case 'http': return { title: 'The Hub could not complete model checks', detail: 'The Hub answered an error. Read the model details, then retry.' }
    case 'check': return { title: 'Model checks could not be completed', detail: 'Read the model details, then retry.' }
  }
}
