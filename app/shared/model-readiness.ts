export interface ModelReadinessRequest {
  providerId?: string | null
  accountId?: string | null
  model?: string | null
  // Opt-in live model-list probe for local HTTP providers (issue 3539). Absent keeps the pure check.
  checkEndpoint?: boolean
}

export function modelForReadiness(input: ModelReadinessRequest | null | undefined, activeProviderId: string | null, activeModel: string | null): string | null {
  if (input && Object.prototype.hasOwnProperty.call(input, 'model')) {
    if (input.model != null && (typeof input.model !== 'string' || input.model.length > 2048)) throw new Error('Invalid model selection.')
    return input.model ?? null
  }
  return (input?.providerId ?? activeProviderId) === activeProviderId ? activeModel : null
}
