import type { Provider } from './types'

/** Pi accepts provider-specific environment credentials, but a generic key has no safe name. */
export function hasExplicitPiCliCredential(provider: Pick<Provider, 'cliEnv'>): boolean {
  return Object.entries(provider.cliEnv ?? {}).some(([name, value]) =>
    Boolean(value?.trim()) && (
      /(?:API_KEY|OAUTH_TOKEN|ACCESS_TOKEN|AUTH_TOKEN|BEARER_TOKEN)$/i.test(name) ||
      ['AWS_PROFILE', 'AWS_ACCESS_KEY_ID', 'AWS_BEARER_TOKEN_BEDROCK'].includes(name)
    ))
}

export function piCredentialHandoffError(provider: Pick<Provider, 'apiKey' | 'cliEnv'>): string | null {
  if (!provider.apiKey?.trim() || hasExplicitPiCliCredential(provider)) return null
  return 'Pi cannot receive TensorFold\'s raw provider API key without exposing it in the process argument list. Configure a provider-specific credential in the Pi provider CLI environment (for example OPENAI_API_KEY or ANTHROPIC_API_KEY), use Pi\'s own auth, or clear the raw API key.'
}

export function assertPiCredentialHandoff(provider: Pick<Provider, 'apiKey' | 'cliEnv'>): void {
  const error = piCredentialHandoffError(provider)
  if (error) throw new Error(error)
}
