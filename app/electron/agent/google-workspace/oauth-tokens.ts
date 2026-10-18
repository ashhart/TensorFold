import { GOOGLE_TOKEN_URL } from './client-config'
import type { GoogleDesktopClient, GoogleWorkspaceTokens } from './private-state'
import { GoogleTransport } from './transport'

function token(value: unknown, required: boolean): string | undefined {
  if (value === undefined && !required) return undefined
  if (typeof value !== 'string' || !value.length || value.length > 16_384 || /[\s\0]/.test(value)) throw new Error('Google returned invalid credentials.')
  return value
}
export function parseGoogleTokens(value: any, now: number, previous?: GoogleWorkspaceTokens): GoogleWorkspaceTokens {
  if (!value || value.token_type?.toLowerCase() !== 'bearer' || !Number.isFinite(value.expires_in) || value.expires_in < 60 || value.expires_in > 86_400) throw new Error('Google returned invalid credentials.')
  const scopes: string[] | undefined = typeof value.scope === 'string' && value.scope.length <= 16_384 ? [...new Set<string>((value.scope as string).trim().split(/\s+/).filter(Boolean))] : previous?.scopes
  if (!scopes?.length || scopes.length > 64) throw new Error('Google did not confirm the granted permissions. Reconnect this account.')
  return { accessToken: token(value.access_token, true)!, refreshToken: token(value.refresh_token, false) ?? previous?.refreshToken, expiresAt: now + value.expires_in * 1000, scopes }
}
export function tokenForm(client: GoogleDesktopClient, fields: Record<string, string>): URLSearchParams {
  return new URLSearchParams({ client_id: client.clientId, ...(client.clientSecret ? { client_secret: client.clientSecret } : {}), ...fields })
}
export async function exchangeGoogleCode(transport: GoogleTransport, client: GoogleDesktopClient, code: string, redirectUri: string, verifier: string, now: number, signal: AbortSignal): Promise<GoogleWorkspaceTokens> {
  return parseGoogleTokens(await transport.form(GOOGLE_TOKEN_URL, tokenForm(client, { grant_type: 'authorization_code', code, redirect_uri: redirectUri, code_verifier: verifier }), signal), now)
}
export async function googleIdentity(transport: GoogleTransport, tokens: GoogleWorkspaceTokens, signal: AbortSignal): Promise<{ subject: string; email: string }> {
  const scopes = new Set(tokens.scopes)
  if (!scopes.has('openid') || !scopes.has('email') && !scopes.has('https://www.googleapis.com/auth/userinfo.email')) throw new Error('Google did not grant account identity permission.')
  const data = await transport.json('https://www.googleapis.com/oauth2/v3/userinfo', tokens.accessToken, undefined, 'GET', signal)
  if (typeof data.sub !== 'string' || !/^[A-Za-z0-9_-]{1,128}$/.test(data.sub) || typeof data.email !== 'string' || data.email.length > 320 || !/^[^\s<>@]+@[^\s<>@]+\.[^\s<>@]+$/.test(data.email) || data.email_verified !== true) throw new Error('Google did not return a verified account identity.')
  return { subject: data.sub, email: data.email }
}
