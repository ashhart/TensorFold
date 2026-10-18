import type { GoogleDesktopClient } from './private-state'

export const GOOGLE_AUTH_URL = 'https://accounts.google.com/o/oauth2/v2/auth'
export const GOOGLE_TOKEN_URL = 'https://oauth2.googleapis.com/token'
export const GOOGLE_REVOKE_URL = 'https://oauth2.googleapis.com/revoke'

export function parseGoogleDesktopClient(input: string): GoogleDesktopClient {
  let object: any = input
  if (typeof input === 'string') {
    if (input.length > 32_768) throw new Error('Google desktop client configuration is too large.')
    try { object = JSON.parse(input) } catch { throw new Error('Select the installed desktop OAuth client JSON from Google Cloud.') }
    if (!object || typeof object !== 'object' || !object.installed || object.web || object.type === 'service_account') throw new Error('Google OAuth client must be a Desktop app, not a web client or service account.')
    const installed = object.installed
    if (installed.auth_uri !== 'https://accounts.google.com/o/oauth2/auth' && installed.auth_uri !== GOOGLE_AUTH_URL) throw new Error('Unexpected Google authorization endpoint.')
    if (installed.token_uri !== GOOGLE_TOKEN_URL) throw new Error('Unexpected Google token endpoint.')
    if (installed.redirect_uris !== undefined && (!Array.isArray(installed.redirect_uris) || !installed.redirect_uris.some((value: unknown) => value === 'http://localhost' || value === 'http://127.0.0.1'))) throw new Error('Google desktop client does not allow the local callback.')
    object = { clientId: installed.client_id, clientSecret: installed.client_secret }
  } else throw new Error('Select the installed desktop OAuth client JSON from Google Cloud.')
  if (!object || typeof object !== 'object' || typeof object.clientId !== 'string' || !/^[0-9]{3,32}-[a-z0-9_-]{3,160}\.apps\.googleusercontent\.com$/.test(object.clientId)) throw new Error('Enter a valid Google desktop OAuth client ID.')
  if (typeof object.clientSecret !== 'string' || !/^[A-Za-z0-9_-]{3,256}$/.test(object.clientSecret)) throw new Error('Google desktop client JSON must include its client secret.')
  return { clientId: object.clientId, ...(object.clientSecret ? { clientSecret: object.clientSecret } : {}) }
}
