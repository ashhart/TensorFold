import { createServer } from 'node:http'
import { createHash, randomBytes, timingSafeEqual } from 'node:crypto'
import { GOOGLE_AUTH_URL, GOOGLE_ISS } from './client-config'
import { googleOAuthErrorMessage } from './oauth-errors'
import type { GoogleDesktopClient } from './private-state'

export interface GoogleOAuthLoopback {
  authorizationUrl: string
  redirectUri: string
  verifier: string
  completion: Promise<string>
  cancel(): void
}
// A parameter's name reaches the status snapshot and the panel, so only a short plain one
// is shown; any other name is described without it.
const PLAIN_PARAMETER_NAME = /^[A-Za-z0-9_.-]{1,40}$/
function plainParameterName(name: string): string | undefined {
  return PLAIN_PARAMETER_NAME.test(name) ? name : undefined
}
export async function createGoogleOAuthLoopback(client: GoogleDesktopClient, scopes: string[], timeoutMs = 180_000): Promise<GoogleOAuthLoopback> {
  const state = randomBytes(32).toString('base64url')
  const verifier = randomBytes(48).toString('base64url')
  const callbackPath = `/google-callback/${randomBytes(16).toString('hex')}`
  let settled = false
  let timer: ReturnType<typeof setTimeout> | undefined
  let resolveCode!: (code: string) => void
  let rejectCode!: (error: Error) => void
  const completion = new Promise<string>((resolve, reject) => { resolveCode = resolve; rejectCode = reject })
  // The service may still be awaiting its browser opener when a callback arrives.
  void completion.catch(() => {})
  const server = createServer({ maxHeaderSize: 8192 }, (request, response) => {
    response.setHeader('Cache-Control', 'no-store')
    response.setHeader('Content-Security-Policy', "default-src 'none'; frame-ancestors 'none'")
    response.setHeader('X-Content-Type-Options', 'nosniff')
    response.setHeader('Content-Type', 'text/plain; charset=utf-8')
    const fail = () => { response.writeHead(400); response.end('Invalid sign-in callback. Return to TensorFold and try again.') }
    if (settled || request.method !== 'GET' || request.headers.host !== `127.0.0.1:${port}` || !request.url || request.url.length > 8192 || !request.url.startsWith('/')) return fail()
    let url: URL
    try { url = new URL(request.url, redirectUri) } catch { return fail() }
    if (url.origin !== `http://127.0.0.1:${port}` || url.pathname !== callbackPath || url.hash) return fail()
    // The state is what proves the callback is ours. A stranger's request keeps the sign-in open.
    const actualState = url.searchParams.get('state') ?? ''
    if (!/^[A-Za-z0-9_-]{43}$/.test(actualState) || actualState.length !== state.length || !timingSafeEqual(Buffer.from(actualState), Buffer.from(state))) return fail()
    // The callback is ours, so it settles the sign-in either way: a malformed one names the problem.
    const refuse = (reason: string) => {
      response.writeHead(400)
      response.end('Sign-in was not completed. Return to TensorFold for details.')
      finish(new Error(reason))
    }
    const allowed = new Set(['state', 'code', 'error', 'error_description', 'error_uri', 'scope', 'authuser', 'prompt', 'hd', 'iss'])
    const names = [...url.searchParams.keys()]
    const unexpected = names.find(name => !allowed.has(name))
    if (unexpected) {
      const named = plainParameterName(unexpected)
      return refuse(named ? `Google sign-in failed: the callback carried an unexpected parameter (${named}).` : 'Google sign-in failed: the callback carried an unexpected parameter.')
    }
    const repeated = names.find(name => url.searchParams.getAll(name).length !== 1)
    if (repeated) {
      const named = plainParameterName(repeated)
      return refuse(named ? `Google sign-in failed: the callback repeated the parameter ${named}.` : 'Google sign-in failed: the callback repeated an unexpected parameter.')
    }
    // RFC 9207: an issuer that is present must be Google's exactly; its absence is still fine.
    const issuer = url.searchParams.get('iss')
    if (issuer !== null && issuer !== GOOGLE_ISS) return refuse(`Google sign-in failed: the callback's issuer was not ${GOOGLE_ISS}, so it was not accepted.`)
    const code = url.searchParams.get('code')
    const error = url.searchParams.get('error')
    if (url.searchParams.has('code') === url.searchParams.has('error') || !code && !error || code && (code.length > 4096 || /[\s\0]/.test(code))) return refuse('Google sign-in failed: the callback carried neither a usable authorization code nor an error.')
    response.writeHead(200)
    response.end(error ? 'Sign-in was not completed. You can close this tab and return to TensorFold.' : 'Google sign-in received. Return to TensorFold to check the connection; you can close this tab.')
    finish(error ? new Error(googleOAuthErrorMessage(error, url.searchParams.get('error_description'))) : undefined, code ?? undefined)
  })
  server.requestTimeout = 5_000
  server.headersTimeout = 5_000
  server.keepAliveTimeout = 1
  server.maxConnections = 8
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => { server.removeListener('error', reject); resolve() })
  })
  const address = server.address()
  if (!address || typeof address === 'string') { server.close(); throw new Error('Could not start the local Google sign-in callback.') }
  const port = address.port
  const redirectUri = `http://127.0.0.1:${port}${callbackPath}`
  function finish(error?: Error, code?: string) {
    if (settled) return
    settled = true
    if (timer) clearTimeout(timer)
    server.close()
    server.closeIdleConnections()
    if (error) rejectCode(error)
    else resolveCode(code!)
  }
  server.on('error', () => finish(new Error('The local Google sign-in callback failed.')))
  timer = setTimeout(() => finish(new Error('Google sign-in timed out. Please try again.')), timeoutMs)
  timer.unref?.()
  const url = new URL(GOOGLE_AUTH_URL)
  url.search = new URLSearchParams({ client_id: client.clientId, response_type: 'code', redirect_uri: redirectUri, scope: scopes.join(' '), state, code_challenge: createHash('sha256').update(verifier).digest('base64url'), code_challenge_method: 'S256', access_type: 'offline', prompt: 'consent select_account' }).toString()
  // Installed OAuth clients cannot use include_granted_scopes incremental authorization.
  return { authorizationUrl: url.toString(), redirectUri, verifier, completion, cancel: () => finish(new Error('Google sign-in was cancelled.')) }
}
