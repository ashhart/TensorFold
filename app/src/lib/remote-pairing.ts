// Paired-session bootstrap (VST-035). Runs BEFORE the app renders (main.tsx) and decides
// which transport the renderer boots with:
//   desktop        → window.tensorfoldStudio exists (Electron preload) — nothing to do.
//   paired         → this page was served by a TensorFold host and holds a live device token.
//   needs-pairing  → served by a host, no (working) token → render the pairing screen.
//   demo           → plain browser (vite dev URL etc.) → the existing mock demo.
//
// The device token lives in localStorage under the host's own origin — the standard place
// for a web session credential; revoking the device on the host kills it server-side.

const TOKEN_KEY = 'vontra.paired.token'
const LABEL_KEY = 'vontra.paired.label'
const PAIRING_JSON_MAX_BYTES = 64 * 1024

export type PairedBootMode = 'desktop' | 'paired' | 'needs-pairing' | 'demo'

export function pairedToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}

export function pairedLabel(): string | null {
  try {
    return localStorage.getItem(LABEL_KEY)
  } catch {
    return null
  }
}

export function clearPairedToken(): void {
  try {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(LABEL_KEY)
  } catch { /* private mode */ }
}

function storePairedToken(token: string, label: string): void {
  try {
    localStorage.setItem(TOKEN_KEY, token)
    localStorage.setItem(LABEL_KEY, label)
  } catch { /* private mode: the session lasts until reload only */ }
}

async function cancelBody(response: Response): Promise<void> {
  try { await response.body?.cancel() } catch { /* best effort */ }
}

async function boundedPairingJson<T>(response: Response): Promise<T | null> {
  // Unit-test fetch doubles expose json() only. Real browser responses take the bounded,
  // MIME-checked path so a hostile service cannot make pairing allocate an unbounded body.
  if (!response.headers || !response.body) {
    try { return await response.json() as T } catch { return null }
  }
  const contentType = String(response.headers.get('content-type') ?? '').toLowerCase()
  if (!/^(?:application\/json|application\/[a-z0-9.+-]+\+json)\b/.test(contentType)) {
    await cancelBody(response)
    return null
  }
  const declared = response.headers.get('content-length')
  if (declared) {
    const size = Number(declared)
    if (!Number.isSafeInteger(size) || size < 0 || size > PAIRING_JSON_MAX_BYTES) {
      await cancelBody(response)
      return null
    }
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let bytes = 0
  let text = ''
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      bytes += value.byteLength
      if (bytes > PAIRING_JSON_MAX_BYTES) {
        await reader.cancel().catch(() => {})
        return null
      }
      text += decoder.decode(value, { stream: true })
    }
    text += decoder.decode()
  } finally {
    reader.releaseLock()
  }
  try { return JSON.parse(text) as T } catch { return null }
}

// Is this page served by a TensorFold host? /health answers {ok:true,listening} pre-auth; the
// vite demo/dev server has no such route.
async function servedByTensorFoldStudioHost(fetchImpl: typeof fetch): Promise<boolean> {
  try {
    const res = await fetchImpl('/health', {
      headers: { accept: 'application/json' },
      cache: 'no-store', credentials: 'omit', redirect: 'error',
      signal: typeof AbortSignal.timeout === 'function' ? AbortSignal.timeout(5_000) : undefined,
    })
    if (!res.ok) { await cancelBody(res); return false }
    const body = await boundedPairingJson<{ ok?: boolean; listening?: boolean }>(res)
    return body?.ok === true && 'listening' in (body ?? {})
  } catch {
    return false
  }
}

export async function redeemPairingCredential(credential: string, fetchImpl: typeof fetch = fetch): Promise<{ ok: boolean; error?: string }> {
  try {
    const res = await fetchImpl('/v1/auth/pair', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ credential: credential.trim(), clientInfo: navigator.userAgent.slice(0, 120) }),
      cache: 'no-store', credentials: 'omit', redirect: 'error',
      signal: typeof AbortSignal.timeout === 'function' ? AbortSignal.timeout(10_000) : undefined,
    })
    const body = await boundedPairingJson<{ token?: string; label?: string; error?: string }>(res)
    if (!res.ok || !body?.token) return { ok: false, error: body?.error === 'already_consumed' ? 'That pairing link was already used — issue a new one on the host.' : body?.error === 'expired' ? 'That pairing link expired — issue a new one on the host.' : 'Pairing failed — check the code and try again.' }
    storePairedToken(body.token, body.label || 'This browser')
    return { ok: true }
  } catch {
    return { ok: false, error: 'Could not reach the host.' }
  }
}

// Extract #pair= from the current URL (and scrub it from history so the one-time credential
// never lingers in the address bar or a bookmark). Legacy ?pair= links remain redeemable so an
// already-issued link does not strand a device, but new issuers use fragments exclusively.
function takePairCredentialFromUrl(): string | null {
  try {
    const url = new URL(window.location.href)
    const fromSearch = url.searchParams.get('pair')
    const fromHash = /(?:^|[#&])pair=([^&]+)/.exec(url.hash)?.[1] ?? null
    const credential = fromSearch || (fromHash ? decodeURIComponent(fromHash) : null)
    if (!credential) return null
    url.searchParams.delete('pair')
    url.hash = url.hash.replace(/(?:^|[#&])pair=[^&]*/g, '').replace(/^#&/, '#').replace(/^#$/, '')
    window.history.replaceState(null, '', url.toString())
    return credential
  } catch {
    return null
  }
}

export async function preparePairedSession(fetchImpl: typeof fetch = fetch): Promise<PairedBootMode> {
  if (typeof window !== 'undefined' && (window as { vontra?: unknown }).vontra) return 'desktop'

  const urlCredential = takePairCredentialFromUrl()
  if (urlCredential) {
    const redeemed = await redeemPairingCredential(urlCredential, fetchImpl)
    if (redeemed.ok) return 'paired'
    // Fall through: a stale link on a host page still shows the pairing screen with context.
  }

  const token = pairedToken()
  if (token) {
    // Verify the token still works — a revoked device must land on the pairing screen, not
    // boot into an app where every request fails.
    try {
      const res = await fetchImpl('/v1/snapshot', {
        headers: { 'x-vontra-token': token },
        cache: 'no-store', credentials: 'omit', redirect: 'error',
        signal: typeof AbortSignal.timeout === 'function' ? AbortSignal.timeout(10_000) : undefined,
      })
      if (res.ok) { await cancelBody(res); return 'paired' }
      if (res.status === 401) clearPairedToken()
      await cancelBody(res)
    } catch { /* fall through to the host probe */ }
  }

  return (await servedByTensorFoldStudioHost(fetchImpl)) ? 'needs-pairing' : 'demo'
}
