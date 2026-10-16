import { requireOptionalNativeModule } from 'expo'
import { fetch as expoFetch } from 'expo/fetch'
import { HostError } from './client'

interface PinnedNetwork { request(url: string, method: string, headers: Record<string, string>, body: string | null, pin: string): Promise<{ status: number; headers: Record<string, string>; body: string }> }
const native = requireOptionalNativeModule<PinnedNetwork>('TensorFoldPinnedNetwork')

export function fetchForPin(pin?: string): typeof fetch {
  if (!pin) return expoFetch as typeof fetch
  if (!/^[a-fA-F0-9]{64}$/.test(pin)) throw new HostError('The host certificate pin is invalid; pair again from the Mac.')
  return (async (input, init) => {
    if (!native) throw new HostError('This build cannot verify a private host certificate; use the native TensorFold mobile build.')
    if (init?.signal?.aborted) throw new HostError('The host request was cancelled.')
    const headers: Record<string, string> = {}
    new Headers(init?.headers).forEach((value, key) => { headers[key] = value })
    let response: Awaited<ReturnType<PinnedNetwork['request']>>
    try { response = await native.request(String(input), init?.method || 'GET', headers, typeof init?.body === 'string' ? init.body : null, pin.toLowerCase()) }
    catch (error) { throw new HostError((error as Error)?.message || 'The private host certificate could not be verified.', 0, 'certificate_verification') }
    if (init?.signal?.aborted) throw new HostError('The host request was cancelled.')
    return new Response(response.body, { status: response.status, headers: response.headers })
  }) as typeof fetch
}
