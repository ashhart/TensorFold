import { resolvePinnedHost, type PinnedHost, type PinnedHostResolver } from './pinned-fetch'
import { isAdmissiblePinnedIp } from './security-addresses'
import { safePublicWebUrl } from './tool-web'

// The app's own update hosts resolve and pin like the Hub's: a TUN proxy's placeholder address is
// admitted, the socket stays pinned to it and the certificate is still checked against the host.
export async function resolveUpdateHost(raw: string, lookup?: PinnedHostResolver): Promise<PinnedHost> {
  const url = safePublicWebUrl(raw)
  return resolvePinnedHost(url, (address) => {
    if (!isAdmissiblePinnedIp(address)) {
      throw new Error(`Refusing to fetch ${url.hostname}: it resolves to a private or non-public address (${address}).`)
    }
  }, { lookup })
}
