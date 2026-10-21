import dns from 'node:dns/promises'
import { syncBuiltinESMExports } from 'node:module'
import { isIP } from 'node:net'

export function installDnsFixture(hosts: Readonly<Record<string, string>>): () => void {
  const previous = dns.lookup
  dns.lookup = (async (hostname: string, options?: { all?: boolean }) => {
    const address = isIP(hostname) ? hostname : hosts[hostname]
    if (!address || !isIP(address)) throw new Error(`Unexpected fixture DNS lookup: ${hostname}`)
    const entry = { address, family: isIP(address) }
    return options?.all ? [entry] : entry
  }) as typeof dns.lookup
  syncBuiltinESMExports()
  return () => {
    dns.lookup = previous
    syncBuiltinESMExports()
  }
}
