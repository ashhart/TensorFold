import { BlockList, isIP } from 'node:net'

// Classify mapped IPv6 by its embedded IPv4, including expanded and URL-normalized spellings.
export function ipv4MappedToDotted(ip: string): string | null {
  let a = String(ip || '').toLowerCase().replace(/^\[|\]$/g, '').replace(/%.*$/, '')
  if (isIP(a) === 6) {
    try { a = new URL(`http://[${a}]/`).hostname.replace(/^\[|\]$/g, '') } catch { return null }
  }
  if (!a.startsWith('::ffff:')) return null
  const tail = a.slice(7)
  if (/^\d{1,3}(\.\d{1,3}){3}$/.test(tail)) return tail // ::ffff:a.b.c.d (dotted) form
  const m = /^([0-9a-f]{1,4}):([0-9a-f]{1,4})$/.exec(tail) // ::ffff:HHHH:HHHH (hex) form
  if (!m) return null
  const hi = parseInt(m[1], 16)
  const lo = parseInt(m[2], 16)
  return [(hi >> 8) & 0xff, hi & 0xff, (lo >> 8) & 0xff, lo & 0xff].join('.')
}

export function isLoopbackHost(hostname: string): boolean {
  const raw = hostname.toLowerCase().replace(/^\[|\]$/g, '')
  const host = ipv4MappedToDotted(raw) ?? raw
  return host === 'localhost' || host === '::1' || host.startsWith('127.')
}

export function isPrivateHost(hostname: string): boolean {
  const raw = hostname.toLowerCase().replace(/^\[|\]$/g, '')
  const host = ipv4MappedToDotted(raw) ?? raw
  if (host === 'metadata.google.internal' || isKnownMetadataIp(host)) return true
  if (isLoopbackHost(host)) return true
  if (host === '0.0.0.0' || host.startsWith('10.') || host.startsWith('192.168.') || host.startsWith('169.254.')) return true
  const m = /^172\.(\d+)\./.exec(host)
  if (!!m && Number(m[1]) >= 16 && Number(m[1]) <= 31) return true
  const cgnat = /^100\.(\d+)\./.exec(host)
  if (!!cgnat && Number(cgnat[1]) >= 64 && Number(cgnat[1]) <= 127) return true
  // IPv6 unique-local and link-local literals are private too.
  if (host.startsWith('fc') || host.startsWith('fd')) return true
  return host.startsWith('fe8') || host.startsWith('fe9') || host.startsWith('fea') || host.startsWith('feb')
}

// DNS answers need the same private-address checks as URL literals.
export function isPrivateIp(ip: string): boolean {
  const addr = ip.toLowerCase().replace(/^\[|\]$/g, '').replace(/%.*$/, '')
  const v4 = ipv4MappedToDotted(addr) ?? addr
  if (isKnownMetadataIp(v4)) return true
  if (/^\d{1,3}(\.\d{1,3}){3}$/.test(v4)) {
    const [a, b] = v4.split('.').map(Number)
    if (a === 0 || a === 10 || a === 127) return true
    if (a === 192 && b === 168) return true
    if (a === 169 && b === 254) return true // link-local incl. cloud metadata 169.254.169.254
    if (a === 172 && b >= 16 && b <= 31) return true
    if (a === 100 && b >= 64 && b <= 127) return true // CGNAT
    return false
  }
  if (addr === '::1' || addr === '::') return true
  if (addr.startsWith('fe8') || addr.startsWith('fe9') || addr.startsWith('fea') || addr.startsWith('feb')) return true // fe80::/10 link-local
  if (addr.startsWith('fc') || addr.startsWith('fd')) return true // fc00::/7 unique-local
  return false
}

// Metadata remains forbidden even for connectors that explicitly allow private networks.
const KNOWN_METADATA_IPS = new BlockList()
KNOWN_METADATA_IPS.addAddress('168.63.129.16', 'ipv4') // Azure platform metadata
KNOWN_METADATA_IPS.addAddress('100.100.100.200', 'ipv4') // Alibaba Cloud IMDS
KNOWN_METADATA_IPS.addAddress('fd00:ec2::254', 'ipv6') // AWS IMDS IPv6 endpoint

export function isKnownMetadataIp(ip: string): boolean {
  const raw = String(ip || '').toLowerCase().replace(/^\[|\]$/g, '').replace(/%.*$/, '')
  const address = ipv4MappedToDotted(raw) ?? raw
  const family = isIP(address)
  if (family === 4) return KNOWN_METADATA_IPS.check(address, 'ipv4')
  if (family === 6) return KNOWN_METADATA_IPS.check(address, 'ipv6')
  return false
}

// Public egress excludes reserved ranges; private-network opt-ins use a separate policy.
export function isGloballyRoutableIp(ip: string): boolean {
  if (isBenchmarkIp(ip) || isKnownMetadataIp(ip)) return false
  const raw = String(ip || '').toLowerCase().replace(/^\[|\]$/g, '').replace(/%.*$/, '')
  const address = ipv4MappedToDotted(raw) ?? raw
  if (/^\d{1,3}(?:\.\d{1,3}){3}$/.test(address)) {
    const octets = address.split('.').map(Number)
    if (octets.length !== 4 || octets.some((value) => value < 0 || value > 255)) return false
    const [a, b, c] = octets
    if (a === 0 || a === 10 || a === 127 || a >= 224) return false
    if (a === 100 && b >= 64 && b <= 127) return false
    if (a === 169 && b === 254) return false
    if (a === 172 && b >= 16 && b <= 31) return false
    if (a === 192 && b === 168) return false
    if (a === 192 && b === 0 && (c === 0 || c === 2)) return false
    if (a === 192 && b === 88 && c === 99) return false
    if (a === 198 && (b === 18 || b === 19 || (b === 51 && c === 100))) return false
    if (a === 203 && b === 0 && c === 113) return false
    return true
  }
  if (isIP(address) !== 6) return false
  const first = Number.parseInt(address.split(':')[0] || '0', 16)
  if (!Number.isFinite(first) || first < 0x2000 || first > 0x3fff) return false
  return !address.startsWith('2001:db8:') && address !== '2001:db8::'
}

export function isLoopbackIp(ip: string): boolean {
  const addr = ip.toLowerCase().replace(/^\[|\]$/g, '').replace(/%.*$/, '')
  const mapped = ipv4MappedToDotted(addr) ?? addr
  return mapped === '::1' || mapped.startsWith('127.')
}

export function isAlwaysForbiddenConnectorIp(ip: string): boolean {
  const addr = ip.toLowerCase().replace(/^\[|\]$/g, '').replace(/%.*$/, '')
  const mapped = ipv4MappedToDotted(addr) ?? addr
  if (isKnownMetadataIp(mapped)) return true
  if (mapped === '0.0.0.0' || mapped === '::') return true
  if (mapped.startsWith('169.254.')) return true
  return mapped.startsWith('fe8') || mapped.startsWith('fe9') || mapped.startsWith('fea') || mapped.startsWith('feb')
}

// Benchmark placeholders are admitted only by the separately scoped HTTPS Hub policy.
const BENCHMARK_IPS = new BlockList()
BENCHMARK_IPS.addSubnet('198.18.0.0', 15, 'ipv4')
BENCHMARK_IPS.addSubnet('2001:2::', 48, 'ipv6')
export function isBenchmarkIp(ip: string): boolean {
  const raw = String(ip || '').toLowerCase().replace(/^\[|\]$/g, '')
  if (raw.includes('%')) return false
  const address = ipv4MappedToDotted(raw) ?? raw
  const family = isIP(address)
  return family === 4 ? BENCHMARK_IPS.check(address, 'ipv4') : family === 6 && BENCHMARK_IPS.check(address, 'ipv6')
}

// A placeholder address a TUN proxy hands out is admitted for a pinned, certificate-checked host; every other private address stays refused.
export function isAdmissiblePinnedIp(ip: string): boolean {
  if (String(ip || '').includes('%')) return false
  if (isAlwaysForbiddenConnectorIp(ip)) return false
  return isGloballyRoutableIp(ip) || isBenchmarkIp(ip)
}

export function isLinkLocalOrMetadataHost(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/^\[|\]$/g, '').replace(/%.*$/, '')
  if (host === 'metadata.google.internal' || isKnownMetadataIp(host)) return true
  const v4 = ipv4MappedToDotted(host) ?? host
  if (/^\d{1,3}(\.\d{1,3}){3}$/.test(v4)) {
    const [a, b] = v4.split('.').map(Number)
    return a === 0 || (a === 169 && b === 254)
  }
  return host === '::' ||
    host.startsWith('fe8') ||
    host.startsWith('fe9') ||
    host.startsWith('fea') ||
    host.startsWith('feb')
}
