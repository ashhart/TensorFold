import { promises as fs, constants } from 'node:fs'
import path from 'node:path'
import type { ClusterLinkReport } from '../../shared/cluster-workspace'

/** Kernel inventory is a hint only; no utility, network peer, driver or byte test is started. */
export async function passiveNodeLinks(platform: string, sysfsRoot: string | false = '/sys'): Promise<ClusterLinkReport[]> {
  if (platform !== 'linux' || sysfsRoot === false) return []
  const root = await fs.realpath(sysfsRoot).catch(() => null)
  if (!root) return []
  const names = async (relative: string) => {
    const values = await fs.readdir(path.join(root, relative)).catch(() => [])
    if (values.length > 128) return []
    const safe: string[] = []
    for (const value of values.slice(0, 32)) {
      if (!/^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$/.test(value)) continue
      const real = await fs.realpath(path.join(root, relative, value)).catch(() => null)
      if (real?.startsWith(root + path.sep)) safe.push(value)
    }
    return safe
  }
  const rdma = await names('class/infiniband'), thunderbolt = await names('bus/thunderbolt/devices'), links: ClusterLinkReport[] = []
  if (rdma.length) {
    let active = 0
    for (const name of rdma) {
      const ports = await fs.readdir(path.join(root, 'class/infiniband', name, 'ports')).catch(() => [])
      for (const port of ports.slice(0, 8).filter(port => /^[1-9][0-9]?$/.test(port))) {
        const file = path.join(root, 'class/infiniband', name, 'ports', port, 'state'), real = await fs.realpath(file).catch(() => null)
        if (!real?.startsWith(root + path.sep)) continue
        const handle = await fs.open(file, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK).catch(() => null)
        if (!handle) continue
        try { const bytes = Buffer.alloc(128), read = await handle.read(bytes, 0, bytes.length, 0); if (/\bACTIVE\b/.test(bytes.subarray(0, read.bytesRead).toString('utf8'))) active++ } finally { await handle.close() }
      }
    }
    links.push({ transport: 'mcdma-rdma', available: false, verified: false, checkedAt: Date.now(), devices: rdma, message: `Passive kernel inventory found ${rdma.length} RDMA adapters and ${active} active ports; no MCDMA byte proof or distributed model admission is established.` })
  }
  if (thunderbolt.length) links.push({ transport: 'thunderbolt-rdma', available: false, verified: false, checkedAt: Date.now(), devices: thunderbolt, message: 'Thunderbolt hardware is visible in passive kernel inventory; native RDMA readiness and byte-check proof remain unknown.' })
  return links
}
