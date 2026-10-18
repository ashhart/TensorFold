import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { passiveNodeLinks } from './passive-links'

const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-passive-links-')))
try {
  const device = path.join(root, 'devices', 'synthetic-rdma'), ports = path.join(device, 'ports', '1')
  await fs.mkdir(ports, { recursive: true }); await fs.writeFile(path.join(ports, 'state'), '4: ACTIVE\n')
  await fs.mkdir(path.join(root, 'class/infiniband'), { recursive: true }); await fs.symlink(device, path.join(root, 'class/infiniband', 'mlx5_0'))
  await fs.mkdir(path.join(root, 'bus/thunderbolt/devices/domain0'), { recursive: true })
  const outside = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-outside-link-'))
  await fs.symlink(outside, path.join(root, 'class/infiniband', 'escaped'))
  try {
    const links = await passiveNodeLinks('linux', root)
    assert.equal(links.length, 2); assert.ok(links.every(link => !link.available && !link.verified && !link.proof))
    assert.deepEqual(links[0].devices, ['mlx5_0']); assert.match(links[0].message, /1 active ports/)
    assert.deepEqual(await passiveNodeLinks('darwin', root), []); assert.deepEqual(await passiveNodeLinks('linux', false), [])
    console.log('PASS passive RDMA/Thunderbolt inventory: bounded fake sysfs, active-port hint without byte proof, escaped-class refusal and no command or peer execution')
  } finally { await fs.rm(outside, { recursive: true, force: true }) }
} finally { await fs.rm(root, { recursive: true, force: true }) }
