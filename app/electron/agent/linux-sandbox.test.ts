import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import net from 'node:net'
import { spawnSync } from 'node:child_process'
import { canUseLinuxSandbox, linuxSandboxCommand } from './linux-sandbox'
if (process.platform !== 'linux') console.log('Native Linux sandbox qualification requires Linux.')
else if (!canUseLinuxSandbox()) {
  assert.equal(process.env.TENSORFOLD_REQUIRE_LINUX_SANDBOX, undefined, 'This Linux qualification lane requires working bubblewrap.')
  console.log('Bubblewrap unavailable; command admission stays closed.')
} else {
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tf-bwrap-test-')))
  const workspace = path.join(root, 'workspace'), outside = path.join(root, 'outside.txt')
  await fs.mkdir(workspace); await fs.mkdir(path.join(workspace, 'nested'));  await fs.writeFile(path.join(workspace, 'input.txt'), 'fixture'); await fs.writeFile(outside, 'private')
  const run = (text: string, network = false, stage = false) => spawnSync(...launch(text, network, stage), { encoding: 'utf8' })
  function launch(text: string, network: boolean, stage: boolean): [string, string[]] {
    const [command, ...args] = linuxSandboxCommand({ command: '/bin/sh', args: ['-c', text], cwd: workspace, allowNetwork: network, writableStage: stage })
    return [command, args]
  }
  try {
    assert.equal(run('cat input.txt').stdout.trim(), 'fixture')
    for (const text of ['echo bad > input.txt', 'mkdir .git', 'mkdir nested/.git', `cat '${outside}'`, 'cat /etc/shadow']) assert.notEqual(run(text).status, 0, text)
    assert.equal(run('echo scratch > /tmp/tf-scratch && test -f /tmp/tf-scratch').status, 0)
    assert.equal(await fs.readFile(path.join(workspace, 'input.txt'), 'utf8'), 'fixture')
    assert.deepEqual((await fs.readdir(workspace)).sort(), ['input.txt', 'nested'])
    const server = net.createServer(socket => socket.end())
    await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
    try {
      const port = (server.address() as net.AddressInfo).port
      const connect = `/usr/bin/python3 -c 'import socket; socket.create_connection(("127.0.0.1",${port}),1).close()'`
      assert.notEqual(run(connect).status, 0, 'a new network namespace cannot reach the host listener')
      assert.equal(run(connect, true).status, 0, 'an explicitly network-enabled profile can reach the same listener')
    } finally { await new Promise<void>(resolve => server.close(() => resolve())) }
    assert.equal(run('echo stage > stage.txt', false, true).status, 0)
    assert.equal(run(`cat '${outside}'`, false, true).status === 0, false)
    console.log('Native bubblewrap denies workspace/Git/private-file writes and reads, preserves temp ordering and admits only private-stage writes.')
  } finally { await fs.rm(root, { recursive: true, force: true }) }
}
