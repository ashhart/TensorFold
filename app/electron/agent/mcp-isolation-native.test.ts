import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import net from 'node:net'
import { randomUUID } from 'node:crypto'
import { McpHost, createMcpServer } from './mcp-client'
import { canUseLinuxSandbox } from './linux-sandbox'
import { canUseWindowsSandbox } from './windows-sandbox'
import { prepareMcpIsolation } from './mcp-isolation'

if (!canUseLinuxSandbox() && !canUseWindowsSandbox()) {
  assert.notEqual(process.env.TENSORFOLD_REQUIRE_PORTABLE_SANDBOX, '1', 'Native MCP qualification needs its real isolation backend.')
  console.log('Native MCP isolation requires a qualified Linux or Windows host.')
} else {
  const parent = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tf-mcp-native-')))
  const workspace = path.join(parent, 'project'), runtime = path.join(parent, 'runtime'), data = path.join(parent, 'data')
  const privateFile = path.join(data, 'private.txt'), explicitFile = path.join(data, 'explicit.pem')
  await fs.mkdir(workspace); await fs.mkdir(runtime); await fs.mkdir(data)
  await fs.writeFile(privateFile, 'private fixture'); await fs.writeFile(explicitFile, 'explicit fixture')
  await fs.mkdir(path.join(workspace, '.git')); await fs.writeFile(path.join(workspace, '.git/config'), 'protected fixture')
  await fs.copyFile(path.join(__dirname, 'fixtures/mcp-isolation-server.mjs'), path.join(runtime, 'server.mjs'))
  const host = new McpHost(), token = `tf-mcp-child-${randomUUID()}`
  const beforeSecret = process.env.TENSORFOLD_ISOLATION_TEST_SECRET
  process.env.TENSORFOLD_ISOLATION_TEST_SECRET = 'synthetic inherited secret'
  const config = createMcpServer({ id: 'native-isolation', name: 'Isolated fixture', command: process.execPath,
    args: [path.join(runtime, 'server.mjs'), explicitFile], cwd: workspace, env: { TF_MCP_FIXTURE_TOKEN: 'explicit fixture' } })
  host.setConfigs([config])
  const server = net.createServer(socket => socket.end())
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const call = async (operation: string, args: Record<string, unknown> = {}) => {
    await host.connect(config.id)
    const tool = host.toolDefinitions().find(tool => tool.name === 'probe')!
    assert.ok(tool)
    return host.callTool(tool.callName, { operation, ...args })
  }
  try {
    if (process.platform === 'linux') {
      const alias = path.join(parent, 'home-alias')
      await fs.symlink(os.homedir(), alias)
      await assert.rejects(prepareMcpIsolation({ ...config, args: [alias] }), /cannot expose a home/)
    }
    await call('write', { value: 'approved fixture' })
    assert.equal(await fs.readFile(path.join(workspace, 'result.txt'), 'utf8'), 'approved fixture')
    const environment = await call('environment')
    assert.match(environment.output, /"inherited":false/); assert.match(environment.output, /"explicit":true/)
    await assert.rejects(call('private', { file: privateFile }), /EACCES|ENOENT|EPERM/)
    await assert.rejects(call('git'), /EACCES|EROFS|EPERM/)
    await assert.rejects(call('new-git'), /Git metadata/)
    assert.equal(await fs.readFile(path.join(workspace, 'result.txt'), 'utf8'), 'approved fixture')
    await assert.rejects(fs.stat(path.join(workspace, 'nested')), /ENOENT/)
    const network = await call('network', { port: (server.address() as net.AddressInfo).port })
    assert.match(network.output, /"connected":false/)
    await fs.writeFile(path.join(workspace, 'result.txt'), 'new user edit')
    assert.match((await call('read')).output, /new user edit/)
    const child = JSON.parse((await call('child', { token })).output)
    await host.shutdown()
    if (process.platform === 'win32') assert.throws(() => process.kill(child.pid, 0), { code: 'ESRCH' })
    if (process.platform === 'linux') {
      const deadline = Date.now() + 5000
      let remaining: string[] = []
      do {
        remaining = []
        for (const entry of await fs.readdir('/proc')) if (/^\d+$/.test(entry)) {
          const args = await fs.readFile(`/proc/${entry}/cmdline`, 'utf8').catch(() => '')
          if (args.includes(token)) remaining.push(entry)
        }
        if (!remaining.length) break
        await new Promise(resolve => setTimeout(resolve, 50))
      } while (Date.now() < deadline)
      assert.deepEqual(remaining, [], 'Disconnected MCP cannot leave detached descendants alive')
    }
    console.log('Native MCP isolation passed: stdio, checked writes, private-file/network/environment denial, Git protection, refresh and descendant cleanup.')
  } finally {
    await host.shutdown()
    await new Promise<void>(resolve => server.close(() => resolve()))
    if (beforeSecret === undefined) delete process.env.TENSORFOLD_ISOLATION_TEST_SECRET
    else process.env.TENSORFOLD_ISOLATION_TEST_SECRET = beforeSecret
    await fs.rm(parent, { recursive: true, force: true })
  }
}
