import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { BotVmService, BOT_VM_IMAGE, type VmRunner } from './service'

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-vm-test-'))
try {
  const binary = path.join(root, 'container')
  await fs.writeFile(binary, 'fictional verified runtime fixture', { mode: 0o700 })
  const calls: { args: string[]; stdin?: string; env?: NodeJS.ProcessEnv }[] = []
  let container: any = null, volume: any = null, stopped = false, execWait = false
  const json = (value: unknown) => ({ code: 0, stdout: JSON.stringify(value), stderr: '' })
  const labels = (args: string[]) => Object.fromEntries(args.flatMap((value, i) => value === '--label' ? [args[i + 1].split('=')] : []))
  const runner: VmRunner = async (_binary, args, options) => {
    calls.push({ args, stdin: options.stdin, env: options.env })
    if (args[0] === 'system' && args[1] === 'version') return json([{ appName: 'container', version: '1.5.0' }])
    if (args[0] === 'list') return json(container ? [container] : [])
    if (args[0] === 'inspect') return json([container])
    if (args[0] === 'volume' && args[1] === 'list') return json(volume ? [volume] : [])
    if (args[0] === 'volume' && args[1] === 'inspect') return json([volume])
    if (args[0] === 'volume' && args[1] === 'create') { const id = args.at(-1); volume = { id, configuration: { name: id, labels: labels(args), driver: 'local', format: 'ext4', sizeInBytes: 8 * 1024 ** 3 } }; return json(id) }
    if (args[0] === 'create') {
      const id = args[args.indexOf('--name') + 1]
      container = { id, configuration: { id, labels: labels(args), image: { reference: BOT_VM_IMAGE }, networks: [], publishedPorts: [], publishedSockets: [], ssh: false, virtualization: false, runtimeHandler: 'container-runtime-linux', readOnly: true, initProcess: { user: { raw: { userString: '1000:1000' } } }, capAdd: [], capDrop: ['ALL'], mounts: [{ destination: '/workspace', type: { volume: { name: volume.id } } }, { destination: '/tmp', type: { tmpfs: {} }, options: ['size=128M', 'mode=1777'] }] }, status: { state: 'stopped' } }
      return json(id)
    }
    if (args[0] === 'start') { container.status.state = 'running'; return json(null) }
    if (args[0] === 'stop') { stopped = true; container.status.state = 'stopped'; return json(null) }
    if (args[0] === 'delete') { assert.equal(args.at(-1), container.id); container = null; return json(null) }
    if (args[0] === 'volume' && args[1] === 'delete') { assert.equal(args.at(-1), volume.id); volume = null; return json(null) }
    if (args[0] === 'exec') {
      if (execWait) await new Promise<void>((_resolve, reject) => { options.signal?.addEventListener('abort', () => reject(new Error('VM operation cancelled.')), { once: true }) })
      return json({ ok: true, output: 'guest only', changedPaths: [] })
    }
    return json(null)
  }
  const options = { appHome: path.join(root, 'profile'), runtimePath: binary, platform: 'darwin' as const, architecture: 'arm64', osVersion: '26.5.2', runner, verifyRuntime: async () => {}, freeDiskBytes: async () => 1000 * 1024 ** 3 }
  const service = new BotVmService(options)
  assert.equal((await service.status('writer')).state, 'not-created')
  assert.equal((await service.provision('writer')).state, 'stopped')
  const originalVolume = volume.id
  assert.equal((await service.start('writer')).state, 'running')
  assert.equal((await service.executeTool('writer', 'run_command', { command: 'printf hi' })).output, 'guest only')
  const exec = calls.find((call) => call.args[0] === 'exec')!
  assert.deepEqual(exec.args.slice(0, 6), ['exec', '--interactive', '--user', '1000:1000', '--workdir', '/workspace'])
  assert.equal(exec.env?.SSH_AUTH_SOCK, undefined)
  assert.equal(exec.env?.OPENAI_API_KEY, undefined)
  const create = calls.find((call) => call.args[0] === 'create')!.args
  assert.equal(create[create.indexOf('--network') + 1], 'none')
  assert.equal(create[create.indexOf('--user') + 1], '1000:1000')
  assert.equal(create[create.indexOf('--cap-drop') + 1], 'ALL')
  assert.ok(create.includes('--read-only'))
  assert.ok(!create.some((arg) => arg.includes('type=bind')))
  await service.rebuild('writer')
  assert.equal(volume.id, originalVolume)
  assert.ok(!calls.some((call) => call.args[0] === 'volume' && ['delete', 'prune'].includes(call.args[1])))
  await service.start('writer')
  const before = calls.length
  assert.equal((await service.executeTool('writer', 'computer_use', {})).ok, false)
  assert.equal(calls.length, before, 'unknown tools never reach either runtime or host')
  container.configuration.labels['dev.tensorfold.owner'] = 'foreign-owner'
  assert.equal((await service.status('writer')).state, 'blocked')
  await assert.rejects(service.stop('writer'), /not owned/)
  assert.equal(container.status.state, 'running', 'foreign resources are not stopped')
  container.configuration.labels = labels(create)
  container.configuration.networks = [{ network: 'default' }]
  assert.equal((await service.status('writer')).state, 'blocked')
  await assert.rejects(service.executeTool('writer', 'read_file', { path: 'x' }), /isolation settings/)
  container.configuration.networks = []
  container.configuration.mounts.push({ type: { virtiofs: {} }, destination: '/host', source: '/fictional-host' })
  assert.equal((await service.status('writer')).state, 'blocked')
  container.configuration.mounts.pop()
  volume.configuration.sizeInBytes = 9 * 1024 ** 3
  assert.equal((await service.status('writer')).state, 'blocked')
  volume.configuration.sizeInBytes = 8 * 1024 ** 3
  const lowDisk = new BotVmService({ ...options, freeDiskBytes: async () => 1024 })
  await assert.rejects(lowDisk.start('writer'), /Free disk space/)
  execWait = true
  const controller = new AbortController()
  const pending = service.executeTool('writer', 'run_command', { command: 'wait' }, controller.signal)
  while (!calls.slice(before).some((call) => call.args[0] === 'exec' && JSON.parse(call.stdin!).args.command === 'wait')) await new Promise((resolve) => setTimeout(resolve, 1))
  controller.abort()
  await assert.rejects(pending, /cancelled/)
  assert.equal(stopped, true)
  assert.equal(container.status.state, 'stopped', 'cancellation stops the entire owned VM, including descendants')
  const unsupported = new BotVmService({ ...options, architecture: 'x64' })
  assert.equal((await unsupported.status('writer')).state, 'unsupported')
  await assert.rejects(service.status('../foreign'), /saved bot/)
  const fresh = new BotVmService({ ...options, appHome: path.join(root, 'linked-profile') })
  await fs.mkdir(path.join(root, 'linked-profile'))
  await fs.symlink(path.join(root, 'profile', 'bot-vms'), path.join(root, 'linked-profile', 'bot-vms'))
  assert.equal((await fresh.status('writer')).state, 'blocked')
  await service.shutdown()
  await assert.rejects(service.start('writer'), /shutting down/)
  service.resume()
  execWait = false
  await service.start('writer')
  assert.equal((await service.quiesce()).stopped, 1)
  assert.equal(container.status.state, 'stopped')
  assert.equal(volume.id, originalVolume, 'quiesce preserves the workspace disk')
  const erased = await service.purgeOwnedData()
  assert.deepEqual(erased, { containers: 1, volumes: 1 })
  assert.equal(container, null); assert.equal(volume, null)
  console.log('bot-vm service lifecycle, ownership, isolation and cancellation checks passed')
} finally { await fs.rm(root, { recursive: true, force: true }) }
