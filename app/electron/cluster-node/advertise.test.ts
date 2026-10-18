import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import type { ChildProcess } from 'node:child_process'
import { OsNodePublisher, NODE_DISCOVERY_SERVICE } from './advertise'
import { createClusterNodeServer } from './server'
import { FakeEngine, deferred, until, sleep } from './testing/fixture'
import { ephemeralTls } from './testing/tls'

class PublisherChild extends EventEmitter {
  stdout = new PassThrough(); stderr = new PassThrough(); signals: string[] = []; exitCode: number | null = null
  drain: ReturnType<typeof deferred> | null = null
  kill(signal: string) { this.signals.push(signal); const done = () => { this.exitCode = 0; this.emit('close', 0) }; if (this.drain) void this.drain.promise.then(done); else queueMicrotask(done); return true }
}
async function main() {
  const calls: Array<{ file: string; args: string[]; options: any; child: PublisherChild }> = []
  const make = (platform: string, acknowledge = true, drain = false) => new OsNodePublisher({ platform, timeoutMs: 30, access: async file => { assert.ok(file.startsWith('/usr/bin/')) }, spawn: (file, args, options) => {
    const child = new PublisherChild(); if (drain) child.drain = deferred(); calls.push({ file, args, options, child })
    if (acknowledge) queueMicrotask(() => child.stderr.write(platform === 'linux' ? "Established under name 'synthetic'\n" : 'Name now registered and active\n'))
    return child as unknown as ChildProcess
  } })
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-node-advertise-')))
  try {
    const darwin = await make('darwin', true, true).start('opaque-node-identity', 8766)
    assert.equal(calls[0].file, '/usr/bin/dns-sd'); assert.deepEqual(calls[0].args.slice(-2), ['protocol=1', 'transport=ssh']); assert.ok(calls[0].args.includes(NODE_DISCOVERY_SERVICE))
    assert.ok(!calls[0].args.join(' ').includes('opaque-node-identity'), 'The DNS label is opaque and does not broadcast a configurable node identity verbatim')
    assert.equal(calls[0].options.shell, false); assert.deepEqual(Object.keys(calls[0].options.env).sort(), ['LANG', 'LC_ALL', 'PATH'])
    let stopped = false; const closing = darwin.stop().then(() => { stopped = true }); await sleep(10); assert.equal(stopped, false); calls[0].child.drain!.resolve(); await closing; assert.equal(stopped, true)
    const linux = await make('linux').start('opaque-node-identity', 8766); assert.equal(calls[1].file, '/usr/bin/avahi-publish-service'); assert.equal(calls[1].args[0], '-s'); await linux.stop()
    await assert.rejects(new OsNodePublisher({ platform: 'darwin', access: async () => { throw Object.assign(new Error('Synthetic missing tool'), { code: 'ENOENT' }) }, spawn: () => { throw new Error('Never spawn a missing utility') } }).start('opaque-node', 8766), /dns-sd.*nothing was published/)
    await assert.rejects(make('linux', false).start('opaque-node', 8766), /deadline/); assert.ok(calls[2].child.signals.includes('SIGTERM'))
    const controller = new AbortController(); const pending = make('darwin', false).start('opaque-node', 8766, controller.signal); await until(() => calls.length >= 4); controller.abort(); await assert.rejects(pending, /cancelled/)
    const publisher = make('darwin')
    const node = await createClusterNodeServer({ config: { schema: 1, dataDir: path.join(root, 'opt-in'), host: '127.0.0.1', port: 0, advertise: true }, engine: new FakeEngine(), publisher })
    await node.listen(); const advertised = calls.at(-1)!; assert.ok(advertised.args.includes('transport=ssh')); await node.close(); assert.ok(advertised.child.signals.includes('SIGTERM'))
    const noPublish = await createClusterNodeServer({ config: { schema: 1, dataDir: path.join(root, 'default-off'), host: '127.0.0.1', port: 0 }, engine: new FakeEngine(), publisher: { start: async () => { throw new Error('Default-off must not execute a publisher') } } })
    await noPublish.listen(); await noPublish.close()
    const failed = await createClusterNodeServer({ config: { schema: 1, dataDir: path.join(root, 'missing-tool'), host: '127.0.0.1', port: 0, advertise: true }, engine: new FakeEngine(), publisher: new OsNodePublisher({ platform: 'linux', access: async () => { throw new Error('Synthetic missing tool') } }) })
    await assert.rejects(failed.listen(), /avahi-publish-service/); assert.equal(failed.address(), null)
    const httpsAdvertisement = await make('darwin').start('opaque-tls-node', 8766, undefined, 'https'); assert.ok(calls.at(-1)!.args.includes('transport=https')); await httpsAdvertisement.stop()
    await assert.rejects(make('darwin').start('opaque-node', 8766, undefined, 'http' as 'ssh'), /supported enrollment transport/)
    const tls = ephemeralTls(), secureRoot = path.join(root, 'tls'); await fs.mkdir(secureRoot, { mode: 0o700 })
    const certFile = path.join(secureRoot, 'cert.pem'), keyFile = path.join(secureRoot, 'key.pem')
    await fs.writeFile(certFile, tls.cert, { mode: 0o600 }); await fs.writeFile(keyFile, tls.key, { mode: 0o600 })
    const secure = await createClusterNodeServer({ config: { schema: 1, dataDir: secureRoot, host: '127.0.0.1', port: 0, tls: { certFile, keyFile }, advertise: true }, engine: new FakeEngine(), publisher: { start: async (_id, _port, _signal, transport) => { assert.equal(transport, 'https', 'TLS configuration selects the HTTPS discovery hint'); return { stop: async () => {} } } } })
    await secure.listen(); await secure.close()
    console.log('PASS opt-in discovery publisher with fake processes only: exact OS argv and opaque labels/TXT, no secrets or shell, SSH default and TLS HTTPS hint, registration acknowledgement, timeout/missing-tool/cancel refusal, physical publisher drain, service lifecycle and default-off behavior')
  } finally { await fs.rm(root, { recursive: true, force: true }) }
}
main().catch(error => { console.error(error); process.exitCode = 1 })
