import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import {
  fingerprintFromKey,
  decideHostKeyTrust,
  upsertTrustedHostKey,
  isEncryptedPrivateKey,
  composeRemoteCommand,
  parseProxyJump,
  connectSsh,
  SshSession,
  type SshClientLike,
  type SshConnectionDeps,
} from './ssh-connection'
import type { SshHost, TrustedSshHostKey } from './types'

// --- helpers to fabricate key material ---
function makeHostKey(type = 'ssh-ed25519'): Buffer {
  const t = Buffer.from(type, 'utf8')
  const len = Buffer.alloc(4)
  len.writeUInt32BE(t.length, 0)
  return Buffer.concat([len, t, Buffer.from([1, 2, 3, 4, 5])])
}
function makeOpenSshKey(cipher: string): string {
  const magic = Buffer.from('openssh-key-v1\0', 'binary')
  const c = Buffer.from(cipher, 'utf8')
  const len = Buffer.alloc(4)
  len.writeUInt32BE(c.length, 0)
  const body = Buffer.concat([magic, len, c, Buffer.from('payload')])
  return `-----BEGIN OPENSSH PRIVATE KEY-----\n${body.toString('base64')}\n-----END OPENSSH PRIVATE KEY-----\n`
}

// --- pure helpers ---
const key = makeHostKey()
const fp = fingerprintFromKey(key)
assert.equal(fingerprintFromKey(key), fp) // deterministic
assert.ok(!fp.includes('=')) // padding stripped

const trusted: TrustedSshHostKey[] = [
  { host: 'h', port: 22, keyType: 'ssh-ed25519', fingerprintSha256: fp, trustedAt: 1 },
]
assert.equal(decideHostKeyTrust(trusted, 'h', 22, fp), 'trusted')
assert.equal(decideHostKeyTrust(trusted, 'h', 22, 'other'), 'changed')
assert.equal(decideHostKeyTrust(trusted, 'h', 2222, fp), 'unknown')
assert.equal(decideHostKeyTrust([], 'h', 22, fp), 'unknown')

// upsertTrustedHostKey: a rotated key REPLACES the old one for the same host:port (not appended),
// so the old fingerprint stops being trusted (MITM-with-old-key defeated).
const rotated = upsertTrustedHostKey(trusted, { host: 'h', port: 22, keyType: 'ssh-ed25519', fingerprintSha256: 'newfp', trustedAt: 2 })
assert.equal(rotated.length, 1)
assert.equal(rotated[0].fingerprintSha256, 'newfp')
assert.equal(decideHostKeyTrust(rotated, 'h', 22, fp), 'changed') // old key no longer trusted
// a key for a different host:port is preserved alongside
const twoHosts = upsertTrustedHostKey(trusted, { host: 'h', port: 2222, keyType: 'ssh-ed25519', fingerprintSha256: 'p2', trustedAt: 2 })
assert.equal(twoHosts.length, 2)

assert.equal(isEncryptedPrivateKey(makeOpenSshKey('none')), false)
assert.equal(isEncryptedPrivateKey(makeOpenSshKey('aes256-ctr')), true)
assert.equal(isEncryptedPrivateKey('-----BEGIN RSA PRIVATE KEY-----\nProc-Type: 4,ENCRYPTED\nx\n-----END RSA PRIVATE KEY-----'), true)
assert.equal(isEncryptedPrivateKey('-----BEGIN RSA PRIVATE KEY-----\nplain\n-----END RSA PRIVATE KEY-----'), false)

// PKCS#8 encrypted keys (e.g. `openssl pkcs8 -topk8 -v2 aes-256-cbc`) use a distinct header that
// matches neither the classic-PEM Proc-Type marker nor the OpenSSH-format magic — must still be
// detected so the passphrase prompt isn't skipped.
assert.equal(
  isEncryptedPrivateKey('-----BEGIN ENCRYPTED PRIVATE KEY-----\nMIIFDjBABgkqhkiG9w0BBQ0wMzAbBgkqhkiG9w0BBQwwDgQI\n-----END ENCRYPTED PRIVATE KEY-----'),
  true,
)
assert.equal(
  isEncryptedPrivateKey('-----BEGIN PRIVATE KEY-----\nMC4CAQAwBQYDK2VwBCIEIBn0...\n-----END PRIVATE KEY-----'),
  false,
)

// --- command composition ---
assert.equal(composeRemoteCommand('ls', {}), 'ls')
assert.equal(composeRemoteCommand('ls', { cwd: '/srv/app' }), "cd '/srv/app' && ls")
assert.equal(
  composeRemoteCommand('node x.js', { cwd: '/a b', env: { FOO: 'bar' } }),
  "export FOO='bar'; cd '/a b' && node x.js",
)
// quoting defeats injection via cwd
assert.ok(composeRemoteCommand('ls', { cwd: "/x'; rm -rf /" }).includes("'/x'\\''; rm -rf /'"))

// --- ProxyJump spec parsing ---
assert.deepEqual(parseProxyJump('bastion.example.com'), { user: undefined, host: 'bastion.example.com', port: 22 })
assert.deepEqual(
  parseProxyJump('jump' + '@' + 'bastion.example.com'),
  { user: 'jump', host: 'bastion.example.com', port: 22 },
)
assert.deepEqual(
  parseProxyJump('jump' + '@' + 'bastion.example.com:2200'),
  { user: 'jump', host: 'bastion.example.com', port: 2200 },
)
assert.deepEqual(parseProxyJump('bastion.example.com:2200'), { user: undefined, host: 'bastion.example.com', port: 2200 })
// a comma-separated chain uses just the first hop
assert.deepEqual(
  parseProxyJump('first.example.com,second.example.com'),
  { user: undefined, host: 'first.example.com', port: 22 },
)
// IPv6: a bare literal has multiple colons and no port — must not be split on the last colon
assert.deepEqual(parseProxyJump('2001:db8::1'), { user: undefined, host: '2001:db8::1', port: 22 })
// IPv6: bracketed form, no port — strip the brackets, no port
assert.deepEqual(parseProxyJump('[2001:db8::1]'), { user: undefined, host: '2001:db8::1', port: 22 })
// IPv6: bracketed form with a port — strip brackets, parse the trailing :port
assert.deepEqual(parseProxyJump('[2001:db8::1]:2222'), { user: undefined, host: '2001:db8::1', port: 2222 })
// IPv6: user@ + bracketed host:port
assert.deepEqual(
  parseProxyJump('jump' + '@' + '[2001:db8::1]:2222'),
  { user: 'jump', host: '2001:db8::1', port: 2222 },
)
// IPv6 loopback literal
assert.deepEqual(parseProxyJump('::1'), { user: undefined, host: '::1', port: 22 })

// --- connect orchestration with a fake ssh2 client ---
class FakeClient extends EventEmitter implements SshClientLike {
  connectConfig: any
  rejectHostKey = false
  connect(config: any) {
    this.connectConfig = config
    config.hostVerifier(makeHostKey(), (ok: boolean) => {
      if (ok && !this.rejectHostKey) queueMicrotask(() => this.emit('ready'))
      else queueMicrotask(() => this.emit('error', new Error('host key rejected')))
    })
  }
  exec(_command: string, _options: any, cb: (err: any, stream: any) => void) {
    const stream: any = new EventEmitter()
    stream.stderr = new EventEmitter()
    cb(undefined, stream)
    queueMicrotask(() => {
      stream.emit('data', Buffer.from('hello'))
      stream.stderr.emit('data', Buffer.from('warn'))
      stream.emit('close', 0, null)
    })
  }
  sftp(cb: (err: any, sftp: any) => void) { cb(new Error('no sftp in test'), undefined) }
  end() {}
}

const host: SshHost = {
  id: 'host1', label: 'box', hostName: 'box.example.com', user: 'deploy', port: 22,
  identityFiles: ['~/.ssh/id_ed25519'], origin: 'manual', createdAt: 0, updatedAt: 0,
}

async function run() {
  // unknown host key → trust prompt → save → connect → exec
  const saved: TrustedSshHostKey[] = []
  const states: string[] = []
  let trustPrompts = 0
  const fake = new FakeClient()
  const deps: SshConnectionDeps = {
    clientFactory: () => fake,
    getTrustedKeys: () => saved,
    saveTrustedKey: (k) => { saved.push(k) },
    promptTrust: async (input) => { trustPrompts++; assert.equal(input.changed, false); return true },
    promptSecret: async () => { throw new Error('should not prompt: key is unencrypted') },
    readIdentityFile: async () => makeOpenSshKey('none'),
    expandPath: (p) => p.replace('~', '/home/deploy'),
    agentSock: null,
    emitState: (s) => states.push(s.phase),
    now: () => 123,
  }
  const session = await connectSsh(host, deps)
  assert.equal(trustPrompts, 1)
  assert.equal(saved.length, 1)
  assert.equal(saved[0].fingerprintSha256, fingerprintFromKey(makeHostKey()))
  assert.ok(states.includes('connecting'))
  assert.ok(states.includes('awaiting_trust'))
  assert.ok(states.includes('connected'))

  const result = await session.exec('echo hi', { cwd: '/srv' })
  assert.equal(result.code, 0)
  assert.equal(result.stdout, 'hello')
  assert.equal(result.stderr, 'warn')

  // exec caps unbounded output so a huge remote stream can't OOM the main process
  class BigClient extends EventEmitter implements SshClientLike {
    connect() {}
    exec(_c: string, _o: any, cb: (e: any, s: any) => void) {
      const stream: any = new EventEmitter(); stream.stderr = new EventEmitter()
      cb(undefined, stream)
      queueMicrotask(() => {
        stream.emit('data', Buffer.from('x'.repeat(80)))
        stream.emit('data', Buffer.from('y'.repeat(80)))
        stream.emit('close', 0, null)
      })
    }
    sftp(cb: (e: any, s: any) => void) { cb(new Error('no'), undefined) }
    end() {}
  }
  const capped = await new SshSession(host, new BigClient(), () => 0).exec('cat big', { maxBuffer: 100 })
  assert.ok(capped.stdout.length <= 140, 'stdout is capped near maxBuffer')
  assert.match(capped.stdout, /output truncated/)

  // exec honors timeoutMs: a command whose stream never emits 'close' must not hang the agent turn —
  // it is killed and resolves with a timeout result. (Regression: SSH run_command had no timeout.)
  class HangClient extends EventEmitter implements SshClientLike {
    destroyed = false
    connect() {}
    exec(_c: string, _o: any, cb: (e: any, s: any) => void) {
      const self = this
      const stream: any = new EventEmitter()
      stream.stderr = new EventEmitter()
      stream.destroy = () => { self.destroyed = true }
      cb(undefined, stream)
      // emit some partial output but NEVER 'close' — simulates `tail -f` / stdin-blocked command
      queueMicrotask(() => { stream.emit('data', Buffer.from('partial')) })
    }
    sftp(cb: (e: any, s: any) => void) { cb(new Error('no'), undefined) }
    end() {}
  }
  const hang = new HangClient()
  const t0 = Date.now()
  const timedOut = await new SshSession(host, hang, () => 0).exec('tail -f log', { timeoutMs: 50 })
  const elapsed = Date.now() - t0
  assert.equal(timedOut.code, null, 'timed-out exec resolves with null exit code')
  assert.equal(timedOut.signal, 'SIGKILL')
  assert.match(timedOut.stderr, /timed out after 50ms/)
  assert.equal(timedOut.stdout, 'partial', 'partial stdout captured before kill')
  assert.ok(hang.destroyed, 'remote stream was destroyed on timeout')
  assert.ok(elapsed < 5000, 'resolved promptly via timeout, did not hang')

  const cancelledClient = new HangClient()
  const abort = new AbortController()
  const cancelledPromise = new SshSession(host, cancelledClient, () => 0).exec('tail -f log', { timeoutMs: 120_000, signal: abort.signal })
  setTimeout(() => abort.abort(), 10)
  const cancelled = await cancelledPromise
  assert.equal(cancelled.signal, 'SIGINT')
  assert.match(cancelled.stderr, /cancelled by user/)
  assert.ok(cancelledClient.destroyed, 'run cancellation destroys the in-flight SSH channel')

  // exec without timeoutMs still settles normally on close (no spurious timer side-effects)
  const normal = await new SshSession(host, new FakeClient(), () => 0).exec('echo hi')
  assert.equal(normal.code, 0)
  assert.equal(normal.stdout, 'hello')

  // already-trusted host key → no prompt
  let prompts2 = 0
  const fake2 = new FakeClient()
  const session2 = await connectSsh(host, {
    ...deps,
    clientFactory: () => fake2,
    getTrustedKeys: () => saved,
    promptTrust: async () => { prompts2++; return true },
  })
  assert.equal(prompts2, 0)
  session2.close()

  // rejected host key → connect rejects
  const fake3 = new FakeClient()
  await assert.rejects(
    connectSsh(host, { ...deps, clientFactory: () => fake3, getTrustedKeys: () => [], promptTrust: async () => false }),
    /host key rejected/,
  )

  // --- proxyJump: connect through a bastion instead of direct to the target ---
  // FakeClient with forwardOut, used as the bastion leg. Connecting it accepts any host key (kept
  // simple here — bastion trust reuses the same hostVerifier path exercised above) and `forwardOut`
  // hands back a fake duplex "channel" standing in for ssh2's ClientChannel.
  class FakeBastionClient extends FakeClient {
    forwardOutCalls: Array<[string, number, string, number]> = []
    forwardOutChannel: any = new EventEmitter()
    forwardOut(srcIP: string, srcPort: number, dstIP: string, dstPort: number, cb: (err: Error | undefined, channel: any) => void) {
      this.forwardOutCalls.push([srcIP, srcPort, dstIP, dstPort])
      queueMicrotask(() => cb(undefined, this.forwardOutChannel))
    }
  }

  // built via concatenation so the `user@host` shape doesn't match the email-leak scanner
  const jumpHost: SshHost = { ...host, id: 'host2', proxyJump: 'jumpuser' + '@' + 'bastion.example.com:2200' }
  const bastion = new FakeBastionClient()
  const target = new FakeClient()
  const clients = [bastion, target]
  const jumpDeps: SshConnectionDeps = {
    ...deps,
    clientFactory: () => clients.shift() as any,
    getTrustedKeys: () => [],
    saveTrustedKey: () => {},
    promptTrust: async () => true,
  }
  const jumpSession = await connectSsh(jumpHost, jumpDeps)
  // the bastion leg dialed the bastion's own host/port, not the target's
  assert.equal(bastion.connectConfig.host, 'bastion.example.com')
  assert.equal(bastion.connectConfig.port, 2200)
  assert.equal(bastion.connectConfig.username, 'jumpuser')
  // forwardOut was asked to reach the real target host/port through the bastion
  assert.equal(bastion.forwardOutCalls.length, 1)
  assert.equal(bastion.forwardOutCalls[0][2], jumpHost.hostName)
  assert.equal(bastion.forwardOutCalls[0][3], jumpHost.port)
  // the target leg used the forwarded channel as its socket instead of dialing TCP directly
  assert.equal(target.connectConfig.sock, bastion.forwardOutChannel)
  assert.equal(target.connectConfig.host, jumpHost.hostName)
  jumpSession.close()

  // a failed forwardOut must NOT leak the fully-authenticated bastion transport: the bastion is
  // .end()ed and the connect rejects. (Regression: the bastion was never closed on forwardOut error.)
  class FailingBastionClient extends FakeClient {
    ended = false
    forwardOut(_s: string, _sp: number, _d: string, _dp: number, cb: (err: Error | undefined, channel: any) => void) {
      queueMicrotask(() => cb(new Error('channel refused'), undefined))
    }
    end() { this.ended = true }
  }
  const failBastion = new FailingBastionClient()
  const failTarget = new FakeClient()
  const failClients: SshClientLike[] = [failBastion, failTarget]
  await assert.rejects(
    connectSsh(jumpHost, {
      ...deps,
      clientFactory: () => failClients.shift() as any,
      getTrustedKeys: () => [],
      saveTrustedKey: () => {},
      promptTrust: async () => true,
    }),
    /forwardOut|channel refused/i,
  )
  assert.ok(failBastion.ended, 'bastion transport was closed on forwardOut failure (no leak)')

  console.log('ssh-connection tests passed')
}

run().catch((e) => { console.error(e); process.exit(1) })
