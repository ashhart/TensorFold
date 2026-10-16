import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import { configEntriesToHosts, mergeHostSources, SshManager, SSH_STATUS_CHANNEL, SSH_TRUST_PROMPT_CHANNEL, SSH_SECRET_PROMPT_CHANNEL } from './ssh'
import type { SshHost, SshConnectionState } from '../../agent/types'
import type { SshClientLike } from '../../agent/ssh-connection'

const CONFIG = `
Host prod
  HostName prod.example.com
  User deploy
  Port 2200
  IdentityFile ~/.ssh/prod_ed25519

Host staging
  HostName 10.0.0.9
`

const hosts = configEntriesToHosts(CONFIG)
assert.deepEqual(hosts.map((h) => h.label), ['prod', 'staging'])
const prod = hosts.find((h) => h.label === 'prod')!
assert.equal(prod.id, 'config:prod')
assert.equal(prod.hostName, 'prod.example.com')
assert.equal(prod.user, 'deploy')
assert.equal(prod.port, 2200)
assert.deepEqual(prod.identityFiles, ['~/.ssh/prod_ed25519'])
assert.equal(prod.origin, 'config')
// staging has no Port → default 22, hostName from config
const staging = hosts.find((h) => h.label === 'staging')!
assert.equal(staging.port, 22)
assert.equal(staging.hostName, '10.0.0.9')

// library entry overrides the config alias of the same name; others merge in
const library: SshHost[] = [
  { id: 'lib1', label: 'prod', configAlias: 'prod', hostName: 'prod.internal', port: 22, identityFiles: [], origin: 'manual', createdAt: 1, updatedAt: 1 },
  { id: 'lib2', label: 'box', hostName: 'box.example.com', port: 22, identityFiles: [], origin: 'manual', createdAt: 1, updatedAt: 1 },
]
const merged = mergeHostSources(hosts, library)
// library 'prod' wins (only one prod), config 'staging' still present, manual 'box' present
assert.deepEqual(merged.map((h) => h.label).sort(), ['box', 'prod', 'staging'])
const mergedProd = merged.filter((h) => h.label === 'prod')
assert.equal(mergedProd.length, 1)
assert.equal(mergedProd[0].hostName, 'prod.internal') // library, not config

// --- SshManager.connect(): a dropped socket after a trust prompt was registered must not
// leak the pendingTrust resolver or leave the renderer's trust dialog stranded. ---
async function run() {
  // FakeClient: hostVerifier fires (registers a trust prompt) but the prompt is never answered;
  // instead the transport errors out (simulating a dropped/timed-out TCP socket).
  class DropAfterTrustPromptClient extends EventEmitter implements SshClientLike {
    connect(config: any) {
      config.hostVerifier(makeHostKey(), () => { /* never invoked by the test: socket drops first */ })
      queueMicrotask(() => this.emit('error', new Error('socket hang up')))
    }
    exec(_command: string, _options: any, cb: (err: any, stream: any) => void) {
      cb(new Error('not connected'), undefined)
    }
    sftp(cb: (err: any, sftp: any) => void) { cb(new Error('not connected'), undefined) }
    end() {}
  }

  function makeHostKey(): Buffer {
    const t = Buffer.from('ssh-ed25519', 'utf8')
    const len = Buffer.alloc(4)
    len.writeUInt32BE(t.length, 0)
    return Buffer.concat([len, t, Buffer.from([1, 2, 3, 4, 5])])
  }

  const host: SshHost = {
    id: 'host1', label: 'box', hostName: 'box.example.com', user: 'deploy', port: 22,
    identityFiles: [], origin: 'manual', createdAt: 0, updatedAt: 0,
  }

  const sentEvents: Array<{ channel: string; payload: any }> = []
  const manager = new SshManager({
    getHosts: () => [host],
    getTrustedKeys: () => [],
    saveTrustedKey: () => {},
    sendEvent: (channel, payload) => { sentEvents.push({ channel, payload }) },
    clientFactory: () => new DropAfterTrustPromptClient(),
    agentSock: () => null,
  })

  await assert.rejects(manager.ensureConnected('host1'), /socket hang up/)

  // The trust prompt was sent to the renderer...
  const trustPromptEvents = sentEvents.filter((e) => e.channel === SSH_TRUST_PROMPT_CHANNEL)
  assert.equal(trustPromptEvents.length, 1, 'a trust prompt was emitted before the drop')
  const promptId = trustPromptEvents[0].payload.promptId as string
  assert.ok(promptId)

  // ...but the connection then failed. The pendingTrust resolver registered for that prompt must be
  // cleaned up (not left dangling).
  const pendingTrust: Map<string, unknown> = (manager as any).pendingTrust
  assert.equal(pendingTrust.size, 0, 'pendingTrust must not leak an entry after connectSsh rejects')
  // A late renderer answer for the now-dead prompt is a safe no-op (no throw, nothing left to resolve).
  manager.resolveTrust(promptId, true)

  // The appropriate failure status was emitted on SSH_STATUS_CHANNEL, with the hostId the renderer
  // needs to clear a stale sshTrustPrompt for this host.
  const statusEvents = sentEvents.filter((e) => e.channel === SSH_STATUS_CHANNEL)
  const failureState = statusEvents.map((e) => e.payload as SshConnectionState).find((s) => s.hostId === 'host1' && (s.phase === 'error' || s.phase === 'disconnected'))
  assert.ok(failureState, 'a failure status was emitted for host1 after the trust prompt leaked')

  console.log('ssh manager connect-failure trust-prompt cleanup test passed')

  // --- the same cleanup must apply to a passphrase/password (secret) prompt: a transport drop
  // during keyboard-interactive auth must not leak the pendingSecrets resolver or strand the
  // renderer's secret dialog. ---
  // Client: host key is accepted (so we get past trust), then it raises a keyboard-interactive
  // prompt (which registers a secret prompt that the test never answers) and the socket drops.
  class DropDuringSecretPromptClient extends EventEmitter implements SshClientLike {
    connect(config: any) {
      config.hostVerifier(makeHostKey(), (ok: boolean) => {
        if (!ok) return
        // emit a keyboard-interactive challenge → connectSsh awaits promptSecret for the answer
        queueMicrotask(() => {
          this.emit('keyboard-interactive', '', '', '', [{ prompt: 'Password:', echo: false }], () => {})
          // ...then the transport drops before the renderer ever answers
          queueMicrotask(() => this.emit('error', new Error('socket hang up')))
        })
      })
    }
    exec(_command: string, _options: any, cb: (err: any, stream: any) => void) { cb(new Error('not connected'), undefined) }
    sftp(cb: (err: any, sftp: any) => void) { cb(new Error('not connected'), undefined) }
    end() {}
  }

  const secretEvents: Array<{ channel: string; payload: any }> = []
  let secretManager: SshManager
  secretManager = new SshManager({
    getHosts: () => [host],
    getTrustedKeys: () => [], // unknown key → trust prompt, auto-answered below
    saveTrustedKey: () => {},
    sendEvent: (channel, payload) => {
      secretEvents.push({ channel, payload })
      // auto-trust the presented key so we proceed to the keyboard-interactive (secret) prompt path
      if (channel === SSH_TRUST_PROMPT_CHANNEL) queueMicrotask(() => secretManager.resolveTrust((payload as { promptId: string }).promptId, true))
    },
    clientFactory: () => new DropDuringSecretPromptClient(),
    readIdentityFile: async () => null, // no identity → keyboard-interactive fallback
    agentSock: () => null,
  })

  await assert.rejects(secretManager.ensureConnected('host1'), /socket hang up/)

  const secretPromptEvents = secretEvents.filter((e) => e.channel === SSH_SECRET_PROMPT_CHANNEL)
  assert.equal(secretPromptEvents.length, 1, 'a secret prompt was emitted before the drop')
  const secretPromptId = secretPromptEvents[0].payload.promptId as string
  assert.ok(secretPromptId)

  const pendingSecrets: Map<string, unknown> = (secretManager as any).pendingSecrets
  assert.equal(pendingSecrets.size, 0, 'pendingSecrets must not leak an entry after connectSsh rejects')
  // a late renderer answer for the dead prompt is a safe no-op
  secretManager.resolveSecret(secretPromptId, 'hunter2')

  console.log('ssh manager connect-failure secret-prompt cleanup test passed')

  // A headless host has no renderer that can answer trust/password dialogs. It must fail
  // immediately with an actionable error instead of parking an unresolved Promise forever.
  const headlessEvents: Array<{ channel: string; payload: any }> = []
  const headlessTrust = new SshManager({
    getHosts: () => [host],
    getTrustedKeys: () => [],
    saveTrustedKey: () => {},
    sendEvent: (channel, payload) => { headlessEvents.push({ channel, payload }) },
    clientFactory: () => new DropAfterTrustPromptClient(),
    agentSock: () => null,
    interactivePrompts: () => false,
  })
  await assert.rejects(headlessTrust.ensureConnected('host1'), /interactive prompts are unavailable in headless mode/)
  assert.equal(headlessEvents.some((event) => event.channel === SSH_TRUST_PROMPT_CHANNEL), false, 'headless mode never emits an unanswerable trust dialog')
  assert.equal((headlessTrust as any).pendingTrust.size, 0, 'headless trust refusal leaves no parked resolver')

  const keyHost = { ...host, id: 'key-host', identityFiles: ['~/.ssh/encrypted'] }
  const headlessSecret = new SshManager({
    getHosts: () => [keyHost],
    getTrustedKeys: () => [],
    saveTrustedKey: () => {},
    sendEvent: (channel, payload) => { headlessEvents.push({ channel, payload }) },
    clientFactory: () => new DropAfterTrustPromptClient(),
    readIdentityFile: async () => '-----BEGIN ENCRYPTED PRIVATE KEY-----\nopaque\n-----END ENCRYPTED PRIVATE KEY-----',
    interactivePrompts: () => false,
  })
  await assert.rejects(headlessSecret.ensureConnected('key-host'), /SSH passphrase is required.*interactive prompts are unavailable in headless mode/)
  assert.equal(headlessEvents.some((event) => event.channel === SSH_SECRET_PROMPT_CHANNEL), false, 'headless mode never emits an unanswerable secret dialog')
  assert.equal((headlessSecret as any).pendingSecrets.size, 0, 'headless secret refusal leaves no parked resolver')

  console.log('ssh manager headless prompts fail fast tests passed')

  // A live renderer can still disappear or fail to answer. Trust and secret prompts therefore
  // have their own bounded lifetime, independent of ssh2's transport timeout.
  class WaitOnTrustPromptClient extends EventEmitter implements SshClientLike {
    connect(config: any) { config.hostVerifier(makeHostKey(), () => {}) }
    exec(_command: string, _options: any, cb: (err: any, stream: any) => void) { cb(new Error('not connected'), undefined) }
    sftp(cb: (err: any, sftp: any) => void) { cb(new Error('not connected'), undefined) }
    end() {}
  }
  class WaitOnKeyboardPromptClient extends EventEmitter implements SshClientLike {
    connect(config: any) {
      config.hostVerifier(makeHostKey(), (ok: boolean) => {
        if (ok) queueMicrotask(() => this.emit('keyboard-interactive', '', '', '', [{ prompt: 'Password:', echo: false }], () => {}))
      })
    }
    exec(_command: string, _options: any, cb: (err: any, stream: any) => void) { cb(new Error('not connected'), undefined) }
    sftp(cb: (err: any, sftp: any) => void) { cb(new Error('not connected'), undefined) }
    end() {}
  }

  const keepAlive = setInterval(() => {}, 1_000)
  try {
    const trustTimeout = new SshManager({
      getHosts: () => [host],
      getTrustedKeys: () => [],
      saveTrustedKey: () => {},
      sendEvent: () => {},
      clientFactory: () => new WaitOnTrustPromptClient(),
      agentSock: () => null,
      promptTimeoutMs: 10,
    })
    await assert.rejects(trustTimeout.ensureConnected('host1'), /SSH host-key prompt timed out after 10ms/)
    assert.equal((trustTimeout as any).pendingTrust.size, 0, 'timed-out trust prompt resolver is removed')

    const secretTimeout = new SshManager({
      getHosts: () => [keyHost],
      getTrustedKeys: () => [],
      saveTrustedKey: () => {},
      sendEvent: () => {},
      clientFactory: () => new WaitOnTrustPromptClient(),
      readIdentityFile: async () => '-----BEGIN ENCRYPTED PRIVATE KEY-----\nopaque\n-----END ENCRYPTED PRIVATE KEY-----',
      promptTimeoutMs: 10,
    })
    await assert.rejects(secretTimeout.ensureConnected('key-host'), /SSH credential prompt timed out after 10ms/)
    assert.equal((secretTimeout as any).pendingSecrets.size, 0, 'timed-out secret prompt resolver is removed')

    let passwordTimeout!: SshManager
    passwordTimeout = new SshManager({
      getHosts: () => [host],
      getTrustedKeys: () => [],
      saveTrustedKey: () => {},
      sendEvent: (channel, payload) => {
        if (channel === SSH_TRUST_PROMPT_CHANNEL) queueMicrotask(() => passwordTimeout.resolveTrust((payload as { promptId: string }).promptId, true))
      },
      clientFactory: () => new WaitOnKeyboardPromptClient(),
      readIdentityFile: async () => null,
      agentSock: () => null,
      promptTimeoutMs: 10,
    })
    await assert.rejects(passwordTimeout.ensureConnected('host1'), /SSH credential prompt timed out after 10ms/)
    assert.equal((passwordTimeout as any).pendingSecrets.size, 0, 'keyboard-interactive timeout terminates the connection and removes its resolver')
  } finally {
    clearInterval(keepAlive)
  }

  console.log('ssh manager trust + secret prompt timeout tests passed')

  // Window teardown cancels every outstanding prompt immediately. Exercise both prompt kinds in
  // one manager so neither resolver can survive a renderer lifecycle transition.
  const disappearingEvents = new Set<string>()
  let disappearingManager!: SshManager
  disappearingManager = new SshManager({
    getHosts: () => [host, keyHost],
    getTrustedKeys: () => [],
    saveTrustedKey: () => {},
    sendEvent: (channel) => {
      if (channel !== SSH_TRUST_PROMPT_CHANNEL && channel !== SSH_SECRET_PROMPT_CHANNEL) return
      disappearingEvents.add(channel)
      if (disappearingEvents.size === 2) queueMicrotask(() => disappearingManager.cancelPendingPrompts('SSH prompt cancelled because the app window closed.'))
    },
    clientFactory: () => new WaitOnTrustPromptClient(),
    readIdentityFile: async () => '-----BEGIN ENCRYPTED PRIVATE KEY-----\nopaque\n-----END ENCRYPTED PRIVATE KEY-----',
    agentSock: () => null,
  })
  const closedResults = await Promise.allSettled([
    disappearingManager.ensureConnected('host1'),
    disappearingManager.ensureConnected('key-host'),
  ])
  assert.deepEqual(closedResults.map((result) => result.status), ['rejected', 'rejected'])
  for (const result of closedResults) {
    assert.match(String(result.status === 'rejected' ? result.reason?.message : ''), /app window closed/)
  }
  assert.equal((disappearingManager as any).pendingTrust.size, 0, 'window teardown removes trust resolver')
  assert.equal((disappearingManager as any).pendingSecrets.size, 0, 'window teardown removes secret resolver')

  console.log('ssh manager renderer-disappearance cleanup tests passed')
}

run().catch((e) => { console.error(e); process.exit(1) })

console.log('ssh manager helper tests passed')
