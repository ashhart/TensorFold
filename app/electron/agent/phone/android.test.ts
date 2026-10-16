import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { deflateSync } from 'node:zlib'
import { AndroidPhoneAdapter, AndroidPhoneError, parseAndroidDevices, parseAndroidLayout, parseAndroidLockState, runAndroidCommand, validateAndroidEndpoint, validateAndroidPng } from './android'
import type { AndroidAction, AndroidCommand, AndroidCommandRunner } from './android'

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-android-adapter-'))
const adb = path.join(root, 'adb')
await fs.writeFile(adb, '#!/bin/sh\nexit 0\n', { mode: 0o700 })
const serial = 'SYNTHETIC_USB_A'
const layout = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><hierarchy rotation="0"><node text="Home &amp; tools" content-desc="Open tools" resource-id="dev.example.phone:id/tools" class="android.widget.Button" package="dev.example.phone" bounds="[0,0][2,2]" clickable="true" focused="true" enabled="true" password="false" /></hierarchy>'
const unlocked = 'WINDOW MANAGER POLICY STATE\n  mAwake=true\n  KeyguardServiceDelegate\n    showing=false\n    inputRestricted=false\n    secure=true\n    screenState=2\n'
const deviceOutput = `List of devices attached\n${serial}\tdevice usb:synthetic model:Example_Handset product:example transport_id:1\nSYNTHETIC_USB_B\tdevice usb:synthetic model:Second_Handset transport_id:2\nSYNTHETIC_UNAUTHORIZED\tunauthorized usb:synthetic\nSYNTHETIC_OFFLINE\toffline usb:synthetic\nadb-SYNTHETIC-pair._adb-tls-connect._tcp\tdevice product:example model:Wireless_Handset transport_id:3\nemulator-5554\tdevice model:Example_Emulator transport_id:4\n`

function checksum(bytes: Buffer): number {
  let value = 0xffffffff
  for (const b of bytes) { value ^= b; for (let bit = 0; bit < 8; bit++) value = (value >>> 1) ^ ((value & 1) ? 0xedb88320 : 0) }
  return (value ^ 0xffffffff) >>> 0
}
function chunk(type: string, body: Buffer): Buffer {
  const result = Buffer.alloc(body.length + 12)
  result.writeUInt32BE(body.length, 0); result.write(type, 4, 'ascii'); body.copy(result, 8)
  result.writeUInt32BE(checksum(result.subarray(4, body.length + 8)), body.length + 8)
  return result
}
function pngFixture(width = 2, height = 2, raw?: Buffer): Buffer {
  const header = Buffer.alloc(13)
  header.writeUInt32BE(width, 0); header.writeUInt32BE(height, 4); header[8] = 8; header[9] = 6
  const rows = raw ?? Buffer.alloc(height * (width * 4 + 1))
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', header), chunk('IDAT', deflateSync(rows)), chunk('IEND', Buffer.alloc(0))])
}
const png = pngFixture()
const ok = (stdout: Buffer | string = '', stderr: Buffer | string = '', code = 0) => ({ code, stdout: Buffer.isBuffer(stdout) ? stdout : Buffer.from(stdout), stderr: Buffer.isBuffer(stderr) ? stderr : Buffer.from(stderr) })
function fixture(overrides: { xml?: string; policy?: string; devices?: string; override?: AndroidCommandRunner } = {}) {
  const requests: AndroidCommand[] = []
  const run: AndroidCommandRunner = async (command) => {
    requests.push(command)
    if (command.args[0] === 'version') return ok('Android Debug Bridge version 1.0.41\nVersion 37.0.0-example\n')
    if (overrides.override) return overrides.override(command)
    const args = command.args.slice(4)
    if (args.join(' ') === 'devices -l') return ok(overrides.devices ?? deviceOutput)
    if (args.includes('policy')) return ok(overrides.policy ?? unlocked)
    if (args.includes('uiautomator')) return ok('UI hierarchy dumped\n')
    if (args.includes('cat')) return ok(overrides.xml ?? layout)
    if (args.includes('screencap')) return ok(png)
    if (args[0] === 'pair') return ok('Successfully paired to private device\n')
    if (args[0] === 'connect') return ok('connected to private device\n')
    if (args.includes('resolve-activity')) return ok('dev.example.phone/.MainActivity\n')
    return ok()
  }
  return { adapter: new AndroidPhoneAdapter({ adbPath: adb, run }), requests }
}
function hasCode(code: string) { return (error: unknown) => error instanceof AndroidPhoneError && error.code === code }

try {
  const parsed = parseAndroidDevices(deviceOutput)
  assert.equal(parsed.length, 6)
  assert.deepEqual(parsed.map((d) => [d.name, d.state, d.transport]), [['Example Handset', 'ready', 'usb'], ['Second Handset', 'ready', 'usb'], ['Android phone', 'unauthorized', 'usb'], ['Android phone', 'offline', 'usb'], ['Wireless Handset', 'ready', 'wifi'], ['Example Emulator', 'ready', 'emulator']])
  assert.equal(parseAndroidDevices(`${deviceOutput}\n${serial}\tdevice model:Duplicated\n-unsafe\tdevice\nmalicious;command\tdevice`).length, 6)
  assert.throws(() => parseAndroidDevices('a'.repeat(70_000)), hasCode('output-limit'))

  for (const endpoint of ['127.0.0.1:41234', '8.8.8.8:41234', '0.0.0.0:41234', 'localhost:41234', 'https://192.168.0.2:41234', '192.168.0.2:5555', '192.168.0.2:5037', '192.168.0.2:80', '192.168.0.2:65536', '192.168.0.255:41234', '192.168.0.2:41234;whoami', '192.168.0.2:41234\n', ' [fd00::2]:41234', '[::1]:41234', '[2001:4860::1]:41234']) assert.throws(() => validateAndroidEndpoint(endpoint), hasCode('invalid-input'))
  assert.equal(validateAndroidEndpoint('10.0.0.2:41234'), '10.0.0.2:41234')
  assert.equal(validateAndroidEndpoint('172.16.0.2:41234'), '172.16.0.2:41234')
  assert.equal(validateAndroidEndpoint('[fd00::2]:41234'), '[fd00::2]:41234')

  assert.equal(parseAndroidLockState(unlocked), 'unlocked')
  assert.equal(parseAndroidLockState(unlocked.replace('showing=false', 'showing=true')), 'locked')
  assert.equal(parseAndroidLockState(unlocked.replace('inputRestricted=false', 'inputRestricted=true')), 'locked')
  assert.equal(parseAndroidLockState(unlocked.replace('screenState=2', 'screenState=0')), 'locked')
  assert.equal(parseAndroidLockState('mShowingLockscreen=true\nmKeyguardShowing=false'), 'locked')
  assert.equal(parseAndroidLockState('mShowingLockscreen=false\nmAwake=false'), 'locked')
  assert.equal(parseAndroidLockState('mShowingLockscreen=false\nmAwake=true'), 'unlocked')
  assert.equal(parseAndroidLockState('unknown manufacturer policy'), 'unknown')
  assert.equal(parseAndroidLockState('x'.repeat(300_000)), 'unknown')

  const screen = parseAndroidLayout(layout)
  assert.equal(screen.text, 'Home & tools\nOpen tools')
  assert.equal(screen.activePackage, 'dev.example.phone')
  assert.deepEqual(screen.nodes[0].bounds, { x: 0, y: 0, width: 2, height: 2 })
  assert.equal(screen.nodes[0].clickable, true)
  assert.equal(screen.nodes[0].focused, true)
  const protectedScreen = parseAndroidLayout(layout.replace('password="false"', 'password="true"').replace('Home &amp; tools', 'SYNTHETIC_SECRET'))
  assert.equal(protectedScreen.secure, true)
  assert.equal(protectedScreen.nodes[0].text, '[protected]')
  assert.equal(protectedScreen.nodes[0].description, '[protected]')
  assert.ok(!JSON.stringify(protectedScreen).includes('SYNTHETIC_SECRET'))
  const inherited = parseAndroidLayout('<hierarchy><node password="true" bounds="[0,0][2,2]"><node text="SYNTHETIC_SECRET" bounds="[0,0][1,1]" /></node></hierarchy>')
  assert.equal(inherited.nodes[1].text, '[protected]')
  assert.equal(parseAndroidLayout(layout.replace('id/tools', 'id/pin_entry')).secure, true)
  assert.equal(parseAndroidLayout(layout.replace('Home &amp; tools', '&#x1f600;')).nodes[0].text, '😀')
  for (const bad of [
    '<!DOCTYPE hierarchy [<!ENTITY secret SYSTEM "file:///private/example">]><hierarchy><node text="&secret;" bounds="[0,0][1,1]" /></hierarchy>',
    layout.replace('Home &amp; tools', '&unknown;'), layout.replace('Home &amp; tools', '&#x110000;'), layout.replace('Home &amp; tools', '&#xD800;'), layout.replace('Home &amp; tools', '&#0;'),
    layout.replace('bounds="[0,0][2,2]"', 'bounds="[2,2][0,0]"'), layout.replace('clickable="true"', 'clickable="yes"'), layout.replace('password="false"', 'password="false" password="true"'),
    layout.replace('</hierarchy>', '</node></hierarchy>'), layout + '<hierarchy/>', '<hierarchy><unsupported /></hierarchy>', '<hierarchy><node bounds="[0,0][1,1]">trailing</node></hierarchy>',
  ]) assert.throws(() => parseAndroidLayout(bad), hasCode('invalid-layout'))
  assert.throws(() => parseAndroidLayout(' '.repeat(600_000)), hasCode('output-limit'))

  assert.deepEqual(validateAndroidPng(png), { width: 2, height: 2 })
  assert.throws(() => validateAndroidPng(Buffer.from('not PNG')), hasCode('invalid-screen'))
  assert.throws(() => validateAndroidPng(png.subarray(0, png.length - 1)), hasCode('invalid-screen'))
  const corrupt = Buffer.from(png); corrupt[44] ^= 1
  assert.throws(() => validateAndroidPng(corrupt), hasCode('invalid-screen'))
  assert.throws(() => validateAndroidPng(Buffer.concat([png, Buffer.from('trailing')])), hasCode('invalid-screen'))
  assert.throws(() => validateAndroidPng(pngFixture(9000, 1, Buffer.alloc(1))), hasCode('invalid-screen'))
  assert.throws(() => validateAndroidPng(pngFixture(2, 2, Buffer.alloc(1000))), hasCode('invalid-screen'))
  const badFilter = Buffer.alloc(18); badFilter[0] = 5
  assert.throws(() => validateAndroidPng(pngFixture(2, 2, badFilter)), hasCode('invalid-screen'))

  {
    const { adapter, requests } = fixture()
    const resolved = await adapter.resolve()
    assert.equal(resolved?.version, '1.0.41')
    assert.equal((await adapter.devices()).length, 6)
    const observation = await adapter.observe(serial)
    assert.deepEqual({ width: observation.width, height: observation.height, lock: observation.lockState, secure: observation.secure }, { width: 2, height: 2, lock: 'unlocked', secure: false })
    assert.equal(observation.nodes.length, 1)
    assert.ok(observation.png.equals(png))
    const deviceRequests = requests.filter((request) => request.args.includes('-s'))
    assert.ok(deviceRequests.every((request) => request.args[5] === serial), 'every phone-specific command pins the exact selected serial, with a second ready device present')
    const tempRequests = requests.filter((request) => request.args.some((arg) => arg.startsWith('/data/local/tmp/tensorfold-ui-')))
    const temporary = tempRequests[0].args.at(-1)!
    assert.ok(/^\/data\/local\/tmp\/tensorfold-ui-[0-9a-f-]{36}\.xml$/.test(temporary))
    assert.ok(tempRequests.every((request) => request.args.at(-1) === temporary))
    assert.ok(tempRequests.some((request) => request.args.includes('chmod') && request.args.includes('600')))
    assert.equal(tempRequests.at(-1)?.args.slice(6, 9).join(' '), 'shell rm -f')
    assert.ok(requests.every((request) => request.args[0] === 'version' || request.args.slice(0, 4).join(' ') === '-H 127.0.0.1 -P 5037'), 'external adb server environment is never selected')
  }
  for (const [selected, code] of [['SYNTHETIC_MISSING', 'disconnected'], ['SYNTHETIC_UNAUTHORIZED', 'unauthorized'], ['SYNTHETIC_OFFLINE', 'offline']] as const) {
    const { adapter, requests } = fixture()
    await assert.rejects(() => adapter.observe(selected), hasCode(code))
    assert.ok(!requests.some((r) => r.args.includes('screencap') || r.args.includes('uiautomator')))
  }
  for (const [policy, code] of [[unlocked.replace('showing=false', 'showing=true'), 'locked'], ['unsupported policy', 'lock-unknown']] as const) {
    const { adapter, requests } = fixture({ policy })
    await assert.rejects(() => adapter.observe(serial), hasCode(code))
    assert.ok(!requests.some((r) => r.args.includes('screencap') || r.args.includes('uiautomator')))
  }
  {
    const { adapter, requests } = fixture({ xml: layout.replace('password="false"', 'password="true"') })
    await assert.rejects(() => adapter.observe(serial), hasCode('secure-screen'))
    assert.ok(requests.some((r) => r.args.includes('rm')), 'protected-screen XML temp file is still removed')
    assert.ok(!requests.some((r) => r.args.includes('screencap')), 'protected-screen screenshots are not captured')
  }
  {
    const aborter = new AbortController()
    const { adapter, requests } = fixture({ override: async (command) => {
      const args = command.args
      if (args.includes('devices')) return ok(deviceOutput)
      if (args.includes('policy')) return ok(unlocked)
      if (args.includes('uiautomator')) { aborter.abort(); return ok() }
      return ok()
    } })
    await assert.rejects(() => adapter.observe(serial, aborter.signal), hasCode('aborted'))
    const cleanup = requests.find((r) => r.args.includes('rm'))
    assert.ok(cleanup, 'cancellation removes the unique temporary screen file')
    assert.equal(cleanup?.signal, undefined, 'cleanup has an independent timeout after cancellation')
    assert.ok(!requests.some((r) => r.args.includes('screencap') || r.args.includes('cat')))
  }
  {
    let policyReads = 0
    const { adapter, requests } = fixture({ override: async (command) => {
      const args = command.args
      if (args.includes('devices')) return ok(deviceOutput)
      if (args.includes('policy')) return ok(++policyReads === 1 ? unlocked : unlocked.replace('showing=false', 'showing=true'))
      if (args.includes('cat')) return ok(layout)
      return ok()
    } })
    await assert.rejects(() => adapter.observe(serial), hasCode('locked'))
    assert.ok(!requests.some((r) => r.args.includes('screencap')), 'locking during layout observation stops screenshot capture')
  }
  {
    let cleanupCalls = 0
    const { adapter, requests } = fixture({ override: async (command) => {
      const args = command.args
      if (args.includes('devices')) return ok(deviceOutput)
      if (args.includes('policy')) return ok(unlocked)
      if (args.includes('cat')) return ok(layout)
      if (args.includes('rm') && ++cleanupCalls === 1) return ok('', 'temporary transport failure', 1)
      if (args.includes('screencap')) return ok(png)
      return ok()
    } })
    await assert.rejects(() => adapter.observe(serial), hasCode('command-failed'))
    assert.ok(!requests.some((r) => r.args.includes('screencap')), 'a failed private-file cleanup prevents capture')
    const failedRemoval = requests.find((r) => r.args.includes('rm'))!.args.at(-1)
    await adapter.observe(serial)
    const removals = requests.filter((r) => r.args.includes('rm'))
    assert.equal(removals[1].args.at(-1), failedRemoval, 'the next request retries only its own pending temporary file')
    assert.equal(removals.length, 3)
  }
  {
    const { adapter, requests } = fixture()
    await adapter.pair({ endpoint: '10.0.0.2:41234', code: '123456' })
    const request = requests.find((r) => r.args.includes('pair'))!
    assert.equal(request.stdin, '123456\n')
    assert.ok(!JSON.stringify(request.args).includes('123456'), 'pairing code must never appear in argv')
    await adapter.connect('10.0.0.2:41235')
    await adapter.disconnect('10.0.0.2:41235')
    await assert.rejects(() => adapter.pair({ endpoint: '10.0.0.2:41234', code: '123456;id' }), hasCode('invalid-input'))
    await assert.rejects(() => adapter.connect('8.8.8.8:41235'), hasCode('invalid-input'))
    assert.ok(!requests.some((r) => r.args.includes('tcpip') || r.args.includes('kill-server')))
  }
  {
    const { adapter, requests } = fixture()
    for (const action of [
      { kind: 'tap', x: 1, y: 1 }, { kind: 'swipe', x1: 0, y1: 0, x2: 1, y2: 1 },
      { kind: 'type', text: "$(id); 'quoted' & 50% | > < `id`" }, { kind: 'key', key: 'home' }, { kind: 'key', key: 'back' }, { kind: 'key', key: 'enter' },
      { kind: 'launch', packageName: 'dev.example.phone' },
    ] satisfies AndroidAction[]) await adapter.act(serial, action, { width: 2, height: 2 })
    const typing = requests.find((r) => r.args.includes('text'))!
    assert.equal(typing.args.at(-1), "'$(id);%s'\\''quoted'\\''%s&%s50%%s|%s>%s<%s`id`'")
    assert.equal(typing.args.length, 10, 'typing remains one explicitly quoted remote-shell argument')
    if (process.platform !== 'win32') {
      const shellParsed = await runAndroidCommand({ executable: '/bin/sh', args: ['-c', `printf '%s' ${typing.args.at(-1)!}`], timeoutMs: 3000, maxOutputBytes: 1024 })
      assert.equal(shellParsed.stdout.toString(), "$(id);%s'quoted'%s&%s50%%s|%s>%s<%s`id`", 'remote quoting keeps substitutions, separators and redirections literal')
    }
    assert.ok(requests.filter((r) => r.args.includes('input')).every((r) => r.args[5] === serial))
    const before = requests.length
    for (const action of [
      { kind: 'tap', x: 2, y: 0 }, { kind: 'tap', x: -1, y: 0 }, { kind: 'tap', x: 0.5, y: 0 },
      { kind: 'swipe', x1: 0, y1: 0, x2: 1, y2: 1, durationMs: 20 },
      { kind: 'key', key: 'toString' }, { kind: 'launch', packageName: 'dev.example;id' },
      { kind: 'shell', command: 'id' },
    ]) await assert.rejects(() => adapter.act(serial, action as AndroidAction, { width: 2, height: 2 }), hasCode('invalid-input'))
    for (const text of ['café', '你好', 'line\nbreak', '%s', 'token\u0000']) await assert.rejects(() => adapter.act(serial, { kind: 'type', text }, { width: 2, height: 2 }), hasCode('unsupported-text'))
    assert.equal(requests.length, before, 'invalid actions never start adb')
    await assert.rejects(() => adapter.act('-unsafe', { kind: 'key', key: 'home' }, { width: 2, height: 2 }), hasCode('invalid-input'))
    await assert.rejects(() => adapter.act(serial, { kind: 'key', key: 'home' }, { width: 0, height: 2 }), hasCode('invalid-input'))
  }
  {
    const { adapter, requests } = fixture({ policy: unlocked.replace('showing=false', 'showing=true') })
    await assert.rejects(() => adapter.act(serial, { kind: 'key', key: 'home' }, { width: 2, height: 2 }), hasCode('locked'))
    assert.ok(!requests.some((r) => r.args.includes('input')), 'Home is never used to wake or bypass a locked phone')
  }
  {
    const { adapter } = fixture({ override: async () => ok('', 'error: device SYNTHETIC_PRIVATE_SERIAL not found at 10.0.0.99 secret 123456', 1) })
    await assert.rejects(() => adapter.devices(), (error: unknown) => error instanceof AndroidPhoneError && error.code === 'disconnected' && !/SYNTHETIC_PRIVATE|10\.0|123456/.test(error.message))
  }
  {
    const aborter = new AbortController(); aborter.abort()
    const { adapter, requests } = fixture()
    await assert.rejects(() => adapter.observe(serial, aborter.signal), hasCode('aborted'))
    assert.equal(requests.length, 0)
  }
  {
    const alias = path.join(root, 'alias-adb')
    await fs.symlink(adb, alias)
    const tool = new AndroidPhoneAdapter({ adbPath: alias, run: async () => ok('Android Debug Bridge version 1.0.41\n') })
    assert.equal((await tool.resolve())?.path, await fs.realpath(adb))
    const bad = new AndroidPhoneAdapter({ adbPath: path.join(root, 'missing-adb') })
    await assert.rejects(() => bad.resolve(), hasCode('invalid-adb'))
    const unknown = new AndroidPhoneAdapter({ adbPath: adb, run: async () => ok('untrusted executable') })
    await assert.rejects(() => unknown.resolve(), hasCode('invalid-adb'))
  }

  // Real child-process tests use Node fixtures, never adb or a connected phone.
  const command = (script: string, options: Partial<AndroidCommand> = {}): AndroidCommand => ({ executable: process.execPath, args: ['-e', script], timeoutMs: 3000, maxOutputBytes: 1024, ...options })
  const binary = await runAndroidCommand(command('process.stdout.write(Buffer.from([0,255,1]));process.stderr.write("ok")'))
  assert.deepEqual([...binary.stdout], [0, 255, 1])
  assert.equal(binary.stderr.toString(), 'ok')
  const secret = await runAndroidCommand(command('let n=0;process.stdin.on("data",d=>n+=d.length);process.stdin.on("end",()=>process.stdout.write(String(n)))', { stdin: '123456\n' }))
  assert.equal(secret.stdout.toString(), '7')
  await assert.rejects(() => runAndroidCommand(command('process.stdout.write("x".repeat(8192))')), hasCode('output-limit'))
  await assert.rejects(() => runAndroidCommand(command('setInterval(()=>{},1000)', { timeoutMs: 30 })), hasCode('timeout'))
  const aborter = new AbortController()
  const pidFile = path.join(root, 'synthetic-child.pid')
  const pending = runAndroidCommand(command('process.on("SIGTERM",()=>{});require("node:fs").writeFileSync(process.argv[1],String(process.pid));setInterval(()=>{},1000)', { signal: aborter.signal, args: ['-e', 'process.on("SIGTERM",()=>{});require("node:fs").writeFileSync(process.argv[1],String(process.pid));setInterval(()=>{},1000)', pidFile] }))
  const deadline = Date.now() + 2000
  let pid = 0
  while (!pid && Date.now() < deadline) {
    try { pid = Number(await fs.readFile(pidFile, 'utf8')) } catch { await new Promise((resolve) => setTimeout(resolve, 5)) }
  }
  assert.ok(pid, 'synthetic child is running and deliberately ignores SIGTERM before cancellation')
  aborter.abort()
  await assert.rejects(() => pending, hasCode('aborted'))
  assert.throws(() => process.kill(pid, 0), 'cancellation kills the child process, including a child that ignores SIGTERM')
  const aborted = new AbortController(); aborted.abort()
  await assert.rejects(() => runAndroidCommand(command('process.exit(0)', { signal: aborted.signal })), hasCode('aborted'))
  const oldSocket = process.env.ADB_SERVER_SOCKET, oldTrace = process.env.ADB_TRACE
  try {
    process.env.ADB_SERVER_SOCKET = 'tcp:untrusted.example:5037'; process.env.ADB_TRACE = 'all'
    const scrubbed = await runAndroidCommand(command('process.stdout.write(JSON.stringify([process.env.ADB_SERVER_SOCKET,process.env.ADB_TRACE]))'))
    assert.equal(scrubbed.stdout.toString(), '[null,null]')
  } finally {
    if (oldSocket === undefined) delete process.env.ADB_SERVER_SOCKET; else process.env.ADB_SERVER_SOCKET = oldSocket
    if (oldTrace === undefined) delete process.env.ADB_TRACE; else process.env.ADB_TRACE = oldTrace
  }
  console.log('Android adapter security, transport, observation and cancellation checks passed with synthetic phones only.')
} finally { await fs.rm(root, { recursive: true, force: true }) }
