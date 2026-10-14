import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import {
  COMPUTER_HELPER_TIMEOUT_MS,
  ComputerHelper,
  ComputerHelperError,
  computerHelperEnv,
  computerHelperPath,
  typeTimeoutMs,
} from './helper'
import type { ComputerControl } from './runner'

const FAKE_HELPER = path.join(__dirname, 'test-fixtures', 'fake-helper.mjs')
const OWN = 'dev.tensorfold.studio'

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`
}

// The client passes only PATH, HOME and TMPDIR, so knobs reach the fake through a wrapper script.
async function wrapper(dir: string, name: string, knobs: Record<string, string> = {}): Promise<string> {
  const file = path.join(dir, name)
  const exports = Object.entries(knobs).map(([key, value]) => `export ${key}=${shellQuote(value)}\n`).join('')
  await fs.writeFile(file, `#!/bin/sh\n${exports}exec ${shellQuote(process.execPath)} ${shellQuote(FAKE_HELPER)} "$@"\n`, { mode: 0o755 })
  return file
}

async function records(file: string): Promise<any[]> {
  try {
    return (await fs.readFile(file, 'utf8')).split('\n').filter(Boolean).map((line) => JSON.parse(line))
  } catch { return [] }
}

function isCode(code: string) {
  return (error: unknown) => error instanceof ComputerHelperError && error.code === code
}

function alive(pid: number): boolean {
  try { process.kill(pid, 0); return true } catch { return false }
}

async function waitFor(check: () => boolean | Promise<boolean>, timeoutMs = 5_000): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (!(await check())) {
    if (Date.now() > deadline) throw new Error('timed out waiting')
    await new Promise((resolve) => setTimeout(resolve, 20))
  }
}

async function main() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tf-computer-helper-'))
  const helpers: ComputerHelper[] = []
  const make = (binaryPath: string, extra: Partial<ConstructorParameters<typeof ComputerHelper>[0]> = {}) => {
    const helper = new ComputerHelper({ binaryPath, ownBundleId: OWN, env: { PATH: process.env.PATH, HOME: root, TMPDIR: root }, ...extra })
    helpers.push(helper)
    return helper
  }
  try {
    const recordFile = path.join(root, 'records.jsonl')
    const logs: string[] = []
    const basic = make(await wrapper(root, 'basic', { FAKE_HELPER_RECORD: recordFile, FAKE_HELPER_NOISE: '1' }), { onLog: (line) => logs.push(line) })

    // Typed calls, the --own-bundle-id argument, and replies without id or ok.
    assert.equal(basic.running, false, 'nothing starts before the first call')
    assert.deepEqual(await basic.permissions(), { accessibility: true, screenRecording: false })
    assert.deepEqual((await records(recordFile)).find((r) => r.kind === 'spawn').args, ['--own-bundle-id', OWN])
    assert.equal((await basic.frontmost()).app?.bundleId, 'com.apple.Notes')
    assert.equal((await basic.apps()).apps[0].active, true)
    const observation = await basic.observe({ maxElements: 20 })
    assert.equal(observation.elements[0].role, 'AXButton')
    assert.equal(observation.screen.scale, 2)
    assert.equal((await basic.click({ element: 1 })).via, 'press')
    assert.equal((await basic.click({ x: 5, y: 6, button: 'right', count: 2 })).via, 'mouse')
    assert.equal((await basic.type({ text: 'héllo' })).typed, 5)
    assert.equal((await basic.requestPermissions({ screenRecording: true })).screenRecording, true)
    await basic.move({ x: 1, y: 2 })
    await basic.drag({ from: { x: 1, y: 2 }, to: { x: 3, y: 4 } })
    await basic.scroll({ element: 1, dy: 3 })
    await basic.key({ keys: 'cmd+n' })
    await basic.openApp({ bundleId: 'com.apple.Notes' })
    const sent = (await records(recordFile)).filter((r) => r.kind === 'request').map((r) => r.request)
    const { id: clickId, ...clickSent } = sent.find((r) => r.cmd === 'click' && r.x === 5)
    assert.equal(typeof clickId, 'number')
    assert.deepEqual(clickSent, { x: 5, y: 6, button: 'right', count: 2, cmd: 'click' })
    assert.deepEqual(sent.find((r) => r.cmd === 'drag').from, { x: 1, y: 2 })
    assert.deepEqual(sent.find((r) => r.cmd === 'scroll'), { element: 1, dy: 3, id: sent.find((r) => r.cmd === 'scroll').id, cmd: 'scroll' })
    const control: ComputerControl = basic
    assert.ok(control, 'the client satisfies the runner interface')
    assert.equal(sent.find((r) => r.cmd === 'open_app').bundleId, 'com.apple.Notes')
    assert.ok(new Set(sent.map((r) => r.id)).size === sent.length, 'every request has its own id')
    assert.ok(logs.some((line) => line.includes('non-JSON line')), 'stray stdout is logged, not fatal')
    assert.ok(logs.includes('fake helper started'), 'stderr lines reach onLog')

    // Out-of-order replies land on the right callers; params cannot override id or cmd.
    const order: string[] = []
    const slow = basic.request<{ request: any }>('echo', { delayMs: 200, tag: 'slow' }).then((r) => { order.push('slow'); return r })
    const fast = basic.request<{ request: any }>('echo', { delayMs: 0, tag: 'fast' }).then((r) => { order.push('fast'); return r })
    const [slowReply, fastReply] = await Promise.all([slow, fast])
    assert.deepEqual(order, ['fast', 'slow'])
    assert.equal(slowReply.request.tag, 'slow')
    assert.equal(fastReply.request.tag, 'fast')
    const sneaky = await basic.request<{ request: any }>('echo', { id: 999_999, cmd: 'stop' })
    assert.equal(sneaky.request.cmd, 'echo')
    assert.notEqual(sneaky.request.id, 999_999)
    assert.equal(basic.running, true)

    // ok:false becomes a ComputerHelperError carrying the helper's code.
    const refusing = make(await wrapper(root, 'refusing', { FAKE_HELPER_FAIL_CMD: 'click', FAKE_HELPER_FAIL_CODE: 'own_app' }))
    await assert.rejects(refusing.click({ x: 1, y: 1 }), (error: unknown) => isCode('own_app')(error) && (error as Error).message === 'fake click refused')
    await assert.rejects(refusing.request('dance'), isCode('unknown_cmd'))
    assert.ok((await refusing.permissions()).accessibility, 'a refusal leaves the helper usable')

    // A timeout rejects that call, ends the stuck helper, and the next call starts a new one.
    const slowRecord = path.join(root, 'slow.jsonl')
    const slowHelper = make(await wrapper(root, 'slow', { FAKE_HELPER_RECORD: slowRecord }), { timeoutMs: 500 })
    const firstPid = (await slowHelper.request<{ pid: number }>('pid')).pid
    const started = Date.now()
    await assert.rejects(slowHelper.request('echo', { delayMs: 5_000 }), isCode('timeout'))
    assert.ok(Date.now() - started < 2_000, 'the timeout fires on time')
    await waitFor(() => !alive(firstPid))
    const secondPid = (await slowHelper.request<{ pid: number }>('pid')).pid
    assert.notEqual(secondPid, firstPid)
    assert.equal((await records(slowRecord)).filter((r) => r.kind === 'spawn').length, 2)

    // type gets a timeout that grows with the text; other calls keep the base timeout.
    assert.equal(typeTimeoutMs('abcd', 100), 120)
    assert.equal(typeTimeoutMs(''), COMPUTER_HELPER_TIMEOUT_MS)
    const delayed = make(await wrapper(root, 'delayed', { FAKE_HELPER_DELAY_MS: '400' }), { timeoutMs: 250 })
    assert.equal((await delayed.type({ text: 'x'.repeat(200) })).typed, 200)
    await assert.rejects(delayed.key({ keys: 'a' }), isCode('timeout'))

    // A crash rejects everything in flight; the next call restarts the helper.
    const crashRecord = path.join(root, 'crash.jsonl')
    const crashy = make(await wrapper(root, 'crashy', { FAKE_HELPER_CRASH_CMD: 'observe', FAKE_HELPER_RECORD: crashRecord }))
    const crashPid = (await crashy.request<{ pid: number }>('pid')).pid
    const inFlight = crashy.request('echo', { delayMs: 1_000 })
    const crashing = crashy.observe()
    await assert.rejects(inFlight, isCode('exited'))
    await assert.rejects(crashing, (error: unknown) => isCode('exited')(error) && /code 3/.test((error as Error).message))
    assert.equal(crashy.running, false)
    const restartedPid = (await crashy.request<{ pid: number }>('pid')).pid
    assert.notEqual(restartedPid, crashPid)
    assert.equal((await records(crashRecord)).filter((r) => r.kind === 'spawn').length, 2)

    // Only PATH, HOME and TMPDIR reach the child.
    const leaky = make(await wrapper(root, 'env'), {
      env: { PATH: process.env.PATH, HOME: root, TMPDIR: path.join(root, 'tmp'), OPENAI_API_KEY: 'sk-test-not-real', ELECTRON_RUN_AS_NODE: '1', NODE_OPTIONS: '--inspect' },
    })
    const childEnv = (await leaky.request<{ env: Record<string, string> }>('env')).env
    assert.equal(childEnv.PATH, process.env.PATH)
    assert.equal(childEnv.HOME, root)
    assert.equal(childEnv.TMPDIR, path.join(root, 'tmp'))
    for (const key of ['OPENAI_API_KEY', 'ELECTRON_RUN_AS_NODE', 'NODE_OPTIONS']) assert.equal(childEnv[key], undefined, `${key} is stripped`)
    // The wrapper's /bin/sh adds PWD, SHLVL and _, and macOS adds __CF_USER_TEXT_ENCODING.
    const shellAdded = new Set(['PATH', 'HOME', 'TMPDIR', 'PWD', 'OLDPWD', 'SHLVL', '_', '__CF_USER_TEXT_ENCODING'])
    assert.deepEqual(Object.keys(childEnv).filter((key) => !shellAdded.has(key)), [], 'nothing else is passed through')
    assert.deepEqual(computerHelperEnv({ PATH: '/bin', HOME: '/h', TMPDIR: '', SECRET: 'x' }), { PATH: '/bin', HOME: '/h' })

    // stop asks the helper to exit, waits for it, and a later call starts again.
    const stopRecord = path.join(root, 'stop.jsonl')
    const stoppable = make(await wrapper(root, 'stoppable', { FAKE_HELPER_RECORD: stopRecord }))
    const stopPid = (await stoppable.request<{ pid: number }>('pid')).pid
    await stoppable.stop()
    assert.equal(stoppable.running, false)
    assert.equal(alive(stopPid), false)
    assert.ok((await records(stopRecord)).some((r) => r.kind === 'request' && r.request.cmd === 'stop'))
    await stoppable.stop()
    assert.notEqual((await stoppable.request<{ pid: number }>('pid')).pid, stopPid)

    // dispose ends the helper at once and fails what was waiting.
    const disposable = make(await wrapper(root, 'disposable'))
    const disposedPid = (await disposable.request<{ pid: number }>('pid')).pid
    const waiting = disposable.request('echo', { delayMs: 2_000 })
    disposable.dispose()
    await assert.rejects(waiting, isCode('exited'))
    await waitFor(() => !alive(disposedPid))

    // A missing binary is a clear error, not a hang.
    await assert.rejects(make(path.join(root, 'nope')).permissions(), isCode('missing_helper'))

    // Packaged path when it exists, else the dev build.
    const resources = path.join(root, 'Resources')
    assert.equal(computerHelperPath(resources, '/dev/root'), path.join('/dev/root', 'build', 'computer-use', 'tensorfold-computer'))
    await fs.mkdir(path.join(resources, 'computer-use'), { recursive: true })
    await fs.writeFile(path.join(resources, 'computer-use', 'tensorfold-computer'), '')
    assert.equal(computerHelperPath(resources, '/dev/root'), path.join(resources, 'computer-use', 'tensorfold-computer'))
  } finally {
    await Promise.all(helpers.map((helper) => helper.stop().catch(() => {})))
    await fs.rm(root, { recursive: true, force: true })
  }
  console.log('computer helper client tests passed')
}

main().catch((error) => { console.error(error); process.exit(1) })
