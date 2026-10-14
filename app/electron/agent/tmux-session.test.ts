import assert from 'node:assert/strict'
import {
  startArgs, sendTextArgs, sendEnterArgs, captureArgs, hasArgs, killArgs,
  startSession, sendLine, capturePane, hasSession, killSession, type TmuxRunner,
} from './tmux-session'

// --- pure arg builders ---
assert.deepEqual(startArgs('vontra_s1', ['claude'], '/repo'), ['new-session', '-d', '-s', 'vontra_s1', '-c', '/repo', 'claude'])
assert.deepEqual(startArgs('s', ['codex', 'exec']), ['new-session', '-d', '-s', 's', 'codex', 'exec'])
assert.deepEqual(sendTextArgs('s', 'hello world'), ['send-keys', '-t', 's', '-l', 'hello world'])
assert.deepEqual(sendEnterArgs('s'), ['send-keys', '-t', 's', 'Enter'])
assert.deepEqual(captureArgs('s'), ['capture-pane', '-t', 's', '-p', '-S', '-'])
assert.deepEqual(hasArgs('s'), ['has-session', '-t', 's'])
assert.deepEqual(killArgs('s'), ['kill-session', '-t', 's'])

async function run() {
  const calls: string[][] = []
  const runner: TmuxRunner = {
    run: async (args) => {
      calls.push(args)
      if (args[0] === 'has-session') return { code: args.includes('alive') ? 0 : 1, stdout: '', stderr: '' }
      if (args[0] === 'capture-pane') return { code: 0, stdout: 'pane contents\n', stderr: '' }
      return { code: 0, stdout: '', stderr: '' }
    },
  }

  // hasSession reflects tmux exit code
  assert.equal(await hasSession(runner, 'alive'), true)
  assert.equal(await hasSession(runner, 'dead'), false)

  // startSession issues new-session with cwd + command
  calls.length = 0
  await startSession(runner, { name: 's', command: ['claude'], cwd: '/repo' })
  assert.deepEqual(calls[0], ['new-session', '-d', '-s', 's', '-c', '/repo', 'claude'])

  // sendLine sends the literal text, THEN Enter (two separate calls)
  calls.length = 0
  await sendLine(runner, 's', 'do the thing')
  assert.deepEqual(calls, [['send-keys', '-t', 's', '-l', 'do the thing'], ['send-keys', '-t', 's', 'Enter']])

  // capturePane returns the pane stdout
  assert.equal(await capturePane(runner, 's'), 'pane contents\n')

  // killSession issues kill-session
  calls.length = 0
  await killSession(runner, 's')
  assert.deepEqual(calls[0], ['kill-session', '-t', 's'])

  console.log('tmux-session tests passed')
}

run().catch((e) => { console.error(e); process.exit(1) })
