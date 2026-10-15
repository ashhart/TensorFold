#!/usr/bin/env node
// Test double for tensorfold-computer: the same newline-delimited JSON protocol with canned replies, and
// nothing that touches the desktop. Knobs: FAKE_HELPER_RECORD (append spawn and request records),
// FAKE_HELPER_DELAY_MS (delay every reply), FAKE_HELPER_FAIL_CMD with FAKE_HELPER_FAIL_CODE (answer ok:false),
// FAKE_HELPER_CRASH_CMD (exit 3 without answering), FAKE_HELPER_NOISE (print a non-JSON line first).
// A request's own delayMs overrides the delay, which lets one test reply out of order.
import { appendFileSync } from 'node:fs'
import readline from 'node:readline'

const env = process.env
const args = process.argv.slice(2)

function record(entry) {
  if (env.FAKE_HELPER_RECORD) appendFileSync(env.FAKE_HELPER_RECORD, `${JSON.stringify(entry)}\n`)
}

function reply(message, then) {
  process.stdout.write(`${JSON.stringify(message)}\n`, then)
}

const app = { name: 'Notes', bundleId: 'com.apple.Notes', pid: 4242 }
const window = { title: 'Groceries', frame: { x: 0, y: 25, width: 800, height: 600 } }

function answer(request) {
  switch (request.cmd) {
    case 'permissions': return { accessibility: true, screenRecording: false }
    case 'request_permissions': return { accessibility: true, screenRecording: request.screenRecording === true }
    case 'apps': return { apps: [{ ...app, active: true }] }
    case 'frontmost': return { app, window }
    case 'observe': return {
      app, window, screen: { x: 0, y: 0, width: 1512, height: 982, scale: 2 }, truncated: false,
      elements: [{ id: 1, role: 'AXButton', title: 'New Note', frame: { x: 10, y: 40, width: 30, height: 20 }, enabled: true, focused: false, secure: false }],
    }
    case 'click': return { app, window, via: request.element ? 'press' : 'mouse' }
    case 'move': case 'drag': case 'scroll': case 'key': case 'open_app': return { app, window }
    case 'type': return { app, window, typed: [...String(request.text ?? '')].length }
    case 'echo': return { request }
    case 'env': return { env: { ...process.env } }
    case 'pid': return { pid: process.pid }
    case 'stop': return {}
    default: return null
  }
}

record({ kind: 'spawn', pid: process.pid, args })
if (env.FAKE_HELPER_NOISE) process.stdout.write('hello from a noisy helper\n')
process.stderr.write('fake helper started\n')

const lines = readline.createInterface({ input: process.stdin })
lines.on('line', (line) => {
  let request
  try {
    request = JSON.parse(line)
  } catch {
    reply({ id: null, ok: false, code: 'bad_json', error: 'Request is not valid JSON.' })
    return
  }
  record({ kind: 'request', pid: process.pid, request })
  if (env.FAKE_HELPER_CRASH_CMD === request.cmd) process.exit(3)
  const delay = Number(request.delayMs ?? env.FAKE_HELPER_DELAY_MS ?? 0)
  setTimeout(() => {
    if (env.FAKE_HELPER_FAIL_CMD === request.cmd) {
      reply({ id: request.id, ok: false, code: env.FAKE_HELPER_FAIL_CODE || 'bad_request', error: `fake ${request.cmd} refused` })
      return
    }
    const body = answer(request)
    if (!body) {
      reply({ id: request.id, ok: false, code: 'unknown_cmd', error: `Unknown command ${request.cmd}.` })
      return
    }
    reply({ ...body, id: request.id, ok: true }, request.cmd === 'stop' ? () => process.exit(0) : undefined)
  }, delay)
})
lines.on('close', () => process.exit(0))
