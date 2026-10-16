import assert from 'node:assert/strict'
import { describeComputerAction, formatObservation, normalizeComputerUseInput } from './actions'
import { createComputerUseRunner, type ComputerControl } from './runner'

const safari = { name: 'Safari', bundleId: 'com.apple.Safari', pid: 42 }
const notes = { name: 'Notes', bundleId: 'com.apple.Notes', pid: 7 }

function fakeControl(options: { accessibility?: boolean; front?: typeof safari; failWith?: string } = {}) {
  const calls: Array<[string, unknown]> = []
  let front = options.front ?? safari
  const fail = (name: string) => {
    if (options.failWith) throw Object.assign(new Error(`helper refused ${name}`), { code: options.failWith })
  }
  const control: ComputerControl = {
    permissions: async () => ({ accessibility: options.accessibility ?? true, screenRecording: false }),
    requestPermissions: async (input) => { calls.push(['requestPermissions', input]); return { accessibility: false, screenRecording: false } },
    apps: async () => ({ apps: [{ ...safari, active: front === safari }, { ...notes, active: front === notes }] }),
    frontmost: async () => ({ app: front, window: { title: 'Start Page', frame: { x: 0, y: 0, width: 800, height: 600 } } }),
    observe: async (input) => {
      calls.push(['observe', input])
      return {
        app: safari, window: { title: 'Start Page', frame: { x: 0, y: 0, width: 800, height: 600 } }, screen: { width: 1512, height: 982, scale: 2 },
        elements: [
          { id: 1, role: 'AXTextField', title: 'Address', value: 'github.com', frame: { x: 100, y: 20, width: 400, height: 30 }, enabled: true, focused: true, secure: false },
          { id: 2, role: 'AXTextField', subrole: 'AXSecureTextField', title: 'Password', value: 'hunter2', frame: { x: 100, y: 80, width: 200, height: 30 }, enabled: true, focused: false, secure: true },
          { id: 3, role: 'AXButton', title: 'Back', frame: { x: 10, y: 20, width: 30, height: 30 }, enabled: false, focused: false, secure: false },
        ],
      }
    },
    click: async (input) => { fail('click'); calls.push(['click', input]) },
    drag: async (input) => { calls.push(['drag', input]) },
    scroll: async (input) => { calls.push(['scroll', input]) },
    type: async (input) => { fail('type'); calls.push(['type', input]) },
    key: async (input) => { calls.push(['key', input]) },
    openApp: async (input) => { calls.push(['openApp', input]); front = input.name === 'Notes' ? notes : front },
  }
  return { control, calls }
}

const run = (control: ComputerControl, args: unknown, indicator?: { shown: string[]; hidden: number }) => createComputerUseRunner({
  control: () => control, settleMs: 0, idleHideMs: 0,
  indicator: indicator && { show: (label) => indicator.shown.push(label), hide: () => { indicator.hidden++ } },
}).run(normalizeComputerUseInput(args), { threadId: 't1' })

async function main() {
  // Input rules: every refusal says how to fix the call.
  assert.throws(() => normalizeComputerUseInput({ action: 'launch', app: 'Safari' }), /action must be one of/)
  assert.throws(() => normalizeComputerUseInput({ action: 'click', element: 1 }), /needs app/)
  assert.throws(() => normalizeComputerUseInput({ action: 'observe', app: 'TensorFold' }), /cannot operate TensorFold/)
  assert.throws(() => normalizeComputerUseInput({ action: 'click', app: 'Safari' }), /element .* or x and y/)
  assert.throws(() => normalizeComputerUseInput({ action: 'scroll', app: 'Safari' }), /dx or dy/)
  assert.throws(() => normalizeComputerUseInput({ action: 'type', app: 'Safari', text: '' }), /needs text/)
  assert.throws(() => normalizeComputerUseInput({ action: 'drag', app: 'Safari', x: 1, y: 2 }), /drag to needs/)
  assert.deepEqual(normalizeComputerUseInput({ action: 'scroll', app: 'Safari', dy: -500 }), { action: 'scroll', app: 'Safari', dx: 0, dy: -50 })
  assert.equal(normalizeComputerUseInput({ action: 'wait', app: 'Safari', seconds: 99 }).seconds, 10)
  assert.equal(describeComputerAction(normalizeComputerUseInput({ action: 'type', app: 'Notes', text: 'hello' })), 'Type 5 characters in Notes')

  // Observations are compact and never show a secure field's value.
  {
    const { control } = fakeControl()
    const text = formatObservation(await control.observe({}))
    assert.match(text, /\[1\] AXTextField "Address" value="github.com" focused at 300,35/)
    assert.match(text, /\[2\] AXTextField\/AXSecureTextField "Password" secure at 200,95/)
    assert.doesNotMatch(text, /hunter2/)
    assert.match(text, /\[3\] AXButton "Back" disabled/)
  }

  // Observe finds the named app by pid; the indicator shows while acting and hides after.
  {
    const { control, calls } = fakeControl()
    const indicator = { shown: [] as string[], hidden: 0 }
    const result = await run(control, { action: 'observe', app: 'safari' }, indicator)
    assert.equal(result.ok, true)
    assert.deepEqual(calls[0], ['observe', { pid: 42, maxElements: 150 }])
    assert.deepEqual(indicator.shown, ['safari'])
    await new Promise((resolve) => setTimeout(resolve, 5))
    assert.equal(indicator.hidden, 1)
    assert.equal((await run(control, { action: 'observe', app: 'Mail' })).ok, false, 'an app that is not running')
  }

  // Input actions only reach the app in front; open_app brings it there first.
  {
    const { control, calls } = fakeControl()
    const refused = await run(control, { action: 'type', app: 'Notes', text: 'hi' })
    assert.equal(refused.ok, false)
    assert.match(refused.output, /Notes is not the app in front \(Safari is\)\. Call open_app/)
    assert.equal(calls.some(([name]) => name === 'type'), false)
    assert.equal((await run(control, { action: 'open_app', app: 'Notes' })).ok, true)
    const typed = await run(control, { action: 'type', app: 'Notes', text: 'hi' })
    assert.equal(typed.ok, true)
    assert.deepEqual(calls.at(-1), ['type', { text: 'hi' }])
    await run(control, { action: 'double_click', app: 'Notes', element: 3 })
    assert.deepEqual(calls.at(-1), ['click', { element: 3, button: 'left', count: 2 }])
  }

  // Missing Accessibility asks macOS once and tells the model what the user must do.
  {
    const { control, calls } = fakeControl({ accessibility: false })
    const result = await run(control, { action: 'click', app: 'Safari', x: 10, y: 10 })
    assert.equal(result.ok, false)
    assert.match(result.output, /Accessibility/)
    assert.deepEqual(calls, [['requestPermissions', { accessibility: true }]])
  }

  // Helper refusal codes become instructions.
  {
    assert.match((await run(fakeControl({ failWith: 'secure_field' }).control, { action: 'type', app: 'Safari', text: 'pw' })).output, /never types into password fields/)
    assert.match((await run(fakeControl({ failWith: 'own_app' }).control, { action: 'click', app: 'Safari', x: 1, y: 1 })).output, /its own window/)
    assert.match((await run(fakeControl({ failWith: 'weird' }).control, { action: 'click', app: 'Safari', x: 1, y: 1 })).output, /Computer use failed: helper refused click/)
  }

  // Stop on the indicator ends the run through the thread's cancel.
  {
    const { control } = fakeControl()
    let stop: (() => void) | null = null
    let cancelled = 0
    const runner = createComputerUseRunner({ control: () => control, settleMs: 0, idleHideMs: 0, indicator: { show: (_label, onStop) => { stop = onStop }, hide: () => {} } })
    await runner.run(normalizeComputerUseInput({ action: 'key', app: 'Safari', keys: 'cmd+l' }), { threadId: 't1', cancel: () => { cancelled++ } })
    stop!()
    assert.equal(cancelled, 1)
  }

  console.log('computer use: input rules, observations, front-app guard, permissions, refusals and stop')
}

main().catch((error) => { console.error(error); process.exit(1) })
