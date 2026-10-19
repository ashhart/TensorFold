// Run: node src/lib/boot-guard.test.ts
import assert from 'node:assert/strict'
import { BOOT_GUARD_TIMEOUT_MS, bootFailureText, installBootGuard } from './boot-guard'

function fakeDom(ready = false) {
  const elements: Record<string, { textContent: string | null; attrs: Record<string, string>; setAttribute(name: string, value: string): void }> = {
    'boot-failure': { textContent: null, attrs: {}, setAttribute(name, value) { this.attrs[name] = value } },
    'boot-failure-detail': { textContent: null, attrs: {}, setAttribute(name, value) { this.attrs[name] = value } },
  }
  return {
    elements,
    body: { classList: { contains: (name: string) => ready && name === 'ready' } },
    getElementById: (id: string) => elements[id] ?? null,
  }
}

function fakeWindow() {
  const handlers = new Map<string, (event: any) => void>()
  let timer: (() => void) | null = null
  return {
    handlers,
    fire: (type: string, event: any) => handlers.get(type)?.(event),
    runTimer: () => timer?.(),
    addEventListener: (type: string, handler: (event: any) => void) => { handlers.set(type, handler) },
    setTimeout: (handler: () => void, ms: number) => { assert.equal(ms, BOOT_GUARD_TIMEOUT_MS); timer = handler; return 1 },
  }
}

assert.match(bootFailureText(new Error('boom')), /boom/)
assert.equal(bootFailureText('plain text'), 'plain text')
assert.equal(bootFailureText(undefined), 'No error detail was provided.')
assert.match(bootFailureText({ stack: 'stack line' }), /stack line/)

{
  const dom = fakeDom()
  const win = fakeWindow()
  installBootGuard(win, dom)
  win.fire('error', { error: new TypeError('renderer exploded') })
  assert.equal(dom.elements['boot-failure'].attrs['data-visible'], 'true', 'a startup exception is surfaced')
  assert.match(String(dom.elements['boot-failure-detail'].textContent), /renderer exploded/)
}

{
  const dom = fakeDom()
  const win = fakeWindow()
  installBootGuard(win, dom)
  win.fire('unhandledrejection', { reason: new Error('promise failed') })
  assert.match(String(dom.elements['boot-failure-detail'].textContent), /promise failed/)
}

{
  const dom = fakeDom()
  const win = fakeWindow()
  installBootGuard(win, dom)
  win.runTimer()
  assert.equal(dom.elements['boot-failure'].attrs['data-visible'], 'true', 'a stalled boot is surfaced after the timeout')
  assert.match(String(dom.elements['boot-failure-detail'].textContent), /10 seconds/)
}

{
  const dom = fakeDom(true)
  const win = fakeWindow()
  installBootGuard(win, dom)
  win.runTimer()
  win.fire('error', { error: new Error('late failure') })
  assert.equal(dom.elements['boot-failure'].attrs['data-visible'], undefined, 'a ready app never shows the boot panel')
}

console.log('boot guard tests passed')
