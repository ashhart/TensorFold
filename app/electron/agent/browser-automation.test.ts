// Run: npx tsx electron/agent/browser-automation.test.ts
import assert from 'node:assert/strict'
import {
  BROWSER_AUTOMATION_ACTIONS,
  assertBrowserAutomationDestination,
  assertBrowserAutomationRequestDestination,
  formatBrowserAutomationResult,
  normalizeBrowserAutomationInput,
} from './browser-automation'

assert.deepEqual(BROWSER_AUTOMATION_ACTIONS, [
  'navigate',
  'click',
  'type',
  'fill_form',
  'tab_new',
  'tab_switch',
  'tab_close',
  'screenshot',
  'console_logs',
  'network_errors',
  'accessibility_snapshot',
])

const nav = normalizeBrowserAutomationInput({
  action: 'navigate',
  url: 'localhost:5173/dashboard',
  sessionId: ' QA session ',
  includeSnapshot: true,
  captureScreenshot: true,
})
assert.equal(nav.action, 'navigate')
assert.equal(nav.url, 'http://localhost:5173/dashboard')
assert.equal(nav.sessionId, 'QA session')
assert.equal(nav.includeSnapshot, true)
assert.equal(nav.captureScreenshot, true)

assert.throws(
  () => normalizeBrowserAutomationInput({ action: 'navigate', url: 'file:///etc/passwd' }),
  /http or https/,
)
assert.throws(
  () => normalizeBrowserAutomationInput({ action: 'navigate', url: 'http://192.168.1.20/admin' }),
  /private, link-local, or metadata/,
)
assert.throws(
  () => normalizeBrowserAutomationInput({ action: 'navigate', url: 'https://user:secret@example.com/' }),
  /must not include credentials/,
)
assert.throws(
  () => normalizeBrowserAutomationInput({ action: 'click' }),
  /selector or ref/,
)

const click = normalizeBrowserAutomationInput({ action: 'click', selector: 'button.save', note: 'confirm overlay is gone' })
assert.equal(click.selector, 'button.save')
assert.equal(click.note, 'confirm overlay is gone')

const fill = normalizeBrowserAutomationInput({
  action: 'fill_form',
  fields: [
    { selector: 'input[name=email]', value: 'maya@example.test' },
    { ref: 'e12', value: 'Launch window' },
  ],
})
assert.equal(fill.fields?.length, 2)
assert.equal(fill.fields?.[1].ref, 'e12')

// `type` must preserve whitespace/newlines verbatim (regression: cleanOptional collapsed them).
const multiline = 'line one\n  indented\ttab\n\nline two   trailing  '
const typed = normalizeBrowserAutomationInput({ action: 'type', selector: 'textarea#body', text: multiline })
assert.equal(typed.text, multiline)

// `type` and `fill_form` must agree on verbatim value handling.
const filledWs = normalizeBrowserAutomationInput({
  action: 'fill_form',
  fields: [{ selector: 'textarea', value: multiline }],
})
assert.equal(filledWs.fields?.[0].value, multiline)
assert.equal(typed.text, filledWs.fields?.[0].value)

// still reject empty / whitespace-only typed text
assert.throws(
  () => normalizeBrowserAutomationInput({ action: 'type', selector: 'x', text: '' }),
  /requires text/,
)
assert.throws(
  () => normalizeBrowserAutomationInput({ action: 'type', selector: 'x', text: '   ' }),
  /requires text/,
)

const output = formatBrowserAutomationResult({
  ok: true,
  action: 'accessibility_snapshot',
  sessionId: 'browser-session',
  tabId: 'tab-1',
  url: 'http://localhost:5173/',
  title: 'TensorFold Studio',
  loadState: 'loaded',
  accessibilitySnapshot: '- button "Save" [ref=e3]',
  screenshot: { path: '/tmp/vontra-browser.png', title: 'Home page' },
  consoleMessages: [{ level: 'error', text: 'Hydration mismatch', at: 1 }],
  networkErrors: [{ url: 'http://localhost:5173/api', method: 'GET', status: 500, errorText: 'Internal Server Error', at: 2 }],
})
assert.match(output, /Action: accessibility_snapshot/)
assert.match(output, /Session: browser-session/)
assert.match(output, /Screenshot: \/tmp\/vontra-browser\.png/)
assert.match(output, /Accessibility snapshot:/)
assert.match(output, /button "Save"/)
assert.match(output, /Console messages:/)
assert.match(output, /Hydration mismatch/)
assert.match(output, /Network errors:/)
assert.match(output, /500 Internal Server Error/)

// Localhost is an explicit product path for UI smoke tests and never consults DNS.
let localhostResolutions = 0
await assertBrowserAutomationDestination('http://localhost:5173/', async () => {
  localhostResolutions++
  return ['127.0.0.1']
}, { allowedLoopbackOrigin: 'http://localhost:5173' })
assert.equal(localhostResolutions, 0)
await assert.rejects(
  () => assertBrowserAutomationDestination('http://localhost:5173/', async () => ['127.0.0.1']),
  /not explicitly opened for that exact localhost origin/,
)

// The localhost exception belongs only to the initially approved URL.origin. A local UI
// can use normal same-origin paths, but cannot pivot to another service, hostname alias, or
// protocol with the tab-wide boolean permission that used to exist.
const localhostOrigin = { allowedLoopbackOrigin: 'http://localhost:5173' }
await assertBrowserAutomationRequestDestination('http://localhost:5173/api/status', async () => {
  localhostResolutions++
  return ['127.0.0.1']
}, localhostOrigin)
assert.equal(localhostResolutions, 0, 'same-origin loopback requests remain literal and DNS-free')
await assert.rejects(
  () => assertBrowserAutomationRequestDestination('http://localhost:3000/admin', async () => ['127.0.0.1'], localhostOrigin),
  /blocked loopback origin http:\/\/localhost:3000/,
)
await assert.rejects(
  () => assertBrowserAutomationRequestDestination('http://127.0.0.1:5173/admin', async () => ['127.0.0.1'], localhostOrigin),
  /blocked loopback origin http:\/\/127\.0\.0\.1:5173/,
)
await assert.rejects(
  () => assertBrowserAutomationRequestDestination('https://localhost:5173/admin', async () => ['127.0.0.1'], localhostOrigin),
  /blocked loopback origin https:\/\/localhost:5173/,
)

await assertBrowserAutomationDestination('https://example.test/', async () => ['93.184.216.34', '2606:4700:4700::1111'])
await assert.rejects(
  () => assertBrowserAutomationDestination('https://rebind.example/', async () => ['93.184.216.34', '169.254.169.254']),
  /resolves to a private or non-public address/,
)
await assert.rejects(
  () => assertBrowserAutomationDestination('https://internal.example/', async () => ['10.0.0.8']),
  /resolves to a private or non-public address/,
)
await assert.rejects(
  () => assertBrowserAutomationDestination('https://reserved.example/', async () => ['203.0.113.20']),
  /resolves to a private or non-public address/,
)
await assert.rejects(
  () => assertBrowserAutomationDestination('https://unresolved.example/', async () => { throw new Error('NXDOMAIN') }),
  /could not safely resolve/,
)

// The same request guard is applied to every Chromium request, so a safe initial URL cannot
// redirect to either a literal internal address or a public name with an internal DNS answer.
await assertBrowserAutomationRequestDestination('https://public.example/start', async () => ['93.184.216.34'])
await assert.rejects(
  () => assertBrowserAutomationRequestDestination('http://169.254.169.254/latest/meta-data', async () => ['169.254.169.254']),
  /private, link-local, or metadata/,
)
await assert.rejects(
  () => assertBrowserAutomationRequestDestination('https://redirect.example/internal', async () => ['127.0.0.1']),
  /resolves to a private or non-public address/,
)
await assert.rejects(
  () => assertBrowserAutomationRequestDestination('http://127.0.0.1:3000/admin', async () => ['127.0.0.1']),
  /not explicitly opened for that exact localhost origin/,
)
await assert.rejects(
  () => assertBrowserAutomationRequestDestination('file:///etc/passwd', async () => []),
  /must use http\(s\) or ws\(s\)/,
)

console.log('browser automation tests passed')
