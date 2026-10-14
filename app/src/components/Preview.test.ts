import assert from 'node:assert/strict'
import { browserClickDiagnostics, formatBrowserEvidence, localPreviewCandidates, normalizeLocalPreviewUrl } from '../lib/browser-preview'

const thread: any = {
  cwd: '/Users/test/Projects/demo-app',
  timeline: [
    { id: 't1', kind: 'tool', output: 'VITE ready: http://localhost:5173/\nNetwork: http://192.168.1.22:5173/' },
    { id: 't2', kind: 'assistant', text: 'Try localhost:3000/dashboard).' },
  ],
}

assert.equal(normalizeLocalPreviewUrl('localhost:5173'), 'http://localhost:5173/')
assert.equal(normalizeLocalPreviewUrl('https://127.0.0.1:4173/path'), 'https://127.0.0.1:4173/path')
assert.equal(normalizeLocalPreviewUrl('http://127.0.0.1:8443/path'), null)
assert.equal(normalizeLocalPreviewUrl('http://192.168.1.22:5173'), null)
assert.equal(normalizeLocalPreviewUrl('file:///tmp/index.html'), null)

const candidates = localPreviewCandidates(thread)
assert.equal(candidates[0].url, 'http://localhost:5173/')
assert.equal(candidates[0].source, 'thread')
assert.ok(candidates.some((candidate) => candidate.url === 'http://localhost:3000/dashboard'))
assert.ok(candidates.some((candidate) => candidate.url === 'http://localhost:5173/'))
assert.ok(!candidates.some((candidate) => candidate.url.includes('192.168.1.22')))

const blockedDiagnostics = browserClickDiagnostics({
  domReadable: false,
  note: 'Clicking Save did nothing',
  error: 'Permission denied',
})
assert.ok(blockedDiagnostics.some((line) => line.includes('User note: Clicking Save did nothing')))
assert.ok(blockedDiagnostics.some((line) => line.includes('sandboxed local iframe')))
assert.ok(blockedDiagnostics.some((line) => line.includes('Controller foreground-window context')))

const readableDiagnostics = browserClickDiagnostics({
  domReadable: true,
  interactiveCount: 4,
  note: 'Clicking Save is intercepted by the lower chat bar',
})
assert.ok(readableDiagnostics.some((line) => line.includes('4 interactive elements')))
assert.ok(readableDiagnostics.some((line) => line.includes('overlays, disabled controls, or a stale preview URL')))

const evidence = formatBrowserEvidence({
  url: 'http://localhost:5174/',
  capturedAt: Date.UTC(2026, 5, 19, 12, 0, 0),
  loadState: 'loaded',
  cwd: thread.cwd,
  inspection: {
    domReadable: false,
    diagnostics: blockedDiagnostics,
  },
})
assert.match(evidence, /^<browser_evidence>/)
assert.match(evidence, /Source: TensorFold local Preview \(explicit user attachment\)/)
assert.match(evidence, /Workspace: \/Users\/test\/Projects\/demo-app/)
assert.match(evidence, /Click diagnostics:/)
assert.match(evidence, /<\/browser_evidence>$/)

console.log('Preview helper tests passed.')
