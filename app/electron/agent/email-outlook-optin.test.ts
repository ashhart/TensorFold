// Run: node electron/agent/email-outlook-optin.test.ts
// A plain email scan must never look inside Outlook's Group Container: macOS guards it with a
// "data from other apps" prompt, and the read blocks until someone answers.
import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { scanEmailMessages } from './email-connector.ts'
import type { EmailConnectorSettings } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-email-optin-'))
await fs.writeFile(path.join(root, 'one.eml'), ['Message-ID: <one@example.test>', 'Date: Fri, 19 Jun 2026 09:00:00 +0000', 'From: A <a@example.test>', 'Subject: Hi', '', 'Body'].join('\n'))
const settings: EmailConnectorSettings = {
  sources: [{ id: 'src', label: 'Test Mail', path: root, kind: 'eml_dir', enabled: true, createdAt: 1, updatedAt: 1, lastSyncAt: null, lastError: null }],
  maxMessages: 20,
  lastSyncAt: null,
}

const touched: string[] = []
const realReaddir = fs.readdir
;(fs as any).readdir = (target: unknown, ...rest: unknown[]) => {
  touched.push(String(target))
  return (realReaddir as any).call(fs, target, ...rest)
}
try {
  const { messages, sources } = await scanEmailMessages(settings)
  assert.equal(messages.length, 1)
  assert.deepEqual(sources.map((s) => s.kind), ['eml_dir'])
  assert.ok(!touched.some((p) => p.includes('Group Containers')), `scan read ${touched.find((p) => p.includes('Group Containers'))}`)
  ok('a default scan reads only the configured sources, never Outlook\'s container')

  // The opt-in probe runs against a fake home, never the real Outlook container.
  const fakeHome = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-email-home-'))
  const realHome = process.env.HOME
  process.env.HOME = fakeHome
  try {
    touched.length = 0
    await scanEmailMessages({ ...settings, detectOutlook: true })
    const probed = touched.some((p) => p.startsWith(fakeHome) && p.includes(path.join('Group Containers', 'UBF8T346G9.Office')))
    assert.equal(probed, process.platform === 'darwin', 'Outlook detection probes only on its supported platform')
    ok('detectOutlook respects the platform gate')
  } finally {
    process.env.HOME = realHome
    await fs.rm(fakeHome, { recursive: true, force: true })
  }
} finally {
  ;(fs as any).readdir = realReaddir
  await fs.rm(root, { recursive: true, force: true })
}

console.log(`\n${passed} passed`)
