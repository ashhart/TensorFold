import assert from 'node:assert/strict'
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

process.env.VONTRA_HOME = path.join(os.tmpdir(), `vontra-audit-${Date.now()}-${Math.random().toString(16).slice(2)}`)
mkdirSync(process.env.VONTRA_HOME, { recursive: true, mode: 0o700 })

const { auditEvent, flushAuditEvents, runAuditBarrier } = await import('./audit.ts')

function events(): any[] {
  const file = path.join(process.env.VONTRA_HOME!, 'security-events.jsonl')
  return readFileSync(file, 'utf8').trimEnd().split('\n').filter(Boolean).map((line) => JSON.parse(line))
}

async function main() {
  const file = path.join(process.env.VONTRA_HOME!, 'security-events.jsonl')
  await auditEvent('audit.first', { marker: 1 })
  const first = events().at(-1)
  assert.ok(first?.hash, 'precondition: first audit event has a hash')

  rmSync(file)
  await auditEvent('audit.after_delete', { marker: 2 })
  const afterDelete = events().at(-1)
  assert.equal(afterDelete.kind, 'audit.after_delete')
  assert.match(String(afterDelete.prevHash), /^missing:/, 'deleted audit log must not restart as genesis')
  assert.match(String(afterDelete.prevHash), new RegExp(String(first.hash)), 'deleted audit log records the prior anchored hash')

  writeFileSync(file, '', { mode: 0o600 })
  await auditEvent('audit.after_truncate', { marker: 3 })
  const afterTruncate = events().at(-1)
  assert.equal(afterTruncate.kind, 'audit.after_truncate')
  assert.match(String(afterTruncate.prevHash), /^truncated:/, 'truncated audit log must not restart as genesis')

  const anchor = path.join(process.env.VONTRA_HOME!, 'security-events.anchor.json')
  assert.ok(existsSync(anchor), 'audit anchor persists the last hash outside the jsonl log')

  void auditEvent('audit.queued_one', { marker: 'erase-me-1' })
  void auditEvent('audit.queued_two', { marker: 'erase-me-2' })
  await flushAuditEvents()
  assert.deepEqual(events().slice(-2).map((event) => event.kind), ['audit.queued_one', 'audit.queued_two'], 'flush waits for every event captured before it')

  void auditEvent('audit.pre_barrier_secret', { marker: 'QUEUED-SECRET' })
  await runAuditBarrier(async () => {
    rmSync(file, { force: true })
    rmSync(anchor, { force: true })
  })
  await auditEvent('audit.after_barrier', { marker: 'fresh' })
  assert.doesNotMatch(readFileSync(file, 'utf8'), /QUEUED-SECRET|audit\.pre_barrier_secret/, 'barrier runs after queued appends and before later appends')

  const opaqueSecret = 'uTy7Jk_3nQp8ZrV2xLm9AaBbCcDdEeFfGgHhIiJj'
  await auditEvent('audit.secret_sanitization', {
    nested: { explanation: `approval label api_key=${opaqueSecret}`, values: [`Bearer ${opaqueSecret}`] },
    count: 7,
    ok: true,
  })
  const sanitized = events().at(-1)
  assert.doesNotMatch(JSON.stringify(sanitized), new RegExp(opaqueSecret), 'central audit boundary redacts secrets even in nested caller fields')
  assert.equal(sanitized.count, 7, 'numeric audit fields remain queryable')
  assert.equal(sanitized.ok, true, 'boolean audit fields remain queryable')
  console.log('audit tamper-anchor tests passed')
}

await main()
