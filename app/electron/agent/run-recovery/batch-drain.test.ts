import assert from 'node:assert/strict'
import { executeToolExecutionBatches } from '../tool-execution-plan'

async function main() {
  let release!: () => void
  let finished = false
  let drained = false
  const pending = new Promise<void>(resolve => { release = resolve })
  const calls = ['failed', 'uncooperative'].map(id => ({ id, type: 'function' as const, function: { name: 'read_file', arguments: '{}' } }))
  const run = executeToolExecutionBatches(calls, () => true, async call => {
    if (call.id === 'failed') throw new Error('Synthetic parallel read failure')
    await pending
    drained = true
  }, () => {}).then(() => { finished = true }, error => { assert.match(error.message, /Synthetic parallel read failure/); finished = true })
  try {
    await new Promise<void>(resolve => setImmediate(resolve))
    assert.equal(finished, false, 'A failed parallel tool must not settle the run while another operation is still active')
  } finally { release(); await run }
  assert.equal(drained, true)
  assert.equal(finished, true)
  let releaseLate!: () => void
  const late = new Promise<void>(resolve => { releaseLate = resolve })
  const chronological = executeToolExecutionBatches(calls, () => true, async call => {
    if (call.id === 'failed') { await late; throw new Error('Later first-source error') }
    throw new Error('First chronological error')
  }, () => {}).then(() => null, error => error as Error)
  await new Promise<void>(resolve => setImmediate(resolve)); releaseLate()
  assert.match((await chronological)!.message, /First chronological error/)
  console.log('PASS real tool-batch rejection waits for every started operation before settlement')
}
main().catch(error => { console.error(error); process.exitCode = 1 })
