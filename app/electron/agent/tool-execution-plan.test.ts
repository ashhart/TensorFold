import assert from 'node:assert/strict'
import { executeToolExecutionBatches, planToolExecutionBatches } from './tool-execution-plan'
import type { ToolCall } from './types'

const call = (id: string, name: string): ToolCall => ({
  id,
  type: 'function',
  function: { name, arguments: '{}' },
})

const calls = [
  call('1', 'read_file'),
  call('2', 'search'),
  call('3', 'write_file'),
  call('4', 'read_file'),
  call('5', 'glob'),
  call('6', 'web_fetch'),
  call('7', 'run_command'),
]
const parallel = new Set(['read_file', 'search', 'glob', 'web_fetch'])
const batches = planToolExecutionBatches(calls, (item) => parallel.has(item.function.name), 2)

assert.deepEqual(
  batches.map((batch) => ({ mode: batch.mode, ids: batch.calls.map((item) => item.id) })),
  [
    { mode: 'parallel', ids: ['1', '2'] },
    { mode: 'ordered', ids: ['3'] },
    { mode: 'parallel', ids: ['4', '5'] },
    { mode: 'parallel', ids: ['6'] },
    { mode: 'ordered', ids: ['7'] },
  ],
)

assert.deepEqual(
  planToolExecutionBatches([call('1', 'read_file'), call('2', 'search')], () => true, 0)
    .map((batch) => batch.calls.map((item) => item.id)),
  [['1'], ['2']],
  'invalid limits fall back to one concurrent call',
)

console.log('tool execution planning tests passed')

let active = 0
let maxActive = 0
const committed: string[] = []
const starts: Array<{ id: string; committed: string[] }> = []
await executeToolExecutionBatches(
  calls,
  (item) => parallel.has(item.function.name),
  async (item) => {
    starts.push({ id: item.id, committed: [...committed] })
    active++
    maxActive = Math.max(maxActive, active)
    await new Promise((resolve) => setTimeout(resolve, item.id === '1' ? 15 : 2))
    active--
  },
  (item) => committed.push(item.id),
  2,
)
assert.equal(maxActive, 2, 'read batches execute concurrently up to the configured limit')
assert.deepEqual(committed, calls.map((item) => item.id), 'results commit in source order')
assert.deepEqual(starts.find((entry) => entry.id === '3')?.committed, ['1', '2'], 'the write barrier starts after prior reads commit')
assert.deepEqual(starts.find((entry) => entry.id === '7')?.committed, ['1', '2', '3', '4', '5', '6'], 'later ordered calls see every prior batch')
console.log('tool execution concurrency tests passed')
