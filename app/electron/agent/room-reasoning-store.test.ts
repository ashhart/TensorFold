import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import type { Room, Ticket } from './types'

if (process.argv.includes('--reload')) {
  const { loadRooms, loadTickets } = await import('../store')
  const rooms = await loadRooms()
  assert.equal(rooms.length, 1)
  assert.deepEqual(rooms[0].seats.map(seat => seat.reasoningEffort), ['low', 'high', undefined])
  assert.equal(Object.hasOwn(rooms[0].seats[2], 'reasoningEffort'), false)
  assert.equal((await loadTickets())[0].execution?.reasoningEffort, 'low', 'the settled worker capture survives reload')
  console.log('A fresh process loaded the saved per-agent efforts and legacy fallback.')
} else {
  const profile = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-room-effort-store-'))
  const previousHome = process.env.VONTRA_HOME
  process.env.VONTRA_HOME = profile
  try {
    const { saveRooms, saveTickets } = await import('../store')
    const seat = { name: 'Reviewer', role: 'Review supplied input', providerId: 'fixture', model: 'fixture' }
    const room: Room = {
      id: 'room-effort', projectId: 'project', name: 'Synthetic room', problem: 'Review supplied input',
      seats: [{ ...seat, id: 'low', reasoningEffort: 'low' }, { ...seat, id: 'high', reasoningEffort: 'high' },
        { ...seat, id: 'legacy' }],
      state: 'ready', events: [], finalText: '', createdAt: 1, updatedAt: 1,
    }
    await saveRooms([room])
    const ticket: Ticket = { id: 'ticket', projectId: 'project', roomId: room.id, title: 'Synthetic work', body: '',
      type: 'task', status: 'review', priority: 'medium', effort: 'small', tags: [], parentId: null,
      assignee: 'Reviewer', threadId: 'worker', runState: 'done', order: 1, createdAt: 1, updatedAt: 2,
      execution: { target: 'room', threadId: 'worker', reasoningEffort: 'low', startedAt: 1, finishedAt: 2,
        checks: [], blockers: [], reviewVerdict: 'pending' } }
    await saveTickets([ticket])
    const output = execFileSync(process.execPath, [
      'node_modules/jiti/lib/jiti-cli.mjs', fileURLToPath(import.meta.url), '--reload',
    ], { env: process.env, timeout: 15_000, encoding: 'utf8' })
    assert.match(output, /A fresh process loaded the saved per-agent efforts and legacy fallback/)
    console.log('Protected Room settings survive a process restart without converting legacy defaults.')
  } finally {
    if (previousHome === undefined) delete process.env.VONTRA_HOME
    else process.env.VONTRA_HOME = previousHome
    await fs.rm(profile, { recursive: true, force: true })
  }
}
