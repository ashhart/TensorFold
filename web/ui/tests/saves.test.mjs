import assert from 'node:assert/strict'
import { test } from 'node:test'
import { Saves } from '../src/saves.ts'

const tick = () => new Promise((r) => setTimeout(r, 0))

// a server that keeps revisions and refuses a save made on an older one, as TensorFold Web does
function server() {
  const db = new Map(); let n = 0
  return {
    db,
    create: async (body) => { const id = `c${++n}`; db.set(id, { body, rev: 1 }); return { id, rev: 1 } },
    save: async (id, body) => {
      const cur = db.get(id) ?? { rev: 0 }
      if (body.rev != null && body.rev !== cur.rev) { const e = new Error('changed elsewhere'); e.status = 409; throw e }
      const rev = cur.rev + 1; db.set(id, { body, rev }); return { id, rev }
    },
  }
}

test('a conversation begun while another is still being created gets its own record', async () => {
  const srv = server(); const holds = []; const created = []
  const api = { create: (b) => new Promise((r) => holds.push(() => r(srv.create(b)))), save: srv.save }
  const s = new Saves(api, (gen, id) => created.push([gen, id]), () => {})
  const a = s.save(1, null, { v: 'A1' })
  const a2 = s.save(1, null, { v: 'A2' })
  const b = s.save(2, null, { v: 'B1' })
  await tick()
  assert.equal(holds.length, 2)
  holds.shift()(); await a
  holds.shift()(); await Promise.all([a2, b])
  assert.deepEqual(created, [[1, 'c1'], [2, 'c2']])
  assert.equal(srv.db.get('c1').body.v, 'A2')
  assert.equal(srv.db.get('c2').body.v, 'B1')
})

test('saves of one conversation go out in order, each on the revision the last one made', async () => {
  const srv = server()
  const s = new Saves(srv, () => {}, () => {})
  await s.save(1, null, { v: 0 })
  for (let i = 1; i <= 5; i++) s.save(1, 'c1', { v: i })
  await s.settled()
  assert.equal(srv.db.get('c1').body.v, 5)
  assert.equal(srv.db.get('c1').rev, 6)
})

test('a save on a revision written elsewhere since is refused, said once, and the conversation is left alone', async () => {
  const srv = server(); const outcomes = []
  const s = new Saves(srv, () => {}, (o) => outcomes.push(o))
  await s.save(1, null, { v: 'mine' })
  srv.db.set('c1', { body: { v: 'another tab' }, rev: 7 })           // written by another page meanwhile
  await s.save(1, 'c1', { v: 'stale' })
  await s.save(1, 'c1', { v: 'staler' })
  assert.deepEqual(outcomes, ['saved', 'conflict'])
  assert.equal(srv.db.get('c1').body.v, 'another tab')
  s.read('c1', 7)                                                     // opened again: saves go on from there
  await s.save(2, 'c1', { v: 'after reopening' })
  assert.equal(srv.db.get('c1').body.v, 'after reopening')
})

test('A, then B, then A again: A is read only once its last save has its answer', async () => {
  const srv = server(); const gates = []
  const api = { create: srv.create, save: (id, b) => (b.v === 'A edit' ? new Promise((r) => gates.push(() => r(srv.save(id, b)))) : srv.save(id, b)) }
  const s = new Saves(api, () => {}, () => {})
  await s.save(1, null, { v: 'A' })
  s.save(1, 'c1', { v: 'A edit' })                                    // slow
  let settled = false
  const wait = s.settled().then(() => { settled = true })
  await tick()
  assert.equal(settled, false)                                        // the page would not read A yet
  gates.shift()()
  await wait
  assert.equal(srv.db.get('c1').body.v, 'A edit')                     // read now, A has its edit
})

test('a failed creation is tried again by the next save', async () => {
  let fail = true; const done = []
  const srv = server()
  const s = new Saves({ create: async (b) => { if (fail) { fail = false; throw new Error('offline') } return srv.create(b) }, save: srv.save },
    () => {}, (o) => done.push(o))
  await s.save(1, null, { v: 1 })
  await s.save(1, null, { v: 2 })
  await s.save(1, null, { v: 3 })
  assert.deepEqual(done, ['failed', 'saved', 'saved'])
  assert.equal(srv.db.get('c1').body.v, 3)
})

test('reopening after a conflict drops the saves made on the old text, even one queued during the reopening', async () => {
  const srv = server(); const outcomes = []
  const s = new Saves(srv, () => {}, (o) => outcomes.push(o))
  await s.save(1, null, { v: 'mine' })
  srv.db.set('c1', { body: { v: 'remote edit' }, rev: 5 })
  await s.save(1, 'c1', { v: 'stale' })                       // the conflict is found
  const queued = s.save(1, 'c1', { v: 'sent while Reopen loads' })   // made on the old text
  s.read('c1', 5)                                              // Reopen read the server's version
  await queued
  assert.equal(srv.db.get('c1').body.v, 'remote edit')         // never overwritten
  await s.save(2, 'c1', { v: 'after reopening' })
  assert.equal(srv.db.get('c1').body.v, 'after reopening')
  assert.deepEqual(outcomes, ['saved', 'conflict', 'saved'])
})
