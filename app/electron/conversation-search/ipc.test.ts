import assert from 'node:assert/strict'
import type { Thread } from '../agent/types'
import { registerConversationSearchIpc } from './ipc'

function fixture(id: string): Thread {
  return { id, title: 'Synthetic conversation', kind: 'chat', cwd: '', providerId: 'local', model: 'synthetic',
    createdAt: 1, updatedAt: 2, status: 'idle', apiMessages: [], timeline: [{ id: `${id}-row`, kind: 'user', text: 'Synthetic needle' }] }
}
const handlers = new Map<string, (event: any, ...args: any[]) => any>()
let profile = 'synthetic-one', permitted = true, reads = 0
let threads: Thread[] = [fixture('normal'), { ...fixture('private'), incognito: true }, { ...fixture('remote'), orchestratorHostId: 'remote-profile' }]
const sender = { once: (_: 'destroyed', listener: () => void) => { destroyed = listener } }
let destroyed: () => void = () => undefined
const event = { sender }
registerConversationSearchIpc({ handle: (channel, listener) => { handlers.set(channel, listener) },
  authorize: () => { if (!permitted) throw new Error('Synthetic access refused') },
  getProfileKey: () => profile, getThreads: () => { reads++; return threads } })
const bind = (id: unknown, e = event) => handlers.get('conversation:search:set-active-thread')!(e, id)
const search = (id: string, e = event) => handlers.get('conversation:search')!(e, { scope: 'thread', threadId: id, query: 'needle' })

await assert.rejects(search('private'), /not authorized/)
await bind('private')
assert.equal((await search('private')).matches[0]?.itemId, 'private-row')
await assert.rejects(search('private', { sender: { once: () => undefined } }), /not authorized/, 'another sender cannot borrow current chat authority')
await assert.rejects(search('normal'), /not authorized/, 'request cannot choose a different current chat')
const global = await handlers.get('conversation:search')!(event, { scope: 'all', query: 'needle' })
assert.deepEqual(global.matches.map((match: any) => match.threadId), ['normal'])
await bind(null)
await assert.rejects(search('private'), /not authorized/)
await bind('private')
profile = 'synthetic-two'
await assert.rejects(search('private'), /not authorized/, 'profile namespace change invalidates a current chat binding')
profile = 'synthetic-one'
await assert.rejects(search('private'), /not authorized/, 'old binding cannot revive after returning to a profile')
await bind('private')
destroyed()
await assert.rejects(search('private'), /not authorized/, 'destroyed sender loses its binding')
await bind('private')
await assert.rejects(bind('missing'), /unavailable/)
await assert.rejects(search('private'), /not authorized/, 'refused navigation clears the earlier binding')
await assert.rejects(bind('remote'), /unavailable/, 'mirrored remote history cannot acquire local current-chat authority')
await bind('normal')
threads = []
await assert.rejects(search('normal'), /no longer available/, 'deleted current thread is checked against live membership')
threads = [fixture('normal')]
permitted = false
const before = reads
await assert.rejects(search('normal'), /access refused/)
await assert.rejects(bind('normal'), /access refused/)
assert.equal(reads, before, 'authorization runs before accessing any profile history')
permitted = true
await assert.rejects(search('normal'), /not authorized/, 'revoked authority also clears current binding')

// Delay the second authorization to exercise changes after snippets were computed.
const raceHandlers = new Map<string, (event: any, ...args: any[]) => any>()
let release: () => void = () => undefined, checks = 0, raceProfile = 'one', roleAllowed = true
const raceThreads = [fixture('race')]
const waiting = () => new Promise<void>((resolve) => { release = resolve })
registerConversationSearchIpc({ handle: (channel, listener) => { raceHandlers.set(channel, listener) },
  authorize: async () => { if (!roleAllowed) throw new Error('role revoked'); if (++checks === 3) await waiting(); if (!roleAllowed) throw new Error('role revoked') },
  getProfileKey: () => raceProfile, getThreads: () => raceThreads })
const raceBind = () => raceHandlers.get('conversation:search:set-active-thread')!(event, 'race')
const raceSearch = () => raceHandlers.get('conversation:search')!(event, { scope: 'thread', threadId: 'race', query: 'needle' })
await raceBind()
const oldProfile = raceSearch()
await new Promise<void>((resolve) => setImmediate(resolve))
raceProfile = 'two'
release()
await assert.rejects(oldProfile, /profile changed/)
checks = 0
await raceBind()
const oldNavigation = raceSearch()
await new Promise<void>((resolve) => setImmediate(resolve))
await raceHandlers.get('conversation:search:set-active-thread')!(event, null)
release()
await assert.rejects(oldNavigation, /conversation changed/)
checks = 0
await raceBind()
const revokedRole = raceSearch()
await new Promise<void>((resolve) => setImmediate(resolve))
roleAllowed = false
release()
await assert.rejects(revokedRole, /role revoked/)
roleAllowed = true
checks = 0
await raceBind()
const edited = raceSearch()
await new Promise<void>((resolve) => setImmediate(resolve))
raceThreads[0].timeline[0].text = 'Edited replacement text'
release()
await assert.rejects(edited, /Conversation changed/, 'computed snippets cannot be returned after their live item was edited')
raceThreads[0].timeline[0].text = 'Synthetic needle'
checks = 1
const becamePrivate = raceHandlers.get('conversation:search')!(event, { scope: 'all', query: 'needle' })
await new Promise<void>((resolve) => setImmediate(resolve))
raceThreads[0].incognito = true
release()
await assert.rejects(becamePrivate, /Conversation changed/, 'global excerpts are rechecked after an asynchronous authority boundary')
console.log('Conversation search IPC authorization, sender binding, navigation and profile tests passed')
