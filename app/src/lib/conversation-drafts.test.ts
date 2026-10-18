import assert from 'node:assert/strict'
import { ConversationDrafts, draftAttachmentReferences } from './conversation-drafts'
import { composerDraftKey, emptyComposerDraft } from '../../shared/composer-drafts'
import type { ComposerDraftPort, ComposerDraftScope, ComposerDraftSnapshot, ComposerDraftWrite, ComposerDraftWriteResult } from '../../shared/composer-drafts'
import type { Attachment } from './types'

const scope = (id: string, hostId: string | null = null): ComposerDraftScope => ({ surface: 'chat', conversationId: id, hostId, incognito: false })
const a = scope('a'), b = scope('b')
const tick = async () => { for (let i = 0; i < 20; i++) await Promise.resolve() }
function deferred<T>() { let resolve!: (value: T) => void, reject!: (error: Error) => void; const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no }); return { promise, resolve, reject } }
function memoryPort() {
  const saved = new Map<string, ComposerDraftSnapshot>()
  const writes: ComposerDraftWrite[] = []
  const port: ComposerDraftPort = {
    async getComposerDraft(target) { return structuredClone(saved.get(composerDraftKey(target)) ?? emptyComposerDraft()) },
    async saveComposerDraft(input) {
      writes.push(structuredClone(input))
      const old = saved.get(composerDraftKey(input.scope)) ?? emptyComposerDraft()
      if (old.revision !== input.expectedRevision) return { saved: false, draft: structuredClone(old) }
      const draft = { text: input.text, attachments: input.attachments, revision: old.revision + 1, updatedAt: Date.now() }
      saved.set(composerDraftKey(input.scope), structuredClone(draft))
      return { saved: true, draft }
    },
    clearComposerDraft(input) { return port.saveComposerDraft({ ...input, text: '', attachments: [] }) },
  }
  return { port, saved, writes }
}

// Exercise late hydration with two active conversation owners, including input typed before load.
{
  const { port, saved } = memoryPort()
  const reads = new Map<string, ReturnType<typeof deferred<ComposerDraftSnapshot>>>()
  port.getComposerDraft = (target) => { const response = deferred<ComposerDraftSnapshot>(); reads.set(target.conversationId!, response); return response.promise }
  const drafts = new ConversationDrafts(port, 60_000)
  drafts.get(a); drafts.update(a, { text: 'new A typed before load' }); drafts.get(b)
  reads.get('b')!.resolve({ revision: 3, text: 'saved B', attachments: [], updatedAt: 1 })
  reads.get('a')!.resolve({ revision: 2, text: 'old A', attachments: [{ id: 'ref', name: 'pending.png', kind: 'image', mime: 'image/png', status: 'needs-reselection' }], updatedAt: 1 })
  await tick()
  assert.equal(drafts.get(b).text, 'saved B', 'Late A hydration cannot overwrite B')
  assert.equal(drafts.get(a).text, 'new A typed before load', 'Late hydration cannot replace newly typed text')
  assert.equal(drafts.get(a).attachments[0].name, 'pending.png', 'Typing before load does not lose untouched saved attachment references')
  saved.set(composerDraftKey(a), { revision: 2, text: 'old A', attachments: [], updatedAt: 1 })
  await drafts.flush(a)
  assert.equal(saved.get(composerDraftKey(a))!.text, 'new A typed before load')
  assert.equal(drafts.get(b).text, 'saved B')
  drafts.dispose()
}

// Pending write + accepted send + newer edit. A late completion never clears other/newer input.
{
  const { port, saved, writes } = memoryPort()
  const original = port.saveComposerDraft.bind(port)
  const gate = deferred<void>()
  let first = true
  port.saveComposerDraft = async (input) => { if (first) { first = false; await gate.promise } return original(input) }
  const drafts = new ConversationDrafts(port, 60_000)
  drafts.get(a); drafts.get(b); await tick()
  drafts.update(a, { text: 'sent A' }); const sent = drafts.capture(a)
  const pendingSave = drafts.flush(a); await tick()
  drafts.update(b, { text: 'unsent B' })
  drafts.update(a, { text: 'new A after send began' })
  assert.equal(await drafts.clearAccepted(sent), false, 'Accepted old send cannot clear newer typing')
  gate.resolve(); await pendingSave; await drafts.flushAll()
  assert.equal(saved.get(composerDraftKey(a))!.text, 'new A after send began')
  assert.equal(saved.get(composerDraftKey(b))!.text, 'unsent B')
  assert.deepEqual(writes.filter((write) => write.scope.conversationId === 'a').map((write) => write.expectedRevision), [0, 1])
  const exact = drafts.capture(a)
  assert.equal(await drafts.clearAccepted(exact), true)
  assert.equal(saved.get(composerDraftKey(a))!.text, '')
  assert.equal(drafts.get(b).text, 'unsent B')
  const restarted = new ConversationDrafts(port, 60_000); restarted.get(a); restarted.get(b); await tick()
  assert.equal(restarted.get(a).text, '', 'Cleared accepted send stays cleared after restart')
  assert.equal(restarted.get(b).text, 'unsent B', 'Other draft survives coordinator restart')
  restarted.dispose(); drafts.dispose()
}

// Empty clear is ordered behind a pending populated save, so old data cannot resurrect.
{
  const { port, saved } = memoryPort()
  const original = port.saveComposerDraft.bind(port), gate = deferred<void>()
  let block = true
  port.saveComposerDraft = async (input) => { if (block) { block = false; await gate.promise } return original(input) }
  const drafts = new ConversationDrafts(port, 60_000); drafts.get(a); await tick()
  drafts.update(a, { text: 'will be accepted' }); const token = drafts.capture(a)
  const pending = drafts.flush(a); await tick()
  const cleared = drafts.clearAccepted(token); await tick()
  gate.resolve(); await pending; await cleared
  assert.equal(saved.get(composerDraftKey(a))!.text, '')
  assert.equal(saved.get(composerDraftKey(a))!.revision, 2)
  drafts.dispose()
}

// Save/read failures remain visible and preserve input; retries use the original CAS revision.
{
  const { port, saved } = memoryPort(), original = port.saveComposerDraft.bind(port)
  let fail = true
  port.saveComposerDraft = async (input) => { if (fail) throw new Error('Synthetic disk full.'); return original(input) }
  const drafts = new ConversationDrafts(port, 60_000); drafts.get(a); await tick()
  drafts.update(a, { text: 'must never disappear' })
  await assert.rejects(() => drafts.flush(a), /could not be saved/)
  assert.equal(drafts.get(a).text, 'must never disappear'); assert.equal(drafts.get(a).dirty, true)
  assert.match(drafts.pendingReason()!, /could not be saved/)
  fail = false; await drafts.retry(a)
  assert.equal(drafts.pendingReason(), null); assert.equal(saved.get(composerDraftKey(a))!.text, 'must never disappear')
  // Another writer is not overwritten by a retry or by accepting a local send.
  saved.set(composerDraftKey(a), { revision: 9, text: 'other window', attachments: [], updatedAt: 1 })
  drafts.update(a, { text: 'mine' })
  await assert.rejects(() => drafts.flush(a), /another window/)
  await assert.rejects(() => drafts.retry(a), /another window/)
  assert.equal(saved.get(composerDraftKey(a))!.text, 'other window'); assert.equal(drafts.get(a).text, 'mine')
  drafts.dispose()
}

// A failed load must not be treated as an empty authoritative draft or written over on Quit.
{
  const { port, saved } = memoryPort(), original = port.getComposerDraft.bind(port)
  let fail = true
  port.getComposerDraft = async (target) => { if (fail) throw new Error('Synthetic unavailable protected state.'); return original(target) }
  const drafts = new ConversationDrafts(port, 60_000); drafts.get(a); await tick()
  await drafts.flushAll()
  assert.equal(drafts.get(a).dirty, false, 'An unreadable untouched prior draft does not trap Quit; original bytes are preserved')
  drafts.update(a, { text: 'new local text' })
  await assert.rejects(() => drafts.flushAll(), /Could not load/)
  assert.equal(saved.size, 0)
  fail = false; await drafts.retry(a)
  assert.equal(saved.get(composerDraftKey(a))!.text, 'new local text')
  drafts.dispose()
}

// In-memory payloads survive chat navigation; disk references deliberately do not contain them.
{
  const { port, saved } = memoryPort()
  const drafts = new ConversationDrafts(port, 60_000); drafts.get(a); await tick()
  const image: Attachment = { id: 'image', kind: 'image', name: 'sample.png', mime: 'image/png', dataUrl: 'data:image/png;base64,fixture', bytes: 7 }
  drafts.update(a, { text: 'attachment draft', attachments: [image] }); drafts.get(b)
  assert.equal(drafts.get(a).attachments[0].dataUrl, image.dataUrl)
  await drafts.flushAll()
  assert.ok(!JSON.stringify([...saved.values()]).includes('base64'))
  assert.equal((saved.get(composerDraftKey(a))!.attachments[0] as any).dataUrl, undefined)
  const restarted = new ConversationDrafts(port, 60_000); restarted.get(a); await tick()
  assert.equal(restarted.get(a).attachments[0].name, 'sample.png')
  assert.equal((restarted.get(a).attachments[0] as any).status, 'needs-reselection')
  assert.equal(restarted.get(a).attachments[0].dataUrl, undefined)
  const accepted = { ...image, extraction: { state: 'accepted', text: 'User corrected this.' }, source: { path: '/private/fixture' } } as any
  const refs = draftAttachmentReferences([accepted])
  assert.equal(refs[0].acceptedText, 'User corrected this.')
  assert.equal((refs[0] as any).source, undefined)
  drafts.dispose(); restarted.dispose()
}

{
  const { port, saved, writes } = memoryPort()
  let readCalls = 0; port.getComposerDraft = async () => { readCalls++; throw new Error('Incognito must not call disk') }
  const drafts = new ConversationDrafts(port, 60_000), privateScope = { ...a, incognito: true }
  drafts.update(privateScope, { text: 'incognito contents' }); await drafts.flushAll()
  assert.equal(readCalls, 0); assert.equal(writes.length, 0); assert.equal(saved.size, 0)
  assert.equal(drafts.get(privateScope).text, 'incognito contents'); assert.equal(drafts.pendingReason(), null)
  assert.notEqual(composerDraftKey(a), composerDraftKey({ ...a, hostId: 'different-host' }))
  assert.notEqual(composerDraftKey(a), composerDraftKey(privateScope))
  assert.equal(composerDraftKey(a), composerDraftKey({ ...a, surface: 'assistant' }), 'Same conversation retains draft across chat surfaces')
  const unavailable = new ConversationDrafts(null)
  unavailable.update(a, { text: 'browser-owned text' }); await unavailable.flushAll()
  assert.match(unavailable.get(a).error!, /local desktop/)
  drafts.dispose(); unavailable.dispose()
}

{
  const { port, writes } = memoryPort()
  const drafts = new ConversationDrafts(port, 60_000)
  drafts.get(a); await tick(); drafts.update(a, { text: 'unfinished before Incognito' }); await drafts.flush(a)
  drafts.update(a, { attachments: [
    { id: 'finished', name: 'ready.txt', kind: 'document', mime: 'text/plain', text: 'Synthetic ready file', status: 'ready' },
    { id: 'in-flight', name: 'still-reading.txt', kind: 'document', mime: 'text/plain', status: 'processing' },
    { id: 'review', name: 'review.png', kind: 'image', mime: 'image/png', status: 'review', extraction: { method: 'apple-vision', state: 'review', text: 'Uncertain transcription', confidence: null, warnings: ['Review required'], pageCount: 1, processedPages: 1, truncated: false } },
    { id: 'accepted', name: 'accepted.png', kind: 'image', mime: 'image/png', status: 'ready', dataUrl: 'data:image/png;base64,fixture', extraction: { method: 'apple-vision', state: 'accepted', text: 'User accepted correction', confidence: null, warnings: [], pageCount: 1, processedPages: 1, truncated: false } },
  ] })
  await drafts.flush(a)
  const writesBefore = writes.length
  drafts.makeIncognito({ ...a, incognito: true })
  assert.equal(drafts.get({ ...a, incognito: true }).text, 'unfinished before Incognito')
  assert.equal(drafts.get(a).text, '')
  const transferred = drafts.get({ ...a, incognito: true }).attachments
  assert.equal(transferred[0].status, 'ready', 'Completed files remain available after the privacy transition')
  assert.equal(transferred[0].text, 'Synthetic ready file')
  assert.equal(transferred[1].status, 'needs-reselection', 'An old normal-scope file completion cannot leave the private row processing forever')
  assert.match(transferred[1].error!, /interrupted.*Incognito/)
  assert.equal(transferred[2].status, 'needs-reselection', 'Unaccepted review cannot silently become ready during a privacy transition')
  assert.equal(transferred[3].status, 'ready')
  assert.equal(transferred[3].dataUrl, 'data:image/png;base64,fixture')
  assert.equal(transferred[3].extraction!.text, 'User accepted correction')
  drafts.update({ ...a, incognito: true }, { text: 'new private typing' }); await drafts.flushAll()
  assert.equal(writes.length, writesBefore, 'Private transition and private edits never trigger a durable draft save')
  drafts.dispose()
}

{
  const { port, saved } = memoryPort()
  const drafts = new ConversationDrafts(port, 60_000); drafts.get(a); await tick()
  const references = Array.from({ length: 200 }, (_, index) => ({ id: `overflow-${index}`, name: `selected-${index}.txt`, kind: 'document' as const, mime: 'text/plain', status: 'error', error: 'Per-file payload limit reached.' } as Attachment))
  drafts.update(a, { attachments: references }); await drafts.flush(a)
  assert.equal(saved.get(composerDraftKey(a))!.attachments.length, 200, 'All bounded error/overflow names survive draft persistence')
  drafts.update(a, { attachments: [...references, { ...references[0], id: 'overflow-201' }] })
  await assert.rejects(() => drafts.flush(a), /more than 200 file references/)
  assert.equal(drafts.get(a).attachments.length, 201, 'Excess rows stay visible in memory with an explicit persistence limit')
  assert.equal(drafts.get(a).saving, false)
  drafts.dispose()
}

// Host Forget invalidates late hydration and saves, clears memory and blocks old send tokens.
{
  const { port } = memoryPort()
  const read = deferred<ComposerDraftSnapshot>()
  port.getComposerDraft = () => read.promise
  const drafts = new ConversationDrafts(port, 60_000)
  drafts.get(a); drafts.update(a, { text: 'before Forget' }); const oldSend = drafts.capture(a)
  drafts.forget()
  read.resolve({ revision: 3, text: 'old saved contents', attachments: [], updatedAt: 1 }); await tick()
  assert.equal(drafts.get(a).text, '')
  assert.equal(drafts.pendingReason(), null)
  assert.equal(await drafts.clearAccepted(oldSend), false)
  drafts.dispose()
}
console.log('Conversation draft ownership, late hydration/writes, send-clear, failure/retry, restart references, host boundaries and incognito passed')
