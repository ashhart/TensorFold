import { reviewedLanguageImport } from '../components/preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import ts from 'typescript'
import * as shared from '../../shared/composer-drafts'
import * as coordinators from './conversation-drafts'
import * as updateSafety from './update-safety'
import * as fileSources from './composer-file-sources'

process.env.VONTRA_HOME = mkdtempSync(path.join(os.tmpdir(), 'tensorfold-draft-hook-'))
const persistence = await import('../../electron/store/composer-drafts')
const port = { getComposerDraft: persistence.readComposerDraft, saveComposerDraft: persistence.writeComposerDraft, clearComposerDraft: persistence.clearComposerDraft }
const cleanups: (() => void)[] = []
const imports: Record<string, unknown> = {
  react: { useMemo: (build: any) => build(), useCallback: (fn: any) => fn, useEffect: (effect: any) => { const cleanup = effect(); if (cleanup) cleanups.push(cleanup) }, useSyncExternalStore: (_subscribe: any, get: any) => get() },
  '../../shared/composer-drafts': shared,
  './conversation-drafts': coordinators,
  './update-safety': updateSafety,
  './composer-file-sources': fileSources,
}
const module = { exports: {} as any }
const output = ts.transpileModule(readFileSync(path.join(process.cwd(), 'src/lib/use-conversation-draft.ts'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
new Function('require', 'module', 'exports', output)((id: string) => { const reviewed = reviewedLanguageImport(id, imports); if (reviewed.matched) return reviewed.value;  assert.ok(id in imports, `Reviewed production hook import ${id}`); return imports[id] }, module, module.exports)
const { useConversationDraft, useConversationDraftLifecycle } = module.exports
const a = { surface: 'chat' as const, conversationId: 'hook-a', hostId: null, incognito: false }, b = { ...a, conversationId: 'hook-b' }
const hookA = useConversationDraft(a, port)
hookA.setText('Draft A from real hook')
const hookB = useConversationDraft(b, port)
hookB.setText('Draft B from real hook')
// A callback captured before navigation continues to address its original scope.
hookA.setAttachments([{ id: 'file-a', name: 'handwriting.png', kind: 'image', mime: 'image/png', dataUrl: 'data:image/png;base64,fixture' }])
await hookA.flush(); await hookB.flush()
assert.equal((await persistence.readComposerDraft(a)).text, 'Draft A from real hook')
assert.equal((await persistence.readComposerDraft(b)).text, 'Draft B from real hook')
assert.equal((await persistence.readComposerDraft(a)).attachments[0].name, 'handwriting.png')
assert.equal((await persistence.readComposerDraft(b)).attachments.length, 0)
const sent = hookA.capture()
hookA.setText((previous: string) => `${previous} plus newer typing`)
assert.equal(await hookA.clearAccepted(sent), false)
assert.equal(useConversationDraft(a, port).text, 'Draft A from real hook plus newer typing')
let flush!: (requestId: string) => void, reset!: (scope: shared.ComposerDraftScope | null) => void
const acknowledgements: any[] = []
const bridge = {
  onComposerDraftFlush(callback: (requestId: string) => void) { flush = callback; return () => {} },
  onComposerDraftReset(callback: (scope: shared.ComposerDraftScope | null) => void) { reset = callback; return () => {} },
  async acknowledgeComposerDraftFlush(input: any) { acknowledgements.push(input) },
}
useConversationDraftLifecycle(port, bridge)
assert.match(updateSafety.rendererUpdateBlockReason()!, /Saving/)
flush('synthetic-quit')
await coordinators.conversationDraftsFor(port).flushAll()
for (let index = 0; index < 20; index++) await Promise.resolve()
assert.deepEqual(acknowledgements, [{ requestId: 'synthetic-quit' }])
assert.equal(updateSafety.rendererUpdateBlockReason(), null)
assert.equal((await persistence.readComposerDraft(a)).text, 'Draft A from real hook plus newer typing', 'Quit acknowledgement follows last-keystroke persistence')
const normalSourceMap = fileSources.composerFileSources(shared.composerDraftKey(a))
const originalFile = new File(['Synthetic attachment bytes'], 'handwriting.png', { type: 'image/png' })
normalSourceMap.set('file-a', originalFile)
const unrelatedSourceMap = fileSources.composerFileSources(shared.composerDraftKey(b))
unrelatedSourceMap.set('file-b', new File(['Synthetic other attachment'], 'other.txt'))
reset({ ...a, incognito: true })
assert.equal(useConversationDraft({ ...a, incognito: true }, port).text, 'Draft A from real hook plus newer typing')
assert.equal(useConversationDraft(a, port).text, '')
assert.equal(fileSources.composerFileSources(shared.composerDraftKey({ ...a, incognito: true })).get('file-a'), originalFile, 'Actual lifecycle privacy reset transfers original handles into the private session')
assert.notEqual(fileSources.composerFileSources(shared.composerDraftKey(a)), normalSourceMap, 'Actual lifecycle privacy reset invalidates normal-scope file callbacks')
assert.equal(unrelatedSourceMap.size, 1, 'A privacy transition leaves other conversations alone')
reset(null)
assert.equal(useConversationDraft(b, port).text, '')
assert.equal(unrelatedSourceMap.size, 0, 'Actual lifecycle full Forget releases raw attachment handles')
for (const cleanup of cleanups) cleanup()
await coordinators.conversationDraftsFor(port).flushAll()
coordinators.conversationDraftsFor(port).dispose()
console.log('Actual production draft hook binds old callbacks to their own chat, sends clear conditionally, flushes last typing before quit ACK, and applies privacy reset')
