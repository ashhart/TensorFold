import type { ComposerDraftPort, ComposerDraftScope } from './composer-drafts'
import type { QuickAskDraft } from './quick-ask'
export async function importAskDraft(port: ComposerDraftPort, scope: ComposerDraftScope, draft: QuickAskDraft): Promise<boolean> {
  if (!draft.text && !draft.attachments.length) return false
  const existing = await port.getComposerDraft(scope)
  if (existing.revision !== 0) return false
  const result = await port.saveComposerDraft({ scope, expectedRevision: 0, text: draft.text, attachments: draft.attachments.map(item => ({ id: item.id, kind: item.kind, name: item.name, mime: item.mime, bytes: item.bytes, status: 'needs-reselection', ...(item.kind === 'document' && item.text ? { acceptedText: item.text } : {}) })) })
  if (!result.saved) throw new Error('The conversation draft changed during handoff. Your original ask is preserved.')
  return true
}
