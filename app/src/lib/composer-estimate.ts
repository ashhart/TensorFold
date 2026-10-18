import type { Attachment } from './types'

export interface ComposerEstimate { tokens: number; partial: boolean; attachmentsCounted: number }
const MAX_ATTACHMENTS = 256, MAX_CHARACTERS = 16_000_000

/** Constant-time string lengths plus a capped attachment walk; never tokenizes conversation history. */
export function boundedComposerEstimate(text: string, attachments: readonly Attachment[]): ComposerEstimate {
  let characters = Math.min(text.length, MAX_CHARACTERS), count = 0
  for (const attachment of attachments) {
    if (count >= MAX_ATTACHMENTS || characters >= MAX_CHARACTERS) break
    characters = Math.min(MAX_CHARACTERS, characters + (attachment.text?.length ?? attachment.name?.length ?? 0)); count++
  }
  return { tokens: 2100 + Math.ceil(characters / 4), partial: count < attachments.length || text.length > MAX_CHARACTERS || characters >= MAX_CHARACTERS, attachmentsCounted: count }
}
