import type { Attachment, Thread } from './types'
type ApiMessage = Thread['apiMessages'][number]

export interface ComposerBudget {
  history: number; system: number; draft: number; files: number; quoted: number
  outputReserve: number | null; input: number; total: number; window: number | null; remaining: number | null
  partial: boolean; unreadyFiles: number; unknownImages: number; stale: boolean; approximate: true
}
const MAX_CHARACTERS = 16_000_000, MAX_ROWS = 20_000, MAX_FILES = 256
const positive = (value: unknown): value is number => typeof value === 'number' && Number.isSafeInteger(value) && value > 0

/** Character estimates never claim tokenizer, image, dynamic tool or provider-side accuracy. */
export function composerBudget(input: {
  text: string; attachments: readonly Attachment[]; messages: readonly ApiMessage[]
  windowTokens: number | null; outputReserve?: number | null; quotedText?: string
  windowVerified?: boolean; snapshotAt?: number | null; updatedAt?: number; running?: boolean; now?: number
}): ComposerBudget {
  let remainingChars = MAX_CHARACTERS, partial = input.windowVerified === false, unknownImages = 0, unreadyFiles = 0
  const count = (text: string) => { const size = Math.min(text.length, remainingChars); if (size < text.length) partial = true; remainingChars -= size; return Math.ceil(size / 4) }
  const draft = count(input.text), quoted = count(input.quotedText ?? '')
  let files = 0, history = 0, system = 0, hasSystem = false
  for (let index = 0; index < Math.min(MAX_FILES, input.attachments.length); index++) {
    const file = input.attachments[index]
    if (file.status && file.status !== 'ready') { unreadyFiles++; partial = true }
    files += count(file.name) + 24
    if (file.text) files += count(file.text)
    else if (file.kind === 'image') { unknownImages++; partial = true }
    else { partial = true }
  }
  if (input.attachments.length > MAX_FILES) partial = true
  // Start from the latest entries when very large histories exceed the bounded walk.
  const start = Math.max(0, input.messages.length - MAX_ROWS)
  if (start) partial = true
  for (let index = start; index < input.messages.length; index++) {
    const message = input.messages[index]
    let tokens = 4
    if (typeof message.content === 'string') tokens += count(message.content)
    else if (Array.isArray(message.content)) {
      for (const block of message.content.slice(0, MAX_FILES)) {
        if (block.type === 'text') tokens += count(block.text)
        else { unknownImages++; partial = true }
      }
      if (message.content.length > MAX_FILES) partial = true
    }
    for (const call of (message.tool_calls ?? []).slice(0, MAX_FILES)) tokens += count(call.function.name) + count(call.function.arguments) + 8
    if ((message.tool_calls?.length ?? 0) > MAX_FILES) partial = true
    if (message.role === 'system') { hasSystem = true; system += tokens } else history += tokens
  }
  // Matches the existing composer estimate when transient instructions are not in history.
  if (!hasSystem) { system = 2_100; partial = true }
  const stale = !!input.running || !!input.snapshotAt && (input.snapshotAt < (input.updatedAt ?? 0) || (input.now ?? Date.now()) - input.snapshotAt > 30_000)
  const window = positive(input.windowTokens) ? input.windowTokens : null
  const outputReserve = positive(input.outputReserve) ? input.outputReserve : null
  partial ||= stale || window === null || outputReserve === null
  const tokens = history + system + draft + files + quoted
  const total = tokens + (outputReserve ?? 0)
  return { history, system, draft, files, quoted, outputReserve, input: tokens, total, window, remaining: window === null ? null : window - total, partial, unreadyFiles, unknownImages, stale, approximate: true }
}
