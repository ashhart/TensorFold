import { normalizeLanguage, responseLanguageInstruction } from '../../shared/languages'
import type { UiPreferences } from '../../shared/ui-preferences'
import type { ApiMessage } from './types'
/** Host selects a fixed allowlisted language, never user-controlled prompt text. */
export function selectedResponseLocale(preferences: UiPreferences | undefined): string | undefined {
  return preferences?.replyLanguage === 'auto' ? undefined : normalizeLanguage(preferences?.locale)
}
/** Ephemeral and non-mutating: changing language never rewrites saved messages or prompts. */
export function withResponseLanguage(messages: ApiMessage[], locale: unknown): ApiMessage[] {
  const preference = responseLanguageInstruction(locale)
  if (!preference) return messages
  const index = messages.findIndex(message => message.role === 'system')
  if (index < 0) return [{ role: 'system', content: preference }, ...messages]
  return messages.map((message, position) => position === index ? { ...message, content: typeof message.content === 'string' ? `${message.content}\n\n${preference}` : Array.isArray(message.content) ? [...message.content, { type: 'text' as const, text: preference }] : preference } : message)
}
