import { normalizeLanguage } from '../../shared/languages'
import type { UiPreferences } from '../../shared/ui-preferences'

/** Host validation is independent of renderer projection; untrusted values never become CSS. */
export function normalizeStoredUiPreferences(input: unknown): UiPreferences {
  const value = input && typeof input === 'object' && !Array.isArray(input) ? input as Record<string, unknown> : {}
  const scale = (candidate: unknown): UiPreferences['chatScale'] => candidate === 1.15 || candidate === 1.3 || candidate === 1.5 ? candidate : 1
  return { locale: normalizeLanguage(value.locale), ...(value.replyLanguage === 'auto' ? { replyLanguage: 'auto' as const } : {}), ...(value.sendKey === 'modified-enter' ? { sendKey: 'modified-enter' as const } : {}), direction: value.direction === 'rtl' || value.direction === 'ltr' ? value.direction : 'auto', chatScale: scale(value.chatScale), codeScale: scale(value.codeScale), density: value.density === 'compact' ? 'compact' : 'comfortable' }
}
