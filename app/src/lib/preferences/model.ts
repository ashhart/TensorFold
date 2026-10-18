import { normalizeLanguage } from '../../../shared/languages'
import type { UiPreferences } from '../../../shared/ui-preferences'

export const DEFAULT_UI_PREFERENCES: Readonly<UiPreferences> = Object.freeze({ locale: 'en', direction: 'auto', chatScale: 1, codeScale: 1, density: 'comfortable' })
export const UI_SCALES = [1, 1.15, 1.3, 1.5] as const
export function normalizeUiPreferences(input: unknown): UiPreferences {
  const value = input && typeof input === 'object' && !Array.isArray(input) ? input as Record<string, unknown> : {}
  return {
    locale: normalizeLanguage(value.locale), ...(value.replyLanguage === 'auto' ? { replyLanguage: 'auto' as const } : {}), ...(value.sendKey === 'modified-enter' ? { sendKey: 'modified-enter' as const } : {}),
    direction: value.direction === 'rtl' || value.direction === 'ltr' ? value.direction : 'auto',
    chatScale: UI_SCALES.includes(value.chatScale as any) ? value.chatScale as UiPreferences['chatScale'] : 1,
    codeScale: UI_SCALES.includes(value.codeScale as any) ? value.codeScale as UiPreferences['codeScale'] : 1,
    density: value.density === 'compact' ? 'compact' : 'comfortable',
  }
}
