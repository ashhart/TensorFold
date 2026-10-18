import type { LanguageId } from './languages'

/** Profile-owned display choices; no renderer-local persistence. */
export interface UiPreferences {
  locale: LanguageId
  /** Omitted follows the selected interface language; auto follows each request. */
  replyLanguage?: 'interface' | 'auto'
  /** Omitted uses Enter; modified Enter requires Command or Control. */
  sendKey?: 'enter' | 'modified-enter'
  direction: 'auto' | 'ltr' | 'rtl'
  chatScale: 1 | 1.15 | 1.3 | 1.5
  codeScale: 1 | 1.15 | 1.3 | 1.5
  density: 'comfortable' | 'compact'
}
