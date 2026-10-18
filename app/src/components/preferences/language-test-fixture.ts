import * as currentLocale from '../../lib/preferences/current-locale'
import * as display from '../../lib/preferences/display'
/** Explicit imports for existing source-evaluation fixtures; translation behavior has separate tests. */
export function fixtureText(source: string, params: Record<string, unknown> = {}): string {
  return source.replace(/\{([a-zA-Z][a-zA-Z0-9_]*)\}/g, (token, key) => Object.hasOwn(params, key) ? String(params[key] ?? '') : token)
}
export function reviewedLanguageImport(id: string, imports: Record<string, unknown> = {}): { matched: boolean; value?: any } {
  if (Object.hasOwn(imports, id)) return { matched: false }
  if (id.endsWith('/preferences/useAuthoredText')) return { matched: true, value: { useAuthoredText: () => fixtureText } }
  if (id.endsWith('/preferences/current-locale')) return { matched: true, value: currentLocale }
  if (id.endsWith('/preferences/display')) return { matched: true, value: display }
  if (id.endsWith('/preferences/language-packs')) return { matched: true, value: { authoredText: (_locale: unknown, source: string, params?: Record<string, unknown>) => fixtureText(source, params), interfaceLanguages: () => [{ id: 'en', nativeName: 'English', direction: 'ltr' }, { id: 'es', nativeName: 'Español', direction: 'ltr' }] } }
  if (id.endsWith('/OriginalDiagnostic')) return { matched: true, value: { OriginalDiagnostic: ({ children }: { children: unknown }) => children } }
  if (id.endsWith('/preferences/LanguageQuickSelect')) return { matched: true, value: { LanguageQuickSelect: () => null } }
  if (id.endsWith('/lib/store')) {
    const state = { settings: { uiPreferences: { locale: 'en' } } }
    return { matched: true, value: { useStore: Object.assign((selector: (input: typeof state) => unknown) => selector(state), { getState: () => state }) } }
  }
  return { matched: false }
}

// Direct-call handler fixtures isolate locale subscription, exercised separately in integration tests.
export const useAuthoredText = () => fixtureText
