import { LANGUAGES, normalizeLanguage } from '../../../shared/languages'
import { languagePacks, languagePackCoverage } from './packs'
export type UiTextParameters = Record<string, string | number | null | undefined>
export function packedText(locale: unknown, source: string): string | undefined {
  const language = normalizeLanguage(locale), pack = languagePacks[language] ?? (language === 'es-419' ? languagePacks.es : undefined)
  if (pack && Object.hasOwn(pack, source)) return pack[source]
  const trimmed = source.trim()
  if (trimmed !== source && pack && Object.hasOwn(pack, trimmed)) return `${source.match(/^\s*/)?.[0] ?? ''}${pack[trimmed]}${source.match(/\s*$/)?.[0] ?? ''}`
  return undefined
}
/** Only app-authored strings may use this lookup; user/model/code text remains original. */
export function authoredText(locale: unknown, source: string, params: UiTextParameters = {}): string {
  const language = normalizeLanguage(locale)
  const template = packedText(locale, source) ?? source
  return template.replace(/\{([a-zA-Z][a-zA-Z0-9_]*)\}/g, (match, key) => {
    if (!Object.hasOwn(params, key)) return match
    const value = params[key]
    if (value == null) return ''
    if (typeof value === 'number') return Number.isFinite(value) ? new Intl.NumberFormat(language).format(value) : ''
    return `\u2068${String(value).replace(/[\u0000-\u001f\u202a-\u202e\u2066-\u2069]/g, ' ').slice(0, 4000)}\u2069`
  })
}

/** Only complete bundled packs are selectable; partial authoring files cannot masquerade as support. */
export function interfaceLanguages() {
  return LANGUAGES.filter(language => language.id === 'en' || (language.id === 'es-419' ? languagePackCoverage.es?.complete : languagePackCoverage[language.id]?.complete))
}
