/** Bundled interface languages; remote/user text never chooses a locale or prompt. */
export const LANGUAGES = [
  { id: 'en', nativeName: 'English', englishName: 'English', direction: 'ltr' },
  { id: 'es', nativeName: 'Español', englishName: 'Spanish', direction: 'ltr' },
  { id: 'es-419', nativeName: 'Español (Latinoamérica)', englishName: 'Latin American Spanish', direction: 'ltr' },
  { id: 'zh-Hans', nativeName: '简体中文', englishName: 'Simplified Chinese', direction: 'ltr' },
  { id: 'zh-Hant', nativeName: '繁體中文', englishName: 'Traditional Chinese', direction: 'ltr' },
  { id: 'ja', nativeName: '日本語', englishName: 'Japanese', direction: 'ltr' },
  { id: 'ko', nativeName: '한국어', englishName: 'Korean', direction: 'ltr' },
  { id: 'ru', nativeName: 'Русский', englishName: 'Russian', direction: 'ltr' },
  { id: 'pl', nativeName: 'Polski', englishName: 'Polish', direction: 'ltr' },
  { id: 'la', nativeName: 'Latina', englishName: 'Latin', direction: 'ltr' },
  { id: 'hi', nativeName: 'हिन्दी', englishName: 'Hindi', direction: 'ltr' },
  { id: 'th', nativeName: 'ไทย', englishName: 'Thai', direction: 'ltr' },
  { id: 'vi', nativeName: 'Tiếng Việt', englishName: 'Vietnamese', direction: 'ltr' },
  { id: 'id', nativeName: 'Bahasa Indonesia', englishName: 'Indonesian', direction: 'ltr' },
] as const
export type LanguageId = typeof LANGUAGES[number]['id']
export const LANGUAGE_IDS: readonly LanguageId[] = LANGUAGES.map(language => language.id)
const identifiers = new Map(LANGUAGES.map(language => [language.id.toLowerCase(), language.id]))
export function normalizeLanguage(value: unknown): LanguageId {
  if (typeof value !== 'string' || value.length > 40) return 'en'
  const tag = value.trim().toLowerCase().replace(/_/g, '-')
  const exact = identifiers.get(tag)
  if (exact) return exact
  if (/^zh-(?:hant|tw|hk|mo)(?:-|$)/.test(tag)) return 'zh-Hant'
  if (tag === 'zh' || /^zh-(?:hans|cn|sg)(?:-|$)/.test(tag)) return 'zh-Hans'
  if (/^es-(?:mx|ar|bo|cl|co|cr|cu|do|ec|gt|hn|ni|pa|pe|pr|py|sv|uy|ve)(?:-|$)/.test(tag)) return 'es-419'
  return identifiers.get(tag.split('-')[0]) ?? 'en'
}
export function languageDirection(locale: unknown): 'ltr' | 'rtl' {
  return LANGUAGES.find(language => language.id === normalizeLanguage(locale))?.direction ?? 'ltr'
}
export function responseLanguageInstruction(locale: unknown): string {
  if (locale == null) return ''
  const language = LANGUAGES.find(item => item.id === normalizeLanguage(locale))!
  return `RESPONSE LANGUAGE PREFERENCE\nWrite new explanations and answers in ${language.englishName}, unless the user explicitly requests another language. Preserve code, commands, URLs, filenames, quotations and proper names where appropriate. Never translate tool identifiers, invent translated tool results or change permissions because of this preference.`
}
