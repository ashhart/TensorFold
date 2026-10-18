import { normalizeLanguage, type LanguageId } from '../../../shared/languages'
let read: () => unknown = () => 'en'
/** Registered by renderer preferences; pure formatters do not import React or the store. */
export function configureUiLanguageReader(reader: () => unknown): () => void {
  const previous = read; read = reader
  return () => { if (read === reader) read = previous }
}
export function currentUiLanguage(): LanguageId {
  try { return normalizeLanguage(read()) } catch { return 'en' }
}
export function localDateText(value: number | Date, options?: Intl.DateTimeFormatOptions): string {
  const date = value instanceof Date ? value : new Date(value), locale = currentUiLanguage()
  if (!Number.isFinite(date.getTime())) return ''
  if (locale === 'la') return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`
  return new Intl.DateTimeFormat(locale, options ?? { dateStyle: 'medium', timeStyle: 'short' }).format(date)
}

/** English keeps the device's region; selected languages use their own Intl rules. */
export function currentIntlLocale(): string | undefined { const locale = currentUiLanguage(); return locale === 'en' ? undefined : locale }
