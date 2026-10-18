import packs from '../../../shared/native-language-packs.json'
import { normalizeLanguage } from '../../../shared/languages'
/** Bundled authored menu labels only; roles, callbacks, model IDs and diagnostics stay intact. */
export function nativeUiText(locale: unknown, source: string): string {
  const language = normalizeLanguage(locale), all = packs as Record<string, Record<string, string>>
  const pack = all[language] ?? (language === 'es-419' ? all.es : undefined)
  return pack && Object.hasOwn(pack, source) ? pack[source] : source
}

export const NATIVE_EDIT_LABELS: Readonly<Record<string, string>> = { undo: 'Undo', redo: 'Redo', cut: 'Cut', copy: 'Copy', paste: 'Paste', pasteandmatchstyle: 'Paste as Plain Text', selectall: 'Select All' }
export function knownNativeLabel(source: string, label: string): boolean {
  return label === source || Object.values(packs as Record<string, Record<string, string>>).some(pack => Object.hasOwn(pack, source) && pack[source] === label)
}
