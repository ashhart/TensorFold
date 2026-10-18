import { useCallback } from 'react'
import { useStore } from '../store'
import { authoredText, type UiTextParameters } from './language-packs'
/** A locale-only subscription preserves component state and in-progress text selection. */
export function useAuthoredText() {
  const locale = useStore(state => state.settings?.uiPreferences?.locale)
  return useCallback((source: string, params?: UiTextParameters) => authoredText(locale, source, params), [locale])
}
