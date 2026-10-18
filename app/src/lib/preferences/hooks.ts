import { configureUiLanguageReader } from './current-locale'
import { useCallback, useLayoutEffect } from 'react'
import { useStore } from '../store'
import { applyUiPreferences } from './apply'
import { normalizeUiPreferences } from './model'
import { translate, type UiTextKey, type UiTextParams } from './catalogue'

configureUiLanguageReader(() => useStore.getState().settings?.uiPreferences?.locale)

export function useUiPreferences(input: unknown): void {
  const value = normalizeUiPreferences(input)
  useLayoutEffect(() => applyUiPreferences(value), [value.locale, value.direction, value.chatScale, value.codeScale, value.density])
}
export function useUiText(): (key: UiTextKey, params?: UiTextParams) => string {
  const locale = useStore((state) => state.settings?.uiPreferences?.locale)
  return useCallback((key: UiTextKey, params?: UiTextParams) => translate(locale, key, params), [locale])
}
