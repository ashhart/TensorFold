import type { Settings } from '../../types'

export function systemPrefersDark(): boolean {
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false
}

export function resolveTheme(s: Settings | null): 'light' | 'dark' {
  if (!s || s.theme === 'system') return systemPrefersDark() ? 'dark' : 'light'
  return s.theme
}
