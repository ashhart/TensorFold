import { PRESETS } from '../agent/theme-presets'
import type { Settings, ThemePalette, ThemePreset } from '../agent/types'

// Editing any colour or font switches a profile to 'custom', so a named preset always means the
// stock palette. A preset that no longer exists moves to the TensorFold default; a named preset
// saved with older values takes its current ones. Custom palettes are never touched.
type NamedPreset = Exclude<ThemePreset, 'custom'>
const same = (a: ThemePalette | undefined, b: ThemePalette) => !!a && JSON.stringify(a) === JSON.stringify(b)
const isNamed = (preset: unknown): preset is NamedPreset => typeof preset === 'string' && Object.hasOwn(PRESETS, preset)

export function migrateThemePreset(settings: Settings): { settings: Settings; changed: boolean } {
  const preset = settings.preset as string
  if (preset === 'custom') return { settings, changed: false }
  const target: NamedPreset = isNamed(preset) ? preset : 'tensorfold'
  const { light, dark } = PRESETS[target]
  if (target === preset && same(settings.light, light) && same(settings.dark, dark)) return { settings, changed: false }
  return { settings: { ...settings, preset: target, light: { ...light }, dark: { ...dark } }, changed: true }
}
