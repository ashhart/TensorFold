// Run: node electron/store/theme-migration.test.ts
import assert from 'node:assert/strict'
import { migrateThemePreset } from './theme-migration.ts'
import { defaultSettings } from './defaults.ts'
import { PRESETS, PRESET_LABELS } from '../agent/theme-presets.ts'
import type { Settings } from '../agent/types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

{
  const s = defaultSettings()
  assert.equal(s.preset, 'tensorfold')
  assert.deepEqual(s.dark, PRESETS.tensorfold.dark)
  assert.equal(migrateThemePreset(s).changed, false)
  ok('fresh profiles start on the TensorFold preset and stay there')
}

{
  assert.deepEqual(Object.keys(PRESETS).sort(), ['glass', 'navy', 'tensorfold'])
  assert.deepEqual(Object.keys(PRESET_LABELS).sort(), Object.keys(PRESETS).sort())
  ok('only TensorFold presets are offered')
}

{
  const stale = { accent: '#123456', background: '#fafafa', foreground: '#111111', uiFont: 'x', codeFont: 'y', translucent: true, contrast: 50 }
  for (const removed of ['synara', 'codex', 'codex-mono', 'zcode', 'claude']) {
    const old = { ...defaultSettings(), preset: removed, light: stale, dark: stale } as unknown as Settings
    const { settings, changed } = migrateThemePreset(old)
    assert.equal(changed, true, removed)
    assert.equal(settings.preset, 'tensorfold', removed)
    assert.deepEqual(settings.light, PRESETS.tensorfold.light)
    assert.deepEqual(settings.dark, PRESETS.tensorfold.dark)
    assert.equal(migrateThemePreset(settings).changed, false, 'the result is stable')
  }
  ok('a removed preset moves to the TensorFold default')
}

{
  const f = { uiFont: PRESETS.tensorfold.light.uiFont, codeFont: PRESETS.tensorfold.light.codeFont }
  const navyUnderDefault = { ...defaultSettings(), preset: 'tensorfold', light: PRESETS.navy.light, dark: PRESETS.navy.dark } as Settings
  assert.deepEqual(migrateThemePreset(navyUnderDefault).settings.dark, PRESETS.tensorfold.dark)
  const oldGlass = { ...defaultSettings(), preset: 'glass',
    light: { accent: '#7C6FE6', background: '#F3F2F8', foreground: '#1B1B22', ...f, translucent: true, contrast: 44 },
    dark: { accent: '#8B7BFF', background: '#16161B', foreground: '#E7E7EC', ...f, translucent: true, contrast: 56 } } as Settings
  const { settings, changed } = migrateThemePreset(oldGlass)
  assert.equal(changed, true)
  assert.equal(settings.preset, 'glass')
  assert.deepEqual(settings.light, PRESETS.glass.light)
  ok('a named preset saved with older values takes its current ones')
}

{
  const custom = { ...defaultSettings(), preset: 'custom', light: { ...PRESETS.tensorfold.light, accent: '#ff00aa' } } as Settings
  const { settings, changed } = migrateThemePreset(custom)
  assert.equal(changed, false)
  assert.equal(settings.light.accent, '#ff00aa')
  ok('custom palettes are kept')
}

console.log(`\n${passed} passed`)
