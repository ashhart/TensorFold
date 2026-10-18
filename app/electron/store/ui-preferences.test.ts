import { LANGUAGE_IDS } from '../../shared/languages'
import { normalizeUiPreferences } from '../../src/lib/preferences/model'
import assert from 'node:assert/strict'
import { normalizeStoredUiPreferences } from './ui-preferences'
const unsafe = normalizeStoredUiPreferences({ locale: '<script>', direction: 'javascript:bad', chatScale: Infinity, codeScale: 'var(--secret)', density: 'collapse', style: 'display:none', sendKey: 'execute' })
assert.deepEqual(unsafe, { locale: 'en', direction: 'auto', chatScale: 1, codeScale: 1, density: 'comfortable' })
assert.deepEqual(normalizeStoredUiPreferences({ locale: 'es', direction: 'rtl', chatScale: 1.3, codeScale: 1.5, density: 'compact' }), { locale: 'es', direction: 'rtl', chatScale: 1.3, codeScale: 1.5, density: 'compact' })
assert.deepEqual(normalizeStoredUiPreferences(null), unsafe)
console.log('Host preferences discard arbitrary CSS, unknown locales and nonfinite scales before persistence')

async function persistedProfiles() {
  const { mkdtemp, rm } = await import('node:fs/promises')
  const os = await import('node:os'), path = await import('node:path')
  const root = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-ui-profile-'))
  const original = process.env.VONTRA_HOME
  process.env.VONTRA_HOME = path.join(root, 'first')
  try {
    const { defaultSettings, saveSettings, loadSettings } = await import('../store')
    const preferences = { locale: 'es', direction: 'rtl', chatScale: 1.3, codeScale: 1.15, density: 'compact' } as const
    await saveSettings({ ...defaultSettings(), uiPreferences: preferences, modelHelperDefaults: { search: { providerId: 'local-example', accountId: null, model: 'example' } } })
    const loaded = await loadSettings()
    assert.deepEqual(loaded.uiPreferences, preferences)
    await saveSettings({ ...loaded, temperature: 0.25 })
    assert.deepEqual((await loadSettings()).uiPreferences, preferences, 'an unrelated settings save preserves independent scales and locale')
    process.env.VONTRA_HOME = path.join(root, 'second')
    assert.equal((await loadSettings()).uiPreferences?.locale, 'en', 'another profile does not inherit the first profile language or helper defaults')
    assert.deepEqual((await loadSettings()).modelHelperDefaults, {})
    await saveSettings({ ...defaultSettings(), uiPreferences: { ...preferences, locale: 'en' } })
    process.env.VONTRA_HOME = path.join(root, 'first')
    assert.deepEqual((await loadSettings()).uiPreferences, preferences)
    assert.deepEqual((await loadSettings()).modelHelperDefaults, { search: { providerId: 'local-example', accountId: null, model: 'example' } })
    for (const locale of LANGUAGE_IDS) {
      const next = { ...preferences, locale, replyLanguage: 'auto' as const, sendKey: 'modified-enter' as const }
      assert.deepEqual(normalizeStoredUiPreferences(next), normalizeUiPreferences(next), 'Host and renderer admit the same fixed locale and reply preference')
      await saveSettings({ ...loaded, uiPreferences: next })
      assert.deepEqual((await loadSettings()).uiPreferences, next, 'Each bundled language survives protected disk reload')
    }
    process.env.VONTRA_HOME = path.join(root, 'second')
    assert.equal((await loadSettings()).uiPreferences?.locale, 'en', 'Language changes remain in their owning profile')
    assert.notEqual((await loadSettings()).uiPreferences?.replyLanguage, 'auto')
    assert.equal((await loadSettings()).uiPreferences?.sendKey, undefined, 'Send-key preferences never carry into another synthetic profile')
    console.log('Actual settings disk roundtrip preserves locale/scales/helper defaults across unrelated saves and isolates two synthetic profiles')
  } finally { if (original === undefined) delete process.env.VONTRA_HOME; else process.env.VONTRA_HOME = original; await rm(root, { recursive: true, force: true }) }
}
void persistedProfiles().catch(error => { console.error(error); process.exitCode = 1 })
