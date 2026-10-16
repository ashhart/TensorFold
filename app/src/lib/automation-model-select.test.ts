// Run: jiti src/lib/automation-model-select.test.ts
//
// Regression for the reported bug: in Automations, changing the MODEL does not "stick" —
// after reload it shows the previous/default model. ROOT CAUSE is NOT persistence (that
// round-trips fine, see electron/automations-persist.test.ts) but the editor's model
// <select>: its controlled value was rebuilt from raw draft fields independently of the
// option set produced by providerModelGroups(). Because that builder DROPS the null
// "default login" group when a provider has named accounts (and canonicalizes CLI model
// ids), the value could reference an option that was never rendered — so the browser
// silently fell back to its FIRST option (the default model), and that default would
// overwrite the pinned model on the next save.
//
// resolveModelSelect() binds the saved (providerId, accountId, model) to a REAL option,
// or surfaces a synthetic "pinned" option when none exists. These tests pin that contract.
import assert from 'node:assert/strict'
import type { Provider } from './types.ts'
import { modelChoiceValue, parseModelChoice, resolveModelSelect } from './automation-model-select.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

const optionValues = (s: ReturnType<typeof resolveModelSelect>) =>
  s.modelGroups.flatMap(({ provider, group }) => group.models.map((m) => modelChoiceValue(provider.id, group.accountId, m)))

// A real provider that exposes named accounts AND a model catalogue.
const httpProvider = (over: Partial<Provider> = {}): Provider => ({
  id: 'openai', name: 'OpenAI', kind: 'cloud', baseUrl: 'https://api.openai.com/v1', apiKey: 'sk',
  models: ['gpt-5.5', 'gpt-5.5-codex-high'],
  ...over,
}) as Provider

console.log('automation model <select> binding:')

// (1) Happy path: single-account provider, model in catalogue, accountId null. Must bind.
await (async () => {
  const providers = [httpProvider()]
  const s = resolveModelSelect(providers, { providerId: 'openai', providerAccountId: null, model: 'gpt-5.5-codex-high' })
  assert.ok(s.selectedModelValue, 'a value is bound')
  assert.ok(optionValues(s).includes(s.selectedModelValue), 'bound value matches a rendered option')
  assert.equal(s.orphanModelValue, '', 'no synthetic option needed')
  ok('single-account provider with a catalogue model binds to a real option')
})()

// (2) THE BUG: provider has named accounts but the automation is pinned to accountId=null.
//     providerModelGroups drops the null default group -> the naive value matched nothing
//     -> select showed the default. resolveModelSelect must fall back to the same provider's
//     group for that model so the value is rendered (not silently reset to default).
await (async () => {
  const providers = [httpProvider({ accounts: [{ id: 'acct-work', name: 'Work', enabled: true, models: ['gpt-5.5', 'gpt-5.5-codex-high'] }] as any })]
  const s = resolveModelSelect(providers, { providerId: 'openai', providerAccountId: null, model: 'gpt-5.5-codex-high' })
  assert.ok(optionValues(s).includes(s.selectedModelValue), 'pinned model binds to an existing option even though the null-default group was dropped')
  assert.equal(s.orphanModelValue, '', 'a real option exists, so no synthetic fallback')
  assert.ok(s.selectedGroup && s.selectedGroup.group.models.includes('gpt-5.5-codex-high'), 'resolved group actually lists the pinned model')
  // And the displayed model must be the pinned one — NOT the catalogue's first/default model.
  assert.ok(s.selectedModelValue.includes('gpt-5.5-codex-high'), 'bound value reflects the PINNED model, not the default')
  ok('named-account provider + accountId:null still surfaces the pinned model (the reported revert)')
})()

// (3) Orphaned pin: the saved model/account no longer maps to any option (e.g. legacy or
//     aliased id, or the account was deleted). It must be shown as a synthetic option so the
//     user sees the real pinned model rather than the default — preventing silent clobber.
await (async () => {
  const providers = [httpProvider()]
  const s = resolveModelSelect(providers, { providerId: 'openai', providerAccountId: null, model: 'gpt-legacy-removed' })
  assert.equal(s.selectedGroup, null, 'no real option represents the orphaned model')
  assert.ok(s.orphanModelValue, 'a synthetic option value is produced')
  assert.equal(s.orphanModelValue, s.selectedModelValue, 'the select binds to that synthetic value (not the default)')
  assert.ok(s.selectedModelValue.includes('gpt-legacy-removed'), 'synthetic value carries the real pinned model id')
  ok('orphaned pinned model is surfaced as a synthetic option (no silent revert to default)')
})()

// (4) End-to-end: pick an option -> serialize -> parse (what pickModel does) -> resolve back.
//     The chosen model must round-trip to the SAME option value, proving the editor would
//     persist and then re-display exactly what the user selected.
await (async () => {
  const providers = [httpProvider({ accounts: [{ id: 'acct-work', name: 'Work', enabled: true, models: ['gpt-5.5', 'gpt-5.5-codex-high'] }] as any })]
  const before = resolveModelSelect(providers, { providerId: 'openai', providerAccountId: 'acct-work', model: 'gpt-5.5' })
  // user opens the dropdown and clicks the codex-high option in the Work group:
  const clicked = optionValues(before).find((v) => v.includes('gpt-5.5-codex-high') && v.includes('acct-work'))!
  const picked = parseModelChoice(clicked)
  assert.ok(picked, 'option value parses back to fields')
  // editablePatch persists { providerId, providerAccountId, model }; reload + reopen resolves:
  const after = resolveModelSelect(providers, picked!)
  assert.equal(after.selectedModelValue, clicked, 'picked option round-trips to the same bound value after reload')
  assert.equal(after.orphanModelValue, '', 'round-tripped pick is a real option, never synthetic')
  ok('pick -> serialize -> parse -> reload resolves back to the exact same option')
})()

console.log(`\n${passed} passed`)
