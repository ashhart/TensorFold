// Run: npx tsx electron/agent/ui-smoke-recipes.test.ts
import assert from 'node:assert/strict'
import { UI_SMOKE_RECIPES, uiSmokeRecipeCapabilities } from './ui-smoke-recipes'

assert.deepEqual(UI_SMOKE_RECIPES.map((recipe) => recipe.id), [
  'ui-smoke-localhost',
  'ui-smoke-interaction',
  'ui-smoke-responsive',
])

for (const recipe of UI_SMOKE_RECIPES) {
  assert.ok(recipe.name.startsWith('$ui-smoke-'))
  assert.ok(recipe.requiredToolNames.includes('run_command'))
  assert.ok(recipe.requiredToolNames.includes('browser_automation'))
  assert.ok(recipe.requiredEvidence.some((item) => /screenshot|snapshot|console|network/i.test(item)))
  assert.ok(recipe.steps.length >= 3)
}

const capabilities = uiSmokeRecipeCapabilities()
const local = capabilities.find((capability) => capability.id === 'skill:ui-smoke-localhost')
assert.ok(local)
assert.equal(local.source, 'skill')
assert.equal(local.lane, 'browser')
assert.equal(local.approvalRequired, true)
assert.ok(local.toolNames.includes('run_command'))
assert.ok(local.toolNames.includes('browser_automation'))
assert.equal(local.metadata?.recipeKind, 'ui_smoke')
assert.deepEqual(local.metadata?.requiredToolNames, ['run_command', 'browser_automation'])

console.log('ui smoke recipe tests passed')
