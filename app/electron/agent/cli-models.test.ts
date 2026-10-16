import assert from 'node:assert/strict'
import { CLAUDE_CLI_MODELS, CODEX_CLI_MODELS, OPENCODE_MODELS, KILO_MODELS, defaultCliModels, isLegacyCliModelList, normalizeLegacyCliModelList } from './cli-models.ts'

// OpenCode + Kilo (VST-033) default model lists route by transport.
{
  assert.deepEqual(defaultCliModels('opencode'), OPENCODE_MODELS)
  assert.deepEqual(defaultCliModels('kilo'), KILO_MODELS, 'Kilo (OpenCode fork) has its own default list')
  assert.notDeepEqual(OPENCODE_MODELS, KILO_MODELS, 'the two engines are distinct providers')
  assert.deepEqual(defaultCliModels('chat-completions'), [], 'HTTP providers have no CLI default set')
}

// Regression (MIGRATE-1/2): a user-curated subset made entirely of CURRENT default models must NOT
// be treated as legacy, or normalizeLegacyCliModelList resets it to the full default on EVERY load.
{
  // ['opus','sonnet'] are both current Claude defaults — a valid curation, not a legacy list.
  assert.equal(isLegacyCliModelList('claude-cli', ['opus', 'sonnet']), false, 'current-default subset is not legacy')
  assert.deepEqual(normalizeLegacyCliModelList('claude-cli', ['opus', 'sonnet']), ['opus', 'sonnet'], 'curated subset survives normalization')

  // A single current-default model (a valid account override) is kept, not expanded.
  assert.equal(isLegacyCliModelList('claude-cli', ['opus']), false, 'single current model is a valid override')
  assert.deepEqual(normalizeLegacyCliModelList('claude-cli', ['opus']), ['opus'], 'single-model account override survives')
}

// Still migrates a genuinely OLD default set (one that EXACTLY equals a legacy default) to the new
// defaults — the codex sentinel and the full old Claude default both expand.
{
  assert.ok(normalizeLegacyCliModelList('codex-cli', ['codex-cli-default']).includes('gpt-5.5'), 'codex sentinel = old default → expands')
  const oldClaude = ['claude-code-default', 'ultracode', 'opus-4.8', 'sonnet', 'opus']
  assert.equal(isLegacyCliModelList('claude-cli', oldClaude), true, 'the exact old Claude default set is legacy')
  assert.deepEqual(normalizeLegacyCliModelList('claude-cli', oldClaude), CLAUDE_CLI_MODELS, 'old default set migrates to current defaults')
  // A PARTIAL list (not the full old-default set) is a user choice and is NOT reset, even if it
  // contains an obsolete alias.
  assert.equal(isLegacyCliModelList('claude-cli', ['ultracode']), false, 'a partial list is not the full old default → kept')
}

// An empty list still seeds the defaults (a CLI provider needs at least its default model).
{
  assert.equal(isLegacyCliModelList('claude-cli', []), true, 'empty list seeds defaults')
  assert.deepEqual(normalizeLegacyCliModelList('claude-cli', []), CLAUDE_CLI_MODELS, 'empty → defaults')
  assert.deepEqual(normalizeLegacyCliModelList('codex-cli', []), CODEX_CLI_MODELS, 'empty → defaults (codex)')
}

// A custom model the user added is never dropped (not in any legacy set).
{
  assert.equal(isLegacyCliModelList('claude-cli', ['opus', 'my-custom-claude']), false, 'a custom model keeps the list')
  assert.deepEqual(normalizeLegacyCliModelList('claude-cli', ['opus', 'my-custom-claude']), ['opus', 'my-custom-claude'], 'custom model survives')
}

console.log('cli-models migration tests passed')
