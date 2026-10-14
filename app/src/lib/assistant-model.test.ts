import assert from 'node:assert/strict'
import { resolveAssistantModel, hasAssistantModelOverride, resolveSurfaceModel, hasSurfaceModelOverride, settingsForSurfaceModelPick } from './assistant-model.ts'
import { surfaceModelOverride } from '../../electron/agent/provider-accounts.ts'

const base: any = {
  activeProviderId: 'openai', activeProviderAccountId: 'acc1', activeModel: 'gpt-4o',
  assistantProviderId: null, assistantProviderAccountId: null, assistantModel: null,
  codeProviderId: null, codeProviderAccountId: null, codeModel: null,
}

// unset → falls back to the globally-active model
assert.deepEqual(resolveAssistantModel(base), { providerId: 'openai', providerAccountId: 'acc1', model: 'gpt-4o' })
assert.equal(hasAssistantModelOverride(base), false)

// set → uses the dedicated assistant model (any provider, incl. local)
const overridden = { ...base, assistantProviderId: 'local', assistantProviderAccountId: null, assistantModel: 'llama3.1:8b' }
assert.deepEqual(resolveAssistantModel(overridden), { providerId: 'local', providerAccountId: null, model: 'llama3.1:8b' })
assert.equal(hasAssistantModelOverride(overridden), true)

// partial (provider set but no model) → not a valid override, falls back
const partial = { ...base, assistantProviderId: 'local', assistantModel: null }
assert.deepEqual(resolveAssistantModel(partial), { providerId: 'openai', providerAccountId: 'acc1', model: 'gpt-4o' })
assert.equal(hasAssistantModelOverride(partial), false)

// Per-surface defaults (VST-015): the code surface has its own trio, independent of chat.
const codeOnly = { ...base, codeProviderId: 'anthropic', codeProviderAccountId: 'work', codeModel: 'claude-sonnet-5' }
const pickedLocal = { providerId: 'tensorfold', providerAccountId: null, model: 'local-model' }
const pickedSettings = settingsForSurfaceModelPick(codeOnly, 'code', pickedLocal)
assert.deepEqual(resolveSurfaceModel(pickedSettings, 'code'), pickedLocal, 'composer selection replaces a pinned code route')
assert.deepEqual(resolveSurfaceModel(pickedSettings, 'chat'), resolveSurfaceModel(codeOnly, 'chat'), 'changing a pinned code route preserves the chat default')
assert.deepEqual(resolveSurfaceModel(codeOnly, 'code'), { providerId: 'anthropic', providerAccountId: 'work', model: 'claude-sonnet-5' })
assert.deepEqual(resolveSurfaceModel(codeOnly, 'chat'), { providerId: 'openai', providerAccountId: 'acc1', model: 'gpt-4o' }, 'a code override never leaks into chat')
assert.equal(hasSurfaceModelOverride(codeOnly, 'code'), true)
assert.equal(hasSurfaceModelOverride(codeOnly, 'chat'), false)

// A partial code override (model without provider) is not an override.
const codePartial = { ...base, codeModel: 'claude-sonnet-5' }
assert.deepEqual(resolveSurfaceModel(codePartial, 'code'), { providerId: 'openai', providerAccountId: 'acc1', model: 'gpt-4o' })

// resolveAssistantModel stays an alias for the chat surface.
assert.deepEqual(resolveAssistantModel(overridden), resolveSurfaceModel({ ...overridden }, 'chat'))

// Cross-boundary parity: the main process resolves surface defaults with its own
// surfaceModelOverride (provider-accounts.ts) because the renderer resolver cannot be
// imported across the boundary. Pin both to identical precedence across a matrix:
// resolveSurfaceModel(s, surface) must equal (override ?? active trio).
const matrix = [
  base,
  overridden,
  partial,
  codeOnly,
  codePartial,
  { ...base, assistantProviderId: 'a', assistantProviderAccountId: 'x', assistantModel: 'm1', codeProviderId: 'b', codeProviderAccountId: null, codeModel: 'm2' },
  { ...base, activeProviderAccountId: null },
]
for (const settings of matrix) {
  for (const surface of ['chat', 'code'] as const) {
    const override = surfaceModelOverride(settings, surface)
    const expected = override
      ? { providerId: override.providerId, providerAccountId: override.providerAccountId, model: override.model }
      : { providerId: settings.activeProviderId, providerAccountId: settings.activeProviderAccountId, model: settings.activeModel }
    assert.deepEqual(resolveSurfaceModel(settings, surface), expected, `renderer/main drift for ${surface} on ${JSON.stringify(settings)}`)
    assert.equal(!!override, hasSurfaceModelOverride(settings, surface), 'override-presence drift')
  }
}

console.log('assistant-model tests passed (incl. surface defaults + main/renderer parity)')
