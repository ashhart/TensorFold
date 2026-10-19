import assert from 'node:assert/strict'
import { engineLoadLimitOverride, isDeniedEngineEnvKey, normalizeEngineEnvOverrides, readEngineEnvOverrides } from '../../../shared/engine-advanced-env'

// Names are strict UPPER_SNAKE; values are plain bounded strings.
assert.deepEqual(normalizeEngineEnvOverrides({ GLM_LOAD_LIMIT_GB: '100' }), { GLM_LOAD_LIMIT_GB: '100' })
assert.deepEqual(normalizeEngineEnvOverrides({}), {})
for (const bad of ['glm_load_limit_gb', 'lower_case', '_LEADING', '1START', 'HAS SPACE', 'HAS-DASH', 'A'.repeat(65), '']) {
  assert.throws(() => normalizeEngineEnvOverrides({ [bad]: '1' }), /letters, digits and underscores|managed/, bad)
}
assert.throws(() => normalizeEngineEnvOverrides({ PATH: '/tmp' }), /managed by TensorFold/)
assert.throws(() => normalizeEngineEnvOverrides({ HF_ENDPOINT: 'https://evil.example' }), /managed by TensorFold/)
assert.throws(() => normalizeEngineEnvOverrides({ DYLD_INSERT_LIBRARIES: '/evil.dylib' }), /managed by TensorFold/)
assert.throws(() => normalizeEngineEnvOverrides({ NODE_OPTIONS: '--require /evil.js' }), /managed by TensorFold/)
assert.throws(() => normalizeEngineEnvOverrides({ TENSORFOLD_API_KEY: 'x' }), /managed by TensorFold/)
assert.throws(() => normalizeEngineEnvOverrides({ OPENAI_API_KEY: 'sk-x' }), /managed by TensorFold/)
assert.throws(() => normalizeEngineEnvOverrides({ SOME_TOKEN: 'x' }), /managed by TensorFold/)
assert.throws(() => normalizeEngineEnvOverrides({ X: '' }), /needs a value/)
assert.throws(() => normalizeEngineEnvOverrides({ X: 'x'.repeat(2049) }), /at most 2048 characters/)
assert.throws(() => normalizeEngineEnvOverrides({ X: 'a\0b' }), /at most 2048 characters/)
assert.throws(() => normalizeEngineEnvOverrides({ X: 42 }), /needs a value/)
assert.throws(() => normalizeEngineEnvOverrides(['X=1']), /set of environment variables/)
assert.throws(() => normalizeEngineEnvOverrides(Object.fromEntries(Array.from({ length: 51 }, (_, i) => [`VAR_${i}`, '1']))), /at most 50 variables/)
assert.equal(isDeniedEngineEnvKey('DYLD_LIBRARY_PATH'), true)
assert.equal(isDeniedEngineEnvKey('GLM_LOAD_LIMIT_GB'), false)

// A persisted map is filtered, never trusted: hand-edited junk drops out silently.
assert.deepEqual(readEngineEnvOverrides({ GLM_LOAD_LIMIT_GB: '100', PATH: '/tmp', lower: '1', X: '' }), { GLM_LOAD_LIMIT_GB: '100' })
assert.deepEqual(readEngineEnvOverrides(undefined), {})
assert.deepEqual(readEngineEnvOverrides('GLM_LOAD_LIMIT_GB=100'), {})

// The app-side memory floor yields to the engine's own load-limit variables.
assert.equal(engineLoadLimitOverride({}), null)
assert.equal(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '0' }), null)
assert.equal(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '-5' }), null)
assert.equal(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: 'lots' }), null)
assert.deepEqual(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '100' })?.bytes, 100 * 1024 ** 3)
assert.equal(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '80', FLASHNEXT_LOAD_LIMIT_GB: '120' })?.key, 'FLASHNEXT_LOAD_LIMIT_GB')
assert.equal(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '200', FLASHNEXT_LOAD_LIMIT_GB: '120' })?.key, 'GLM_LOAD_LIMIT_GB')

console.log('engine environment tests passed')
