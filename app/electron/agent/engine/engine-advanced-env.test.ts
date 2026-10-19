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
// The documented single-line rule holds on save: no embedded newlines.
assert.throws(() => normalizeEngineEnvOverrides({ X: 'one\ntwo' }), /single-line/)
assert.throws(() => normalizeEngineEnvOverrides({ X: 'one\r\ntwo' }), /single-line/)
assert.throws(() => normalizeEngineEnvOverrides({ X: 'trailing\n' }), /single-line/)
assert.equal(isDeniedEngineEnvKey('DYLD_LIBRARY_PATH'), true)
assert.equal(isDeniedEngineEnvKey('GLM_LOAD_LIMIT_GB'), false)

// A persisted map is filtered, never trusted: hand-edited junk drops out silently.
assert.deepEqual(readEngineEnvOverrides({ GLM_LOAD_LIMIT_GB: '100', PATH: '/tmp', lower: '1', X: '' }), { GLM_LOAD_LIMIT_GB: '100' })
assert.deepEqual(readEngineEnvOverrides(undefined), {})
assert.deepEqual(readEngineEnvOverrides('GLM_LOAD_LIMIT_GB=100'), {})
// The bound and the single-line rule hold on read too.
assert.equal(Object.keys(readEngineEnvOverrides(Object.fromEntries(Array.from({ length: 60 }, (_, i) => [`VAR_${i}`, '1'])))).length, 50)
assert.deepEqual(readEngineEnvOverrides({ GOOD: '1', CRASHED: 'one\ntwo' }), { GOOD: '1' })

// Native load-limit semantics (families/glm/load_plan.zig, native/cache_fit.zig, native/flashnext_batch_host.zig):
// decimal GB (1e9), std.fmt.parseFloat-compatible text, family isolation by checkpoint model_type,
// and a byte result capped at the machine ceiling instead of overflowing.
const RAM = 64 * 1024 ** 3
assert.equal(engineLoadLimitOverride({}, 'glm5_next', RAM), null)
assert.deepEqual(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '100' }, 'glm5_next', 256 * 1024 ** 3), { key: 'GLM_LOAD_LIMIT_GB', gb: 100, bytes: 100e9, capped: false }, 'decimal GB, never 1024**3')
assert.equal(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '60' }, 'glm_moe_dsa', RAM)?.bytes, 60e9)
assert.equal(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '60' }, 'qwen4_exp', RAM), null, 'the GLM variable never applies to the Flash Next family')
assert.equal(engineLoadLimitOverride({ FLASHNEXT_LOAD_LIMIT_GB: '55' }, 'glm5_next', RAM), null, 'the Flash Next variable never applies to the GLM family')
assert.deepEqual(engineLoadLimitOverride({ FLASHNEXT_LOAD_LIMIT_GB: '55', GLM_LOAD_LIMIT_GB: '60' }, 'qwen4_exp', RAM), { key: 'FLASHNEXT_LOAD_LIMIT_GB', gb: 55, bytes: 55e9, capped: false }, 'each family reads only its own variable')
assert.equal(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '60' }, null, RAM), null, 'an unknown or absent model_type keeps the default policy')
assert.equal(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '60' }, 'some_new_family', RAM), null)
assert.deepEqual(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: ' 60 ' }, 'glm5_next', RAM)?.gb, 60, 'native trims leading whitespace')
assert.deepEqual(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '60.5' }, 'glm5_next', RAM)?.bytes, 60.5e9)
assert.deepEqual(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '.5e2' }, 'glm5_next', RAM)?.bytes, 50e9, 'the forms std.fmt.parseFloat reads still work')
assert.deepEqual(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '1_000' }, 'glm5_next', 2048 * 1024 ** 3)?.bytes, 1000e9, 'underscore digit separators match parseFloat')
assert.equal(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '1_0x0' }, 'glm5_next', RAM), null, 'an underscore outside digits is junk')
for (const bad of ['0', '-5', 'lots', '', '0x64', 'Infinity', 'NaN', ' ', '+-1', '12GB']) {
  assert.equal(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: bad }, 'glm5_next', RAM), null, bad)
}
assert.equal(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '60' }, 'glm5_next', 60e9)?.capped, true, 'reaching the ceiling is reported, not silently uncapped')
assert.deepEqual(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '1e308' }, 'glm5_next', RAM), { key: 'GLM_LOAD_LIMIT_GB', gb: 1e308, bytes: RAM, capped: true }, 'an absurd request is capped at RAM, never Infinity')
assert.equal(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '60' }, 'glm5_next', 0), null)

console.log('engine environment tests passed')
