import assert from 'node:assert/strict'
import { assertDeclaredEngineEnv, ENGINE_ENV_PRESETS, engineLoadLimitOverride, isDeclaredEngineEnvKey, isDeniedEngineEnvKey, normalizeEngineEnvOverrides, PINNED_ENGINE_ENV_KEYS, readEngineEnvOverrides } from '../../../shared/engine-advanced-env'
import { pinnedEngineEnv } from './engine-env'

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
// decimal GB (1e9), text accepted exactly by Zig 0.17's std.fmt.parseFloat (decimal and hex
// floats, underscore separators only between two digits), family isolation by checkpoint
// model_type, and a byte result capped at the machine ceiling instead of overflowing.
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
// Hex floats and the underscore rules agree with the native parser (verified against Zig 0.17):
assert.deepEqual(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '0x64' }, 'glm5_next', 256 * 1024 ** 3)?.bytes, 100e9, 'hex floats parse exactly like the native parseFloat: 0x64 is 100')
assert.deepEqual(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '0x1p6' }, 'glm5_next', RAM)?.bytes, 64e9, 'a hex mantissa with a p-exponent parses as the native one does')
assert.equal(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '100__0' }, 'glm5_next', RAM), null, 'double underscores are refused, as the native parser refuses them')
assert.equal(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '1_' }, 'glm5_next', RAM), null, 'a trailing underscore is refused, as the native parser refuses it')
assert.equal(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '1_.5' }, 'glm5_next', RAM), null, 'an underscore beside the point is refused, as the native parser refuses it')
for (const bad of ['0', '-5', 'lots', '', 'Infinity', 'NaN', ' ', '+-1', '12GB']) {
  assert.equal(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: bad }, 'glm5_next', RAM), null, bad)
}
assert.equal(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '60' }, 'glm5_next', 60e9)?.capped, true, 'reaching the ceiling is reported, not silently uncapped')
assert.deepEqual(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '1e308' }, 'glm5_next', RAM), { key: 'GLM_LOAD_LIMIT_GB', gb: 1e308, bytes: RAM, capped: true }, 'an absurd request is capped at RAM, never Infinity')
assert.equal(engineLoadLimitOverride({ GLM_LOAD_LIMIT_GB: '60' }, 'glm5_next', 0), null)

console.log('engine environment tests passed')

// The names the app pins for every engine process are refused here and scrubbed at the launcher,
// from one list, so the page can never offer a key the spawn environment owns.
for (const key of PINNED_ENGINE_ENV_KEYS) {
  assert.equal(isDeniedEngineEnvKey(key), true, key)
  assert.throws(() => normalizeEngineEnvOverrides({ [key]: 'x' }), /managed by TensorFold/, key)
  assert.equal(readEngineEnvOverrides({ [key]: 'x' })[key], undefined, `${key} drops out of a hand-edited file`)
}
const pinned = pinnedEngineEnv({ ...process.env, HF_ENDPOINT: 'https://evil.example', TENSORFOLD_API_KEY: 'sk-x', TENSORFOLD_NO_UPDATE_CHECK: '0', HF_TOKEN: 'hf-secret', KEEP: '1' }, 'https://huggingface.co')
assert.equal(pinned.HF_ENDPOINT, 'https://huggingface.co', 'the origin that authorized the checks wins')
assert.equal(pinned.TENSORFOLD_API_KEY, undefined, 'an inherited key never reaches the engine')
assert.equal(pinned.TENSORFOLD_NO_UPDATE_CHECK, undefined)
assert.equal(pinned.HF_TOKEN, undefined, "the app's own Hub token is never handed down")
assert.equal(pinned.KEEP, '1', 'ordinary variables are untouched')

// What the engine declares it honours is the authority on a name, not a hand-written list.
// This is the env list of a Metal build (zig/src/server/cli.zig).
const DECLARED = ['TENSORFOLD_API_KEY', 'TENSORFOLD_NO_LIVE', 'TENSORFOLD_SEED_SALT', 'TENSORFOLD_REQUEST_LOG', 'TENSORFOLD_NO_UPDATE_CHECK', 'HF_HOME', 'HF_HUB_CACHE', 'HF_HUB_OFFLINE', 'BRAVE_SEARCH_API_KEY']
for (const key of DECLARED) assert.equal(isDeclaredEngineEnvKey(DECLARED, key), true, key)
for (const preset of ENGINE_ENV_PRESETS) assert.equal(isDeclaredEngineEnvKey(DECLARED, preset.key), true, `${preset.key} is offered but not declared`)
assert.equal(isDeclaredEngineEnvKey(DECLARED, 'TENSORFOLD_CUDA_KERNELS'), false, 'a CUDA-only variable is not offered on a Metal build')
assert.equal(isDeclaredEngineEnvKey(DECLARED, 'GLM_LOAD_LIMIT_GB'), true, 'the load limit is read by the native plan, not declared as an env variable')
assert.equal(isDeclaredEngineEnvKey(null, 'ANY_NAME_AT_ALL'), true, 'an engine that could not be asked leaves the deny list as the boundary')
assert.doesNotThrow(() => assertDeclaredEngineEnv({ HF_HUB_OFFLINE: '1', GLM_LOAD_LIMIT_GB: '100' }, DECLARED))
assert.doesNotThrow(() => assertDeclaredEngineEnv({ TENSORFOLD_CUDA_KERNELS: '1' }, null))
assert.throws(() => assertDeclaredEngineEnv({ HF_HUB_OFFLINE: '1', TENSORFOLD_CUDA_KERNELS: '1' }, DECLARED), /TENSORFOLD_CUDA_KERNELS is not an environment variable this engine build honours/)
// Every declared name the app has a preset for is one it can actually set.
for (const key of ['HF_HOME', 'HF_HUB_OFFLINE', 'HF_HUB_CACHE', 'TENSORFOLD_REQUEST_LOG', 'TENSORFOLD_NO_LIVE', 'TENSORFOLD_SEED_SALT']) {
  assert.equal(isDeclaredEngineEnvKey(DECLARED, key), true, key)
}

// A chip carries no sample value, and no preset suggests a shared temporary directory for
// request bodies; the engine would create the file owner-only, but /tmp is still the wrong advice.
for (const preset of ENGINE_ENV_PRESETS) assert.deepEqual(Object.keys(preset).sort(), ['desc', 'key'], preset.key)
for (const preset of ENGINE_ENV_PRESETS) {
  assert.equal(/\/tmp\b|\/var\/tmp\b/.test(JSON.stringify(preset)), false, `${preset.key} names a shared temporary directory`)
}
assert.equal(ENGINE_ENV_PRESETS.length, 7)
assert.deepEqual(ENGINE_ENV_PRESETS.map((preset) => preset.key), ['GLM_LOAD_LIMIT_GB', 'FLASHNEXT_LOAD_LIMIT_GB', 'HF_HUB_OFFLINE', 'HF_HOME', 'TENSORFOLD_REQUEST_LOG', 'TENSORFOLD_NO_LIVE', 'TENSORFOLD_SEED_SALT'])
