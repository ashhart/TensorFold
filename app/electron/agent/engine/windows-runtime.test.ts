import assert from 'node:assert/strict'
import { resolveEngineBinaryPath } from './paths'
import { engineChildEnv } from './engine-env'

assert.equal(resolveEngineBinaryPath({
  packaged: true, platform: 'win32', resourcesPath: 'C:\\Program Files\\TensorFold\\resources', appRoot: '',
  env: { TENSORFOLD_STUDIO_ENGINE_BIN: 'C:\\other\\untrusted.exe' },
}), 'C:\\Program Files\\TensorFold\\resources\\engine\\bin\\tensorfold-native.exe')
const env = engineChildEnv({ SystemRoot: 'C:\\Windows', USERPROFILE: 'C:\\Users\\example', HF_HOME: 'C:\\models', OPENAI_API_KEY: 'synthetic', GH_TOKEN: 'synthetic' })
assert.equal(env.SystemRoot, 'C:\\Windows')
assert.equal(env.HF_HOME, 'C:\\models')
assert.equal(env.OPENAI_API_KEY, undefined)
assert.equal(env.GH_TOKEN, undefined)
console.log('Windows engine resolves its packaged executable and retains only approved OS/cache environment')
