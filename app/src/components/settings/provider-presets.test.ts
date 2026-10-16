import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const shared = readFileSync(fileURLToPath(new URL('./shared.tsx', import.meta.url)), 'utf8')
const piStart = shared.indexOf("{ key: 'pi-cli'")
const piEnd = shared.indexOf("{ key: 'cursor-cli'", piStart)
assert.ok(piStart >= 0 && piEnd > piStart, 'Pi remains an addable provider preset')
const pi = shared.slice(piStart, piEnd)
assert.match(pi, /auth: 'cli-local'/, 'Pi is not presented as generic API-key auth')
assert.match(pi, /apiKey: ''/, 'the Pi seed never stores a generic raw key')
assert.match(pi, /Authenticate in Pi itself|provider-specific key environment/i)
assert.doesNotMatch(pi, /optional \(--api-key\)/, 'the unsafe argv guidance is gone')
assert.match(shared, /providerPresetNeedsApiKey[\s\S]*preset\.auth === 'api-key'/, 'only API-key presets require the generic password input')

const providerSettings = readFileSync(fileURLToPath(new URL('./useProviderSettings.ts', import.meta.url)), 'utf8')
assert.match(providerSettings, /apiKey: isCli \|\| !needsApiKey \? '' : providerSetup\.apiKey/, 'new cli-local providers persist an empty generic key')

const models = readFileSync(fileURLToPath(new URL('./ModelsSettings.tsx', import.meta.url)), 'utf8')
assert.match(models, /selectedCloudPreset\.auth === 'cli-local'/, 'model setup has a dedicated local-CLI auth branch')
assert.match(models, /Clear incompatible key/, 'legacy Pi raw keys have an explicit removal path')

const onboarding = readFileSync(fileURLToPath(new URL('../Onboarding.tsx', import.meta.url)), 'utf8')
assert.match(onboarding, /providerPresetNeedsApiKey/, 'onboarding no longer requires a key for cli-local presets')
assert.match(onboarding, /auth === 'cli-local'/, 'onboarding explains local CLI auth instead of rendering a password field')

console.log('Pi preset, Settings, onboarding, and raw-key migration UI are aligned')
