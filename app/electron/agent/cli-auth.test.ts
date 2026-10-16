import assert from 'node:assert/strict'
import { cliAuthCommands, parseAuthStatus, modelReadiness } from './cli-auth'
import { cliAuthBinary } from './cli-auth-runtime'

// --- command specs: both CLIs are symmetric (status + login + logout) ---
const codex = cliAuthCommands('codex-cli')
assert.deepEqual(codex.statusArgs, ['login', 'status'], 'codex status = `codex login status`')
assert.deepEqual(codex.loginArgs, ['login'], 'codex login = `codex login` (OAuth)')
assert.deepEqual(codex.logoutArgs, ['logout'], 'codex logout = `codex logout`')

const claude = cliAuthCommands('claude-cli')
assert.deepEqual(claude.statusArgs, ['auth', 'status'], 'claude status = `claude auth status`')
assert.deepEqual(claude.loginArgs, ['auth', 'login'], 'claude login = `claude auth login` (OAuth)')
assert.deepEqual(claude.logoutArgs, ['auth', 'logout'], 'claude logout = `claude auth logout`')

// --- parseAuthStatus: Claude emits JSON ---
const cIn = parseAuthStatus('claude-cli', JSON.stringify({ loggedIn: true, email: 'user@example.com', subscriptionType: 'max' }), 0)
assert.equal(cIn.loggedIn, true, 'claude JSON loggedIn:true parsed')
assert.ok((cIn.detail || '').includes('user@example.com'), 'claude detail includes the account email')

const cOut = parseAuthStatus('claude-cli', JSON.stringify({ loggedIn: false }), 0)
assert.equal(cOut.loggedIn, false, 'claude JSON loggedIn:false parsed')

// --- parseAuthStatus: Codex emits plain text ---
const xIn = parseAuthStatus('codex-cli', 'Logged in using ChatGPT', 0)
assert.equal(xIn.loggedIn, true, 'codex "Logged in using ChatGPT" => logged in')
assert.ok((xIn.detail || '').toLowerCase().includes('chatgpt'), 'codex detail includes the method')

const xOut = parseAuthStatus('codex-cli', 'Not logged in', 1)
assert.equal(xOut.loggedIn, false, 'codex "Not logged in" => logged out')

// --- modelReadiness: the user-facing guard ---
// no provider at all -> point to Settings (your exact wording)
const none = modelReadiness({ provider: null })
assert.equal(none.ready, false)
assert.equal(none.reason, 'no-provider')
assert.ok(/settings/i.test(none.message) && /model/i.test(none.message), 'no-provider message points to Settings + model')

// CLI provider, not logged in -> "isn't authorized"
const unauthed = modelReadiness({ provider: { name: 'Claude', transport: 'claude-cli' } as any, cliBinaryPresent: true, cliLoggedIn: false })
assert.equal(unauthed.ready, false)
assert.equal(unauthed.reason, 'cli-unauthed')
assert.ok(/authori[sz]/i.test(unauthed.message) && /Claude/.test(unauthed.message), 'names the provider + not authorized')

// CLI provider, binary missing
const missing = modelReadiness({ provider: { name: 'Codex', transport: 'codex-cli' } as any, cliBinaryPresent: false })
assert.equal(missing.reason, 'cli-missing')

// CLI provider, logged in -> ready
const cliReady = modelReadiness({ provider: { name: 'Codex', transport: 'codex-cli' } as any, cliBinaryPresent: true, cliLoggedIn: true })
assert.equal(cliReady.ready, true)
assert.equal(cliReady.reason, 'ok')

// HTTP cloud provider missing key -> needs-key
const noKey = modelReadiness({ provider: { name: 'OpenAI', transport: 'chat-completions', baseUrl: 'https://api.openai.com/v1', kind: 'cloud' } as any, hasApiKey: false })
assert.equal(noKey.reason, 'needs-key')
assert.ok(/settings/i.test(noKey.message) && /model/i.test(noKey.message), 'needs-key shows generic configure-a-model guidance (does not blame a provider the user did not pick)')
assert.ok(!/openai/i.test(noKey.message), 'needs-key does not name the default provider')

// HTTP cloud provider with key -> ready
const withKey = modelReadiness({ provider: { name: 'OpenAI', transport: 'chat-completions', baseUrl: 'https://api.openai.com/v1', kind: 'cloud' } as any, hasApiKey: true })
assert.equal(withKey.ready, true)

// HTTP local provider (Ollama) needs no key -> ready
const local = modelReadiness({ provider: { name: 'Ollama', transport: 'chat-completions', baseUrl: 'http://localhost:11434/v1', kind: 'local', source: 'ollama' } as any, hasApiKey: false })
assert.equal(local.ready, true, 'local provider needs no key')

// HTTP provider with no baseUrl -> needs-endpoint (checked before key)
const noUrl = modelReadiness({ provider: { name: 'Custom', transport: 'chat-completions', baseUrl: '', kind: 'cloud' } as any, hasApiKey: true })
assert.equal(noUrl.reason, 'needs-endpoint')

// --- detection resolves the binary like EXECUTION, so an installed CLI is never reported missing ---
const prov = (transport: 'codex-cli' | 'claude-cli', cliPath?: string) =>
  ({ id: 'x', name: 'x', baseUrl: '', apiKey: '', models: [], kind: 'cloud', transport, cliPath } as any)
assert.equal(cliAuthBinary(prov('codex-cli', '/custom/codex')), '/custom/codex', 'an explicit configured cliPath wins')
assert.equal(
  cliAuthBinary(prov('claude-cli'), { dirs: ['/opt/homebrew/bin'], exists: (p) => p === '/opt/homebrew/bin/claude' }),
  '/opt/homebrew/bin/claude',
  'resolves an installed CLI to its ABSOLUTE path from a known dir (matches the agent runner) rather than relying on the stripped GUI PATH',
)
assert.equal(cliAuthBinary(prov('codex-cli'), { dirs: ['/nope'], exists: () => false }), 'codex', 'falls back to the bare name (startup-augmented PATH) only when not found anywhere')

// New-engine transports (VST-033) must not fall through to the HTTP needs-endpoint gate.
{
  const piMissing = modelReadiness({ provider: { name: 'Pi', transport: 'pi-cli', baseUrl: '' } as any, cliBinaryPresent: false })
  assert.equal(piMissing.ready, false)
  assert.equal(piMissing.reason, 'cli-missing', 'pi with no binary is cli-missing, not needs-endpoint')
  const piUnknown = modelReadiness({ provider: { name: 'Pi', transport: 'pi-cli', baseUrl: '' } as any })
  assert.equal(piUnknown.ready, true, 'unknown binary presence never blocks the composer (the doctor proves it separately)')
  const piUnsafeKey = modelReadiness({ provider: { name: 'Pi', transport: 'pi-cli', baseUrl: '', apiKey: 'generic-key' } as any })
  assert.equal(piUnsafeKey.ready, false, 'legacy generic-key Pi config is rejected before a run reaches argv handoff')
  assert.equal(piUnsafeKey.reason, 'cli-unauthed')
  assert.match(piUnsafeKey.message, /provider-specific credential.*Pi.*own auth|Pi.*provider-specific credential/i)
  const piEnvKey = modelReadiness({ provider: { name: 'Pi', transport: 'pi-cli', baseUrl: '', apiKey: 'legacy-key', cliEnv: { OPENAI_API_KEY: 'safe-env-key' } } as any })
  assert.equal(piEnvKey.ready, true, 'an explicit provider credential environment matches the runtime handoff policy')
  const oc = modelReadiness({ provider: { name: 'OpenCode', transport: 'opencode', baseUrl: '' } as any })
  assert.equal(oc.ready, true, 'opencode has no baseUrl/API-key prerequisite; liveness is the doctor\'s job')
  const kilo = modelReadiness({ provider: { name: 'Kilo', transport: 'kilo', baseUrl: '' } as any })
  assert.equal(kilo.ready, true)
  const cursorMissing = modelReadiness({ provider: { name: 'Cursor', transport: 'cursor-cli', baseUrl: '' } as any, cliBinaryPresent: false })
  assert.equal(cursorMissing.reason, 'cli-missing', 'cursor-agent gates on binary presence')
  const gemini = modelReadiness({ provider: { name: 'Gemini', transport: 'gemini-cli', baseUrl: '' } as any })
  assert.equal(gemini.ready, true, 'unknown binary presence never blocks; the doctor proves it')
}

console.log('cli-auth tests passed')
