// Standalone entry: run TensorFold as an ACP agent over stdio, so an ACP-aware editor
// (e.g. Zed) can spawn it directly and speak the Agent Client Protocol. Mirrors
// mcp-server-main.ts's companion-entry pattern for consistency. Register, e.g. in
// an editor's ACP server config:
//   node <repo>/node_modules/.bin/jiti <repo>/electron/agent/acp-server-main.ts <cwd>
// The default working directory for new sessions is argv[2], $VONTRA_ACP_CWD, or
// process.cwd() — a session/new request's own `cwd` always takes precedence.
//
// Packaging this as a real distributable binary (electron-builder, dist-electron
// build output wiring) is OUT OF SCOPE for this task — this file just makes the
// entry point exist and be correct; an operator runs it via jiti today, the same
// way mcp-server-main.ts is documented to be run.
//
// Provider/model selection (v1 simplification — see acp-server.ts's
// TensorFoldStudioTurnLoopDeps doc comment): this process is NOT the live Electron app, so
// it cannot read electron/main/state.ts's `store` (that module assumes a running
// BrowserWindow/app). Instead it reads ~/.vontra/settings.json directly with plain
// node:fs — the same on-disk file electron/store.ts's loadSettings() reads — and
// picks store.settings.activeProviderId/activeModel as the "sensible default"
// provider/model for every ACP session this process hosts, exactly as the task
// describes. If settings.json is missing, unreadable, or the active provider can't
// be resolved, it falls back to the same hardcoded default electron/store.ts's own
// defaultSettings() uses (OpenAI / gpt-4o) and prints a clear warning to stderr.
import { readFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { Readable, Writable } from 'node:stream'
import * as acp from '@agentclientprotocol/sdk'
import { decryptWithKey, resolveLocalSecretKeySync } from '../local-crypto'
import { connectTensorFoldStudioAcpAgent, createTensorFoldStudioAcpTurnRunner } from './acp-server'
import { configurePlanningGuard } from './planning-guard'
import type { Provider } from './types'

const SECRET_PREFIX = 'safe:v1:'

function vontraHomeDir(): string {
  return process.env.VONTRA_HOME?.trim() || path.join(os.homedir(), '.vontra')
}

// Minimal, hardcoded fallback mirroring electron/store.ts's defaultSettings() literal
// for the 'openai' entry — used only when settings.json can't be read or the active
// provider can't be resolved from it.
const FALLBACK_PROVIDER: Provider = {
  id: 'openai',
  name: 'OpenAI',
  baseUrl: 'https://api.openai.com/v1',
  apiKey: process.env.OPENAI_API_KEY ?? '',
  models: ['gpt-4o', 'gpt-4o-mini', 'o4-mini'],
  kind: 'cloud',
  source: 'openai',
}
const FALLBACK_MODEL = 'gpt-4o'

interface MinimalSettings {
  providers?: Provider[]
  activeProviderId?: string | null
  activeModel?: string | null
}

type ResolvedLocalSecretKeys = ReturnType<typeof resolveLocalSecretKeySync>

function revealStoredSecret(value: string | undefined, keys: ResolvedLocalSecretKeys): string | undefined {
  if (!value?.startsWith(SECRET_PREFIX)) return value
  const payload = value.slice(SECRET_PREFIX.length)
  try { return decryptWithKey(payload, keys.key) } catch { /* try the readable profile-key migration source */ }
  if (keys.profileFallbackKey) {
    try { return decryptWithKey(payload, keys.profileFallbackKey) } catch { /* sanitize below */ }
  }
  // Wrong/rotated key or a tampered blob: never pass the opaque safe:v1 value to a
  // provider process/HTTP transport where it could be mistaken for a credential.
  return ''
}

function revealProviderSecrets(provider: Provider, keys: ResolvedLocalSecretKeys): Provider {
  return {
    ...provider,
    apiKey: revealStoredSecret(provider.apiKey, keys) ?? '',
    serverPassword: revealStoredSecret(provider.serverPassword, keys),
    cliEnv: provider.cliEnv
      ? Object.fromEntries(Object.entries(provider.cliEnv).map(([name, value]) => [name, revealStoredSecret(value, keys) ?? '']))
      : provider.cliEnv,
    accounts: provider.accounts?.map((account) => ({
      ...account,
      apiKey: revealStoredSecret(account.apiKey, keys),
      env: account.env
        ? Object.fromEntries(Object.entries(account.env).map(([name, value]) => [name, revealStoredSecret(value, keys) ?? '']))
        : account.env,
    })),
  }
}

/** Resolves the default provider/model for ACP sessions from ~/.vontra/settings.json (v1 simplification — see file header). */
export function resolveDefaultProviderModel(): { provider: Provider; model: string } {
  // Validate the managed key policy before any fallback path. In particular,
  // VONTRA_REQUIRE_EXTERNAL_SECRET_KEY must abort this standalone process when the
  // independently-held key is absent; treating the failure as an unset API key
  // would silently weaken the deployment policy.
  const keys = resolveLocalSecretKeySync(path.join(vontraHomeDir(), '.secretkey'))
  try {
    const raw = readFileSync(path.join(vontraHomeDir(), 'settings.json'), 'utf8')
    const settings = JSON.parse(raw) as MinimalSettings
    const providerId = settings.activeProviderId
    const provider = settings.providers?.find((p) => p.id === providerId)
    const model = settings.activeModel
    if (provider && model) return { provider: revealProviderSecrets(provider, keys), model }
    console.error(`[acp-server] settings.json found but no matching provider for activeProviderId="${providerId ?? ''}"/activeModel="${model ?? ''}" — falling back to OpenAI/gpt-4o.`)
  } catch (e: any) {
    console.error(`[acp-server] could not read ${path.join(vontraHomeDir(), 'settings.json')} (${e?.message ?? e}) — falling back to OpenAI/gpt-4o.`)
  }
  return { provider: FALLBACK_PROVIDER, model: FALLBACK_MODEL }
}

/** Settings' confidential terms, read fresh per call: external providers plan only here too. */
function readConfidentialTerms(): string[] {
  try {
    const terms = (JSON.parse(readFileSync(path.join(vontraHomeDir(), 'settings.json'), 'utf8')) as { confidentialTerms?: unknown }).confidentialTerms
    return Array.isArray(terms) ? terms.filter((term): term is string => typeof term === 'string') : []
  } catch {
    return []
  }
}

function main() {
  const cwd = process.argv[2] || process.env.VONTRA_ACP_CWD || process.cwd()
  configurePlanningGuard({ confidentialTerms: readConfidentialTerms })
  const { provider, model } = resolveDefaultProviderModel()
  const { runTurn, forgetSession } = createTensorFoldStudioAcpTurnRunner({ provider, model })

  const output = Writable.toWeb(process.stdout) as WritableStream<Uint8Array>
  const input = Readable.toWeb(process.stdin) as ReadableStream<Uint8Array>
  const stream = acp.ndJsonStream(output, input)

  // Drop a session's in-memory Thread when the handlers evict it (idle TTL / session cap),
  // so the maps and accumulated apiMessages/timeline don't grow without bound over a long run.
  connectTensorFoldStudioAcpAgent({ runTurn, cwd, name: 'TensorFold', version: '0.1.0', onSessionClosed: forgetSession }, stream)
}

// Run only when invoked directly (not when imported by a test).
if (process.argv[1] && /acp-server-main\.(ts|js|mjs)$/.test(process.argv[1])) main()
