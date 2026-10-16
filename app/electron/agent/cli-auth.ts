import type { CliAuthStatus, CliAuthTransport, Provider, Readiness } from './types'
import { piCredentialHandoffError } from './cli-credential-policy'

// Codex and Claude CLIs both expose a symmetric status/login/logout surface;
// they just live under different subcommand namespaces:
//   codex login status | codex login | codex logout
//   claude auth status  | claude auth login | claude auth logout
// `codex login` / `claude auth login` are the OAuth (subscription) flows;
// API-key sign-in is `codex login --with-api-key` / ANTHROPIC_API_KEY for Claude.
export interface CliAuthCommands {
  statusArgs: string[]
  loginArgs: string[]
  logoutArgs: string[]
}

export function cliAuthCommands(transport: CliAuthTransport): CliAuthCommands {
  if (transport === 'codex-cli') {
    return { statusArgs: ['login', 'status'], loginArgs: ['login'], logoutArgs: ['logout'] }
  }
  return { statusArgs: ['auth', 'status'], loginArgs: ['auth', 'login'], logoutArgs: ['auth', 'logout'] }
}

// Claude prints JSON from `claude auth status`; Codex prints a line of text.
// Parse defensively so transient format drift degrades to a sensible guess.
export function parseAuthStatus(transport: CliAuthTransport, stdout: string, exitCode = 0): CliAuthStatus {
  const text = (stdout || '').trim()

  if (transport === 'claude-cli') {
    try {
      const j = JSON.parse(text) as { loggedIn?: unknown; email?: unknown; subscriptionType?: unknown }
      const detail = typeof j.email === 'string' && j.email
        ? `${j.email}${typeof j.subscriptionType === 'string' && j.subscriptionType ? ` (${j.subscriptionType})` : ''}`
        : undefined
      return { loggedIn: !!j.loggedIn, detail }
    } catch {
      // not JSON — fall through to text heuristics
    }
    if (/not logged in|logged out|no (?:active )?(?:credential|session)/i.test(text)) return { loggedIn: false }
    if (/logged in|authenticated/i.test(text)) return { loggedIn: true }
    return { loggedIn: exitCode === 0 }
  }

  // codex: "Logged in using ChatGPT" / "Not logged in"
  if (/not logged in|logged out/i.test(text)) return { loggedIn: false }
  const m = text.match(/logged in(?: using)?\s+(.+)/i)
  if (m) return { loggedIn: true, detail: m[1].trim() }
  if (/logged in/i.test(text)) return { loggedIn: true }
  return { loggedIn: exitCode === 0 && text.length > 0 }
}

export interface ReadinessInput {
  provider?: Provider | null
  hasApiKey?: boolean // resolved from the provider or the active account
  cliBinaryPresent?: boolean // CLI providers: is the binary installed?
  cliLoggedIn?: boolean // CLI providers: from parseAuthStatus()
}

const CONNECT_A_MODEL = 'Please go to Settings and connect a cloud or local model.'
const LOCAL_SOURCES = ['ollama', 'lmstudio', 'omlx']

function httpNeedsKey(provider: Provider): boolean {
  if (provider.kind === 'local') return false
  if (provider.source && LOCAL_SOURCES.includes(provider.source)) return false
  return true
}

// The single point-of-use guard: is there a usable model for this provider?
// Pure so the renderer (composer/room guard) and main process can share it.
export function modelReadiness(input: ReadinessInput): Readiness {
  const provider = input.provider
  if (!provider) return { ready: false, reason: 'no-provider', message: CONNECT_A_MODEL }

  const name = provider.name || 'This provider'

  if (provider.transport === 'codex-cli' || provider.transport === 'claude-cli') {
    if (input.cliBinaryPresent === false) {
      return { ready: false, reason: 'cli-missing', message: `${name} CLI isn't installed — open Settings to set its path.` }
    }
    if (input.cliLoggedIn === false) {
      return { ready: false, reason: 'cli-unauthed', message: `${name} isn't authorized — open Settings to connect.` }
    }
    return { ready: true, reason: 'ok', message: '' }
  }

  // Cursor Agent and Gemini CLI are local CLIs with key/env auth like pi: binary presence
  // is the only hard prerequisite the composer gate checks.
  if (provider.transport === 'cursor-cli' || provider.transport === 'gemini-cli') {
    if (input.cliBinaryPresent === false) {
      return { ready: false, reason: 'cli-missing', message: `${name} CLI isn't installed — open Settings to set its path.` }
    }
    return { ready: true, reason: 'ok', message: '' }
  }

  // Pi is a local CLI with key/env auth (no OAuth login step): the binary is the only
  // hard prerequisite. Unknown presence (undefined) doesn't block — the readiness doctor's
  // CLI inventory reports installed/missing with proof separately.
  if (provider.transport === 'pi-cli') {
    if (input.cliBinaryPresent === false) {
      return { ready: false, reason: 'cli-missing', message: `${name} CLI isn't installed — open Settings to set its path.` }
    }
    const credentialError = piCredentialHandoffError(provider)
    if (credentialError) return { ready: false, reason: 'cli-unauthed', message: credentialError }
    return { ready: true, reason: 'ok', message: '' }
  }

  // OpenCode-protocol servers (opencode/kilo) default to localhost:4096 and gate their own
  // auth via the optional server password — there is no baseUrl/API-key prerequisite to
  // check here. Live reachability is the readiness doctor's job, not the composer gate's.
  if (provider.transport === 'opencode' || provider.transport === 'kilo') {
    return { ready: true, reason: 'ok', message: '' }
  }

  // chat-completions (HTTP / OpenAI-compatible). Keep the guidance generic so a
  // fresh install doesn't blame a default provider the user never chose
  // (e.g. "OpenAI needs an API key") — just point them at Settings.
  if (!provider.baseUrl?.trim() || (httpNeedsKey(provider) && !input.hasApiKey)) {
    return { ready: false, reason: provider.baseUrl?.trim() ? 'needs-key' : 'needs-endpoint', message: CONNECT_A_MODEL }
  }
  return { ready: true, reason: 'ok', message: '' }
}
