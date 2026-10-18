// Uniform ProviderSession SPI. Owned-loop providers (OpenAI-compatible HTTP /
// local servers) and delegated-session providers (codex / claude app-server CLIs)
// both implement one contract, so runAgent can branch ONCE at the top on the
// session kind instead of threading transport special-cases through openai.ts.
//
// This file is intentionally mostly types + a single pure router. The concrete
// session impls are dependency-injected by the integrator (mirroring how
// workspace-git takes a GitRunner and ssh-connection takes a clientFactory):
// a ProviderSessionFactory is passed runtime collaborators (a streamer for the
// owned loop, a transport for the delegated CLIs) and returns a ProviderSession.

import type { Provider } from '../../electron/agent/types'
import type { StreamChunk } from '../../electron/agent/openai'

// Owned-loop = we drive the agent loop ourselves over an OpenAI-compatible
// stream. Delegated = an external app-server (codex / claude CLI) owns the loop
// and we relay turns to it.
export type ProviderSessionKind = 'owned-loop' | 'delegated-codex' | 'delegated-claude'

// What a session supports. Owned-loop sessions are stateless per turn (no native
// resume) but can interrupt mid-turn via AbortSignal. Delegated sessions keep a
// server-side cursor (resume) and may or may not interrupt mid-turn.
export interface ProviderSessionCapabilities {
  // The provider exposes a server-side cursor that can resume an existing thread.
  sessionResume: boolean
  // A turn can be interrupted while it is still streaming.
  midTurnInterrupt: boolean
}

// Inputs to open a session. cwd + approvalMode matter for the delegated CLIs
// (sandbox / plan), model/temperature/reasoning for the owned loop. resumeCursor
// reattaches to a prior delegated session when sessionResume is supported.
export interface SessionStartInput {
  provider: Provider
  model: string
  cwd?: string | null
  approvalMode?: string
  temperature?: number
  reasoningEffort?: string
  serviceTier?: string
  // Opaque cursor from a previous session's resumeCursor(), when resuming.
  resumeCursor?: unknown
  signal?: AbortSignal
}

// Per-turn options. The owned loop needs the running message history + tool
// schemas each turn; delegated sessions usually carry that server-side and only
// need the new prompt, but accept the same shape for uniformity.
export interface SendTurnOptions {
  // OpenAI-style message history (owned loop replays this each turn).
  messages?: unknown[]
  // Tool JSON schemas for the owned loop. Delegated CLIs manage their own tools.
  tools?: unknown[]
  cwd?: string | null
  approvalMode?: string
  signal?: AbortSignal
}

// The uniform contract. Both owned-loop and delegated impls satisfy this, so the
// caller never branches on transport again after picking the factory.
export interface ProviderSession {
  // Open / reattach the underlying session. Resolves once the session is ready.
  startSession(opts?: SessionStartInput): Promise<void> | void
  // Stream one turn. Yields the same StreamChunk shape openai.ts already emits.
  sendTurn(prompt: string, opts?: SendTurnOptions): AsyncGenerator<StreamChunk>
  // Stop the in-flight turn (best-effort; honored when midTurnInterrupt is true).
  interruptTurn(): void
  // Opaque resume token for the next startSession (undefined when not resumable).
  resumeCursor(): unknown
  readonly kind: ProviderSessionKind
  readonly capabilities: ProviderSessionCapabilities
}

// Collaborators the integrator injects so sessions are testable with no real
// subprocess / network. The owned loop calls `stream` (a streamChat-shaped fn);
// the delegated kinds call `transport` (a CLI app-server relay). Both are
// optional here because this module only routes + declares the contract.
export interface ProviderSessionDeps {
  // OpenAI-compatible streamer for owned-loop sessions (inject openai.streamChat).
  stream?: (params: unknown) => AsyncGenerator<StreamChunk>
  // App-server relay for delegated codex / claude sessions.
  transport?: unknown
}

// A factory builds a session for a given kind. The integrator supplies one impl
// per kind and selects via providerSessionKind. Kept as a type so this module
// stays pure (no concrete session logic lives here).
export type ProviderSessionFactory = (
  kind: ProviderSessionKind,
  deps: ProviderSessionDeps,
) => ProviderSession

// Pure routing: map a provider's transport to its session kind. This is the ONE
// decision runAgent makes at the top of a run. 'codex-cli' / 'claude-cli' are
// delegated app-server sessions; everything else (chat-completions, undefined)
// is the owned HTTP/local loop.
export function providerSessionKind(provider: Pick<Provider, 'transport'> | undefined | null): ProviderSessionKind {
  switch (provider?.transport) {
    case 'codex-cli':
      return 'delegated-codex'
    case 'claude-cli':
      return 'delegated-claude'
    default:
      return 'owned-loop'
  }
}

// Convenience: true when the provider's loop is owned by us (HTTP / local) rather
// than delegated to an external app-server. Lets callers gate owned-loop-only
// behavior (tool-schema injection, repeat guards) without re-deriving the kind.
export function isOwnedLoop(provider: Pick<Provider, 'transport'> | undefined | null): boolean {
  return providerSessionKind(provider) === 'owned-loop'
}

// Default capabilities per kind. Concrete impls may override, but this gives the
// router a single source of truth for what each transport can do.
export function defaultCapabilities(kind: ProviderSessionKind): ProviderSessionCapabilities {
  switch (kind) {
    case 'delegated-codex':
      return { sessionResume: true, midTurnInterrupt: true }
    case 'delegated-claude':
      return { sessionResume: true, midTurnInterrupt: false }
    case 'owned-loop':
    default:
      return { sessionResume: false, midTurnInterrupt: true }
  }
}
