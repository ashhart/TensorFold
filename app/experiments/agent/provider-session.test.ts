import assert from 'node:assert/strict'
import {
  providerSessionKind,
  isOwnedLoop,
  defaultCapabilities,
  type ProviderSession,
  type ProviderSessionKind,
  type ProviderSessionDeps,
  type ProviderSessionFactory,
} from './provider-session'
import type { StreamChunk } from '../../electron/agent/openai'
import type { Provider } from '../../electron/agent/types'

function providerWith(transport?: Provider['transport']): Pick<Provider, 'transport'> {
  return { transport }
}

// --- pure routing: every transport maps to the right session kind ---
assert.equal(providerSessionKind(providerWith('codex-cli')), 'delegated-codex')
assert.equal(providerSessionKind(providerWith('claude-cli')), 'delegated-claude')
assert.equal(providerSessionKind(providerWith('chat-completions')), 'owned-loop')
// undefined transport (legacy HTTP provider) and a null/undefined provider both
// fall through to the owned loop rather than throwing.
assert.equal(providerSessionKind(providerWith(undefined)), 'owned-loop')
assert.equal(providerSessionKind(undefined), 'owned-loop')
assert.equal(providerSessionKind(null), 'owned-loop')

// --- isOwnedLoop is the boolean shorthand of the router ---
assert.equal(isOwnedLoop(providerWith('chat-completions')), true)
assert.equal(isOwnedLoop(providerWith(undefined)), true)
assert.equal(isOwnedLoop(providerWith('codex-cli')), false)
assert.equal(isOwnedLoop(providerWith('claude-cli')), false)

// --- default capabilities per kind ---
assert.deepEqual(defaultCapabilities('owned-loop'), { sessionResume: false, midTurnInterrupt: true })
assert.deepEqual(defaultCapabilities('delegated-codex'), { sessionResume: true, midTurnInterrupt: true })
assert.deepEqual(defaultCapabilities('delegated-claude'), { sessionResume: true, midTurnInterrupt: false })

// --- the ProviderSession interface compiles + a DI fake satisfies it ---
// This proves the contract is implementable with NO real subprocess/network: the
// fake streams canned StreamChunks and exposes a resume cursor, exactly how the
// integrator's owned-loop / delegated impls will.
class FakeSession implements ProviderSession {
  readonly kind: ProviderSessionKind
  readonly capabilities = defaultCapabilities('owned-loop')
  private interrupted = false
  private cursor: unknown = undefined
  constructor(private readonly deps: ProviderSessionDeps) {
    this.kind = 'owned-loop'
  }
  startSession(): void {
    this.cursor = 'cursor-0'
  }
  async *sendTurn(prompt: string): AsyncGenerator<StreamChunk> {
    if (this.deps.stream) {
      for await (const ch of this.deps.stream({ prompt })) yield ch
      return
    }
    yield { type: 'content', text: prompt }
    yield { type: 'finish', reason: 'stop' }
  }
  interruptTurn(): void {
    this.interrupted = true
  }
  resumeCursor(): unknown {
    return this.cursor
  }
}

const factory: ProviderSessionFactory = (kind, deps) => {
  assert.equal(kind, 'owned-loop')
  return new FakeSession(deps)
}

async function exercise(): Promise<void> {
  const injected: StreamChunk[] = [
    { type: 'reasoning', text: 'thinking' },
    { type: 'content', text: 'hello' },
    { type: 'finish', reason: 'stop' },
  ]
  const deps: ProviderSessionDeps = {
    async *stream() {
      for (const ch of injected) yield ch
    },
  }
  const session = factory(providerSessionKind(providerWith(undefined)), deps)
  session.startSession()
  assert.equal(session.resumeCursor(), 'cursor-0')

  const got: StreamChunk[] = []
  for await (const ch of session.sendTurn('hi')) got.push(ch)
  assert.deepEqual(got, injected)

  // interrupt is best-effort and must not throw
  session.interruptTurn()

  // a session with no injected streamer still yields a valid turn
  const bare = factory('owned-loop', {})
  bare.startSession()
  const bareOut: StreamChunk[] = []
  for await (const ch of bare.sendTurn('echo')) bareOut.push(ch)
  assert.deepEqual(bareOut, [
    { type: 'content', text: 'echo' },
    { type: 'finish', reason: 'stop' },
  ])
}

exercise().then(() => {
  console.log('provider-session.test.ts: all assertions passed')
})
