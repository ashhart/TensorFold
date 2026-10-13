// Live mid-turn steering for delegated engine sessions (VST-033).
//
// The native loop applies queued user cues between its own steps. A delegated engine turn
// (codex app-server, claude stream-json, pi rpc) is one long streamChat call, so without
// help a cue waits for the WHOLE engine turn before the follow-up path picks it up. This
// module is the bridge: the run loop offers queued cues into an EngineSteeringChannel and
// the active engine adapter — when its protocol supports mid-turn input — forwards them
// into the live session.
//
// Wire facts behind the per-transport gate (verified against the installed binaries):
// - codex 0.142.5: `turn/steer` in the app-server protocol (generate-json-schema), with an
//   `expectedTurnId` precondition that fails when the turn is no longer active.
// - claude 2.1.170: `-p --input-format stream-json` accepts additional user messages on the
//   live stdin ("realtime streaming input").
// - pi 0.80.3: rpc mode's `steer` command (shipped docs/rpc.md).
// - opencode/kilo: no documented mid-turn input on the server protocol — the existing
//   drain-at-turn-end follow-up path keeps covering them.

import type { Provider } from './types'

export interface SteerInput {
  // Timeline item id of the queued cue, so acceptance can be reflected on the exact item.
  id: string
  text: string
}

type SteerConsumer = (input: SteerInput) => Promise<boolean> | boolean

export interface EngineSteeringChannel {
  // Engine side: claim the channel while the turn can accept input. `isReady` lets an
  // engine gate acceptance on protocol state (codex: an active turn id). Returns a
  // disconnect function; a second connect replaces the first (resume-fallback reruns).
  connect(consumer: SteerConsumer, isReady?: () => boolean): () => void
  // Loop side: true when an engine is connected and (per its own gate) ready to take input.
  readonly canAccept: boolean
  // Loop side: deliver one cue. False when no consumer is ready or the engine refused it.
  offer(input: SteerInput): Promise<boolean>
}

export function createSteeringChannel(): EngineSteeringChannel {
  let active: { consumer: SteerConsumer; isReady?: () => boolean } | null = null
  return {
    connect(consumer, isReady) {
      const entry = { consumer, isReady }
      active = entry
      return () => {
        if (active === entry) active = null
      }
    },
    get canAccept() {
      if (!active) return false
      try {
        return active.isReady ? active.isReady() : true
      } catch {
        return false
      }
    },
    async offer(input) {
      const entry = active
      if (!entry) return false
      if (entry.isReady) {
        try {
          if (!entry.isReady()) return false
        } catch {
          return false
        }
      }
      try {
        return (await entry.consumer(input)) === true
      } catch {
        return false
      }
    },
  }
}

// Which providers can steer a live turn. codex-cli qualifies through the app-server route,
// which the loop always takes for thread runs (it passes a threadId to key the session pool).
export function providerSupportsLiveSteering(provider: Pick<Provider, 'transport'> | null | undefined): boolean {
  if (!provider) return false
  return provider.transport === 'codex-cli' || provider.transport === 'claude-cli' || provider.transport === 'pi-cli'
}

export interface SteerCue {
  itemId: string
  text: string
}

export interface SteerCuePumpDeps {
  channel: EngineSteeringChannel
  signal: AbortSignal
  // The thread-runtime cue seam the native loop already uses.
  waitForCue: (timeoutMs: number, signal: AbortSignal) => Promise<void>
  drainCues: () => SteerCue[]
  // A cue the engine took mid-turn.
  onSteered: (cue: SteerCue) => void
  // A cue drained in the narrow window where the engine stopped accepting — surfaced as an
  // explicit failure state, never silently dropped.
  onRejected: (cue: SteerCue) => void
  pollMs?: number
}

function abortableSleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal.aborted) return resolve()
    const timer = setTimeout(done, ms)
    function done() {
      clearTimeout(timer)
      signal.removeEventListener('abort', done)
      resolve()
    }
    signal.addEventListener('abort', done, { once: true })
  })
}

// Runs beside a delegated engine turn. Cues stay QUEUED (visible, drainable by the loop's
// turn-end follow-up path) until the engine is actually ready to take live input; only then
// are they drained and offered.
export async function runSteerCuePump(deps: SteerCuePumpDeps): Promise<void> {
  const pollMs = deps.pollMs ?? 400
  while (!deps.signal.aborted) {
    await deps.waitForCue(pollMs, deps.signal)
    if (deps.signal.aborted) return
    if (!deps.channel.canAccept) {
      // A cue may already be queued, which makes waitForCue return immediately — back off
      // on a real timer so waiting for engine readiness never busy-spins.
      await abortableSleep(pollMs, deps.signal)
      continue
    }
    for (const cue of deps.drainCues()) {
      if (deps.signal.aborted) {
        // The turn ended while draining: hand the rest back as explicit rejections.
        deps.onRejected(cue)
        continue
      }
      const accepted = await deps.channel.offer({ id: cue.itemId, text: cue.text })
      if (accepted) deps.onSteered(cue)
      else deps.onRejected(cue)
    }
  }
}

// Homegrown-loop steering (VST-047). The OpenAI chat-completions endpoint is a
// sealed POST: you cannot inject into a running stream the way codex's turn/steer
// can. The feasible parity shape is "abort the stream + append the cue as a user
// message + re-request the turn from the top." This watcher sits beside the stream
// and, on the first cue arrival, fires the abort and hands the drained cues back
// to the loop so it can fold them into apiMessages before the re-request.
//
// Only ONE steer per watcher: the loop builds a fresh watcher per attempt. The
// loop caps steer-restarts per turn (MAX_STEER_RESTARTS) so a user mashing Guide
// can't loop forever; late cues beyond the cap fall through to the between-step
// drain path.
export interface HomegrownSteerWatcher {
  // True once a cue has arrived and the abort has fired. The loop reads this in
  // its catch block to distinguish a steer abort from a transient failure.
  readonly requested: boolean
  // The cues drained at the moment the abort fired. Empty until requested.
  readonly cues: SteerCue[]
  // Stop watching (turn completed normally, or the run was cancelled).
  stop(): void
}

export interface HomegrownSteerDeps {
  signal: AbortSignal
  waitForCue: (timeoutMs: number, signal: AbortSignal) => Promise<void>
  drainCues: () => SteerCue[]
  // Called once when the first cue arrives. The loop passes its streamAbort so
  // the abort actually stops the in-flight SSE reader.
  onSteer: (cues: SteerCue[]) => void
  pollMs?: number
}

export function createHomegrownSteerWatcher(deps: HomegrownSteerDeps): HomegrownSteerWatcher {
  const pollMs = deps.pollMs ?? 400
  let requested = false
  let stopped = false
  const collected: SteerCue[] = []
  // True once the loop's catch block has read .cues (the only legitimate reader).
  // Distinguishes "steer was consumed" from "steer was armed but the stream
  // completed normally before the abort could throw" — the cue-loss race.
  let claimed = false
  // Cues drained from the host queue and armed for a steer, but never claimed by
  // the catch block because the stream completed normally instead of throwing on
  // the abort. In that race the cues are already out of the host queue (so the
  // host's turn-end markUndrainedCues cannot recover them) AND the loop breaks
  // out of the attempt loop without reading .cues. loop.ts reads this field in
  // its attempt-end finally (after stop()) and re-injects the cues into
  // apiMessages so they reach the model on the next attempt/step.
  const unclaimed: SteerCue[] = []
  const state: HomegrownSteerWatcher = {
    get requested() { return requested },
    get cues() {
      claimed = true
      return collected
    },
    stop() {
      // Stream completed normally (or the run was cancelled). If a steer was
      // armed (requested) but the catch block never claimed the cues, splice them
      // out of `collected` into `unclaimed` so they are not silently lost.
      if (requested && !claimed && collected.length) {
        unclaimed.push(...collected.splice(0, collected.length))
      }
      stopped = true
    },
    // Not part of the HomegrownSteerWatcher interface (kept off the type so loop.ts
    // is unaffected); present on the runtime object for the recovery seam above.
    get unclaimedCues() { return unclaimed },
  } as HomegrownSteerWatcher
  void (async () => {
    while (!stopped && !deps.signal.aborted && !requested) {
      await deps.waitForCue(pollMs, deps.signal)
      if (stopped || deps.signal.aborted || requested) return
      // waitForCue resolves on a real cue OR on its poll timeout. drainCues is the
      // only way to distinguish the two (the host exposes no peek), and it is
      // destructive. We must NOT fire onSteer on a bare wake: that would abort a
      // healthy stream every pollMs (the "Provider stream was aborted" error
      // classifies as transient and would burn the retry budget on long turns).
      const cues = deps.drainCues()
      if (cues.length) {
        collected.push(...cues)
        requested = true
        deps.onSteer(cues)
        return
      }
    }
  })()
  return state
}
