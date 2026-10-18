import { runRetryVerdict } from '../../../shared/run-recovery'
import type { RunDeliveryState, RunPauseReason, RunRecoverySnapshot } from '../../../shared/run-recovery'

export class RunRecoveryTracker {
  private pending = new Map<string, boolean>()
  private completedTools = 0
  private completedMutations = 0
  private uncertainEffects = false
  private stopping = false
  private delivery: RunDeliveryState = 'not_sent'
  private reason?: RunPauseReason
  private message = 'Run active.'
  constructor(private runId: string, private publish?: (snapshot: RunRecoverySnapshot) => void, private now = Date.now) {}
  start(): void { this.emit('running') }
  begin(id: string, mutating = false): void {
    if (this.stopping) throw new Error('Cannot dispatch more work after Stop was requested.')
    if (this.pending.has(id)) throw new Error('Run operation identity is already active.')
    this.pending.set(id, mutating)
    this.emit('running')
  }
  markMutating(id: string): void { if (this.pending.has(id)) this.pending.set(id, true); this.emit(this.stopping ? 'stopping' : 'running') }
  toolUncertain(): void { this.uncertainEffects = true }
  end(id: string): void { this.pending.delete(id); this.emit(this.stopping ? 'stopping' : 'running') }
  requestStarted(): void { this.delivery = 'unknown'; this.emit(this.stopping ? 'stopping' : 'running') }
  requestOutput(): void { this.delivery = 'partial' }
  requestComplete(): void { this.delivery = 'complete' }
  requestFailed(delivery: RunDeliveryState): void {
    this.delivery = delivery
    if (delivery === 'partial' || delivery === 'unknown' || delivery === 'accepted') {
      this.reason = delivery === 'partial' ? 'partial_response' : 'acceptance_unknown'
      this.message = delivery === 'partial' ? 'The connection ended after partial output; the original response is preserved and was not replayed.' : 'The connection ended without proof that the provider refused the request; it was not replayed.'
    }
  }
  toolCompleted(mutating: boolean, ok: boolean, uncertainFailure = mutating && !ok): void {
    this.completedTools++
    if (mutating && ok) this.completedMutations++
    if (uncertainFailure) this.uncertainEffects = true
  }
  stopRequested(): void {
    if (this.stopping) return
    this.stopping = true
    this.reason ??= 'user_stop'
    this.message = 'Stop requested; waiting for already-started work to settle.'
    this.emit('stop_requested')
    this.emit('stopping')
  }
  pause(reason: RunPauseReason, message: string): void { this.reason = reason; this.message = message; this.emit(this.stopping ? 'stopping' : 'running') }
  fail(message: string): void { this.reason ??= 'run_error'; if (this.reason === 'run_error') this.message = message.slice(0, 400) }
  settle(ok: boolean): RunRecoverySnapshot {
    if (this.pending.size) throw new Error('The run cannot settle while started work is still active.')
    const uncertain = this.uncertainEffects || ['partial', 'unknown', 'accepted'].includes(this.delivery)
    if (this.uncertainEffects) { this.reason = 'tool_uncertain'; this.message = 'An action returned without a confirmed successful outcome; inspect its effects before replaying it.' }
    if (this.stopping && !uncertain) this.message = this.completedMutations ? 'Stopped after all started work settled; completed workspace changes remain recorded.' : 'Stopped after all started work settled.'
    const state = uncertain ? 'needs_review' : this.stopping ? 'stopped' : this.reason && this.reason !== 'run_error' ? 'paused' : ok ? 'completed' : 'failed'
    return this.emit(state)
  }
  private emit(state: RunRecoverySnapshot['state']): RunRecoverySnapshot {
    const retry = runRetryVerdict({ delivery: this.delivery, uncertainEffects: this.uncertainEffects })
    const terminal = ['stopped', 'paused', 'failed', 'needs_review', 'completed'].includes(state)
    const snapshot: RunRecoverySnapshot = { runId: this.runId, state, reason: this.reason, message: this.message, completedTools: this.completedTools, completedMutations: this.completedMutations, pendingOperations: this.pending.size, pendingMutations: [...this.pending.values()].filter(Boolean).length, delivery: this.delivery, retry: !terminal || state === 'completed' ? 'unavailable' : retry.allowed ? 'safe' : 'requires_review', retryReason: retry.reason, canStartNewRun: terminal && this.pending.size === 0, updatedAt: this.now() }
    // Observers cannot change the lifecycle or turn successful cleanup into an exception.
    try { this.publish?.(structuredClone(snapshot)) } catch { /* observational callback */ }
    return snapshot
  }
}
