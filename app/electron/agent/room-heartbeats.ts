import type { RoomDeliberationEffort, Seat, SeatHeartbeat } from './types'

// Minimal shape the heartbeat helpers need — lets tests exercise stall detection
// without standing up a full SeatRuntime (messages/recent are irrelevant here).
export interface SeatRuntimeLike {
  seat: Pick<Seat, 'id' | 'name'>
  heartbeat: SeatHeartbeat
}

// Default grace after nextDueAt before a slow seat is flagged stalled (BUG #53).
// Tolerates transient model latency so a normal slow turn does not false-positive.
const SEAT_STALL_GRACE_SECONDS = 30

// Heartbeat cadence derived from the room deliberation effort. Deeper sessions
// expect longer per-turn thinking, so we widen the interval to avoid false stalls.
export function seatHeartbeatIntervalSeconds(effort?: RoomDeliberationEffort | null): number {
  switch (effort) {
    case 'quick': return 30
    case 'deep': return 90
    case 'max': return 120
    case 'normal':
    default: return 60
  }
}

// Seed a healthy heartbeat for a seat at room start.
export function initSeatHeartbeat(seat: Pick<Seat, 'id' | 'name'>, intervalSeconds: number, now: number): SeatHeartbeat {
  return {
    seatId: seat.id,
    seatName: seat.name,
    intervalSeconds,
    stallGraceSeconds: SEAT_STALL_GRACE_SECONDS,
    required: true,
    count: 0,
    lastBeatAt: now,
    nextDueAt: now + intervalSeconds * 1000,
    missedSince: null,
  }
}

// Record that a seat produced a turn: bump count, advance the clock, clear any stall.
export function recordSeatBeat(hb: SeatHeartbeat, now: number): void {
  hb.count++
  hb.lastBeatAt = now
  hb.nextDueAt = hb.required ? now + hb.intervalSeconds * 1000 : null
  hb.missedSince = null
}

// Names of seats currently flagged stalled (missedSince set), excluding one seat
// by name (the seat trying to FINAL is obviously alive). Used to block a premature
// finalize while a teammate is hung (BUG #53).
export function stalledSeatNames(runtimes: SeatRuntimeLike[], exclude?: string): string[] {
  return runtimes
    .filter((rt) => rt.heartbeat.required && rt.heartbeat.missedSince != null && rt.seat.name !== exclude)
    .map((rt) => rt.seat.name)
}

// Re-arm the stall window for seats that actually took a turn in the previous pass.
// A single pass over several slow models can take longer than interval+grace, so a
// seat that beat last pass and is merely waiting its turn must not be false-flagged;
// pushing its nextDueAt forward to this pass start prevents that. Crucially, seats
// that did NOT run last pass (e.g. one that keeps timing out without ever calling
// recordSeatBeat, or any seat on the very first pass) are left untouched so their
// window keeps counting and evaluateSeatHeartbeats can actually flag a real stall —
// re-arming every non-stalled seat unconditionally is what made stall detection dead
// code (BUG #53 regression).
export function rearmSeatHeartbeats(
  runtimes: SeatRuntimeLike[],
  ranLastPass: Set<string>,
  passStart: number,
): void {
  for (const rt of runtimes) {
    const hb = rt.heartbeat
    if (hb.required && hb.missedSince == null && ranLastPass.has(rt.seat.id)) {
      hb.nextDueAt = passStart + hb.intervalSeconds * 1000
    }
  }
}

// Detect seats whose turn is overdue past the grace window. Mutates each newly
// stalled heartbeat's missedSince so a stall is reported once (not every tick),
// and returns human-readable notices for the moderator to surface.
export function evaluateSeatHeartbeats(runtimes: SeatRuntimeLike[], now: number): string[] {
  const stalled: string[] = []
  for (const rt of runtimes) {
    const hb = rt.heartbeat
    if (!hb.required || hb.lastBeatAt == null || hb.nextDueAt == null) continue
    if (hb.missedSince != null) continue // already flagged; do not re-report
    const stallAt = hb.nextDueAt + hb.stallGraceSeconds * 1000
    if (now >= stallAt) {
      hb.missedSince = hb.nextDueAt
      const ago = Math.round((now - hb.lastBeatAt) / 1000)
      stalled.push(`Seat "${rt.seat.name}": heartbeat overdue (last beat ${ago}s ago)`)
    }
  }
  return stalled
}
