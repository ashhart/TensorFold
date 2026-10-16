// Room outcome modes (VST-019). Renderer-side copy of the electron/agent/room-outcomes.ts
// profiles (runtime code cannot cross the renderer/main boundary); room-outcomes.test.ts
// pins ids + labels to the profiles so the two cannot drift.
import type { RoomDeliberationEffort, RoomMode } from './types'

export const ROOM_MODE_OPTIONS: Array<{ id: RoomMode; label: string; detail: string; budget: RoomDeliberationEffort }> = [
  { id: 'plan', label: 'Plan', detail: 'Ends with an ordered plan; objections resolved or named as dissent.', budget: 'normal' },
  { id: 'implement', label: 'Implement', detail: 'Ends when the change is built AND verified (tests/checks stated).', budget: 'deep' },
  { id: 'review', label: 'Review', detail: 'Ends with an explicit verdict: approved or changes requested.', budget: 'normal' },
  { id: 'debug', label: 'Debug', detail: 'Ends when the root cause is named with evidence, plus the fix or next step.', budget: 'deep' },
  { id: 'incident', label: 'Incident', detail: 'Ends when mitigated, root-caused, and follow-ups are assigned.', budget: 'max' },
  { id: 'research', label: 'Research', detail: 'Ends with sourced findings and a conclusion, or named unknowns.', budget: 'normal' },
]

export function roomModeDefaultEffort(mode: RoomMode): RoomDeliberationEffort {
  return ROOM_MODE_OPTIONS.find((option) => option.id === mode)?.budget ?? 'normal'
}
