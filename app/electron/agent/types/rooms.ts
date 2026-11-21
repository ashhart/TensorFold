import type { RoomMode } from '../room-outcomes'
import type { SharedMemoryState } from '../shared-room-memory'
import type { WorkspaceLocation } from './workspace'
import type { ReasoningEffortSetting } from './settings'

// ---- Multi-agent Rooms (Project → Room → Seats) ----
export interface Seat {
  id: string
  name: string
  role: string
  providerId: string
  providerAccountId?: string | null
  model: string
  reasoningEffort?: ReasoningEffortSetting
  lane?: RoomSeatLane
  roleProfileId?: AgentRoleProfileId
  fanOutBudget?: FanOutBudget
}

// Per-seat liveness for room deliberation (BUG #53). Parallels MissionHeartbeat:
// tracks when a seat last took a turn so a stuck/slow seat surfaces as stalled
// instead of silently locking the room. Runtime-ephemeral — stored on Room.
export interface SeatHeartbeat {
  seatId: string
  seatName: string
  intervalSeconds: number // expected cadence between turns
  stallGraceSeconds: number // tolerance after nextDueAt before signalling a stall
  required: boolean
  count: number // turns this seat has taken
  lastBeatAt: number | null // when the seat last took a turn
  nextDueAt: number | null // when the next turn is expected
  missedSince: number | null // when the stall was first detected (null = healthy)
}

export type RoomSeatLane = 'planning' | 'development' | 'review' | 'research' | 'operations'

export type AgentRoleProfileId = 'planner' | 'builder' | 'reviewer' | 'researcher' | 'operator'

export type FanOutBudget = 'single' | 'paired' | 'council' | 'swarm'

export type RoomCollaborationMode =
  | 'balanced'
  | 'frontier_plans_local_builds'
  | 'local_private'
  | 'frontier_council'
  | 'supervisor'

export type RoomDeliberationEffort = 'quick' | 'normal' | 'deep' | 'max'

export interface RoomWorkerBudget {
  maxActive: number
  maxStarts: number
}

export interface RoomMemoryEntry {
  id: string
  kind: 'summary' | 'decision' | 'preference' | 'finding'
  text: string
  source: 'manual' | 'room'
  createdAt: number
  updatedAt: number
}

export interface RoomMemoryDiff {
  added: RoomMemoryEntry[]
  removed: RoomMemoryEntry[]
  summaryChanged: boolean
  previousCount: number
  nextCount: number
}

export type RoomManualMemoryCommand =
  | { action: 'add'; kind: RoomMemoryEntry['kind']; text: string }
  | { action: 'remove'; entryId: string }

export type RoomState =
  | 'idle'
  | 'created'
  | 'ready'
  | 'degraded'
  | 'queued'
  | 'running'
  | 'waiting_user'
  | 'waiting_external'
  | 'complete'
  | 'stopped'
  | 'failed'
  | 'archived'
  | 'error'

export interface RoomEventItem {
  id: string
  tick: number
  kind: string // task | speak | ask | assign | object | claim | check | vote | yield | final | tool | notice | error
  seat: string // seat name | 'you' | 'system' | 'moderator'
  target: string
  text: string
  replyTo?: string // for ask/assign — the room event id this message answers (lenient RE: marker)
  tool?: string // for kind 'tool' — the tool name (read_file, edit_file, run_command, …)
  args?: Record<string, unknown> // for kind 'tool' — normalized tool arguments and lifecycle metadata
  subAgentId?: string // for delegate tool runs
  parentSeat?: string // for delegate tool runs
  status?: string // for delegate tool runs
  path?: string // for file tools — the file the agent touched
  createdAt: number
}

export interface Room {
  id: string
  projectId: string
  name: string
  // Outcome mode (VST-019): what the room is FOR — sets its stop condition, expected
  // final artifact, seat/moderator behavior, and default budget. Unset = legacy room
  // with no outcome gate. See electron/agent/room-outcomes.ts.
  mode?: RoomMode | null
  seats: Seat[]
  rootCwd?: string | null
  workspaceCwd?: string | null
  scratchpadCwd?: string | null
  workspace?: WorkspaceLocation | null
  collaborationMode?: RoomCollaborationMode
  deliberationEffort?: RoomDeliberationEffort
  workerBudget?: RoomWorkerBudget
  memory?: RoomMemoryEntry[]
  memorySummary?: string
  // Live shared scratchpad (decisions/plan/blockers) seats read and write during a
  // run via the memory_write room tool. Persisted with the room (rooms.json).
  sharedMemory?: SharedMemoryState
  // Per-seat liveness while the room is running (BUG #53). Keyed by seat id.
  // Runtime-ephemeral; surfaced to the supervisor job + UI for stall detection.
  seatHeartbeats?: Record<string, SeatHeartbeat>
  problem: string
  state: RoomState
  events: RoomEventItem[]
  finalText: string
  archivedAt?: number | null
  /** Origin host for an Orchestrator-owned renderer projection/archive record. */
  orchestratorHostId?: string | null
  createdAt: number
  updatedAt: number
}

// Room worktree merge-back result (room:mergeWorktree). 'gated' = final gates not green,
// 'blocked' = git-hygiene preflight refused, 'conflict' = merge aborted and ticketed.
export type RoomMergeStatus = 'merged' | 'nothing' | 'conflict' | 'gated' | 'blocked' | 'error'

export interface RoomMergeConflictInfo {
  branch: string
  files: string[]
  ticketId: string | null
}

export interface RoomMergeResult {
  ok: boolean
  status: RoomMergeStatus
  baseBranch: string | null
  merged: string[]
  remaining: string[]
  conflict: RoomMergeConflictInfo | null
  blockers: string[]
  message: string
}

export type RoomStreamEvent = (
  | { roomId: string; type: 'status'; state: Room['state'] }
  | { roomId: string; type: 'turn'; seat: string }
  | { roomId: string; type: 'delta'; seat: string; chunk: string }
  | { roomId: string; type: 'event'; item: RoomEventItem }
  | { roomId: string; type: 'memory'; memory: RoomMemoryEntry[]; memorySummary: string; diff?: RoomMemoryDiff }
  | { roomId: string; type: 'final'; text: string }
  | { roomId: string; type: 'done' }
  | { roomId: string; type: 'error'; message: string; fatal?: boolean }
) & { orchestratorHostId?: string | null }
