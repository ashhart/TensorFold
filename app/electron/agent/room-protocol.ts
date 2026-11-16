import type { Project, Provider, Room, RoomEventItem, Seat } from './types'
import { isExternalProvider } from '../../shared/provider-trust'
import { PROJECT_CONTEXT_WITHHELD_NOTE } from './planning-guard'
import { VONTRA_STYLE_PROMPT } from './prompt-style'
import { formatRoomMemory, seatCollaborationInstructions } from './room-collaboration'
import { roomModePromptBlock, roomOutcomeProfile } from './room-outcomes'
import { renderSharedMemory } from './shared-room-memory'

// Ported from LLMComposer's room/huddle engine: multiple provider-bound seats
// deliberate on one shared problem as an attributed dialogue, with SENSE
// injection so each seat knows it is one of several real teammates.

const VERBS = new Set(['SPEAK', 'ASK', 'OBJECT', 'ASSIGN', 'CLAIM', 'CHECK', 'VOTE', 'YIELD', 'FINAL', 'TOOL'])

export function buildPrimer(project: Project, room: Room, seat: Seat, roster: Seat[], provider?: Provider): string {
  const rosterText = roster.map((s) => `- ${s.name}: ${s.role} (model ${s.model})`).join('\n')
  const roomMemory = formatRoomMemory(room)
  const sharedMemory = room.sharedMemory ? renderSharedMemory(room.sharedMemory) : ''
  return [
    `You are ${seat.name} in a live multi-agent working session with the teammates below.`,
    '',
    `ROOM ROLE\nYour visible room role is ${seat.role}. Use it to decide when to volunteer, challenge, defer, or take a ticket, but keep your general reasoning intact.`,
    `\n${seatCollaborationInstructions(room, seat, provider)}`,
    roomOutcomeProfile(room.mode) ? `\nROOM OUTCOME\n${roomModePromptBlock(roomOutcomeProfile(room.mode)!)}` : '',
    project.team ? `\nTEAM CHARTER\n${project.team}` : '',
    // External seats plan only and never see project memory (planning-guard.ts).
    project.memory ? (isExternalProvider(provider) ? `\n${PROJECT_CONTEXT_WITHHELD_NOTE}` : `\nPROJECT MEMORY\n${project.memory}`) : '',
    roomMemory ? `\nROOM MEMORY\n${roomMemory}` : '',
    sharedMemory ? `\n${sharedMemory}` : '',
    '',
    'ROOM CONTRACT',
    'Messages tagged [room] are what teammates actually said; [moderator] hands you the floor.',
    'Speak in first person, briefly and concretely. React to what was just said instead of restating it.',
    "Never write [room] or [moderator] lines yourself, and never invent a teammate's reply.",
    'Never repeat a question or point you already made.',
    'End every reply with exactly one line: ACTION: <VERB> <content>.',
    'Verbs: SPEAK, ASK, OBJECT, ASSIGN, CLAIM, CHECK, VOTE, YIELD, FINAL, TOOL.',
    'ACTION: ASK <teammate> <question> addresses one teammate by name. ACTION: FINAL <answer> ends the room only when the task is genuinely done.',
    'When you answer a specific ASK or ASSIGN, cite it by appending RE: #<id> to your ACTION line so the reply links back to the question.',
    'Only state what you can actually see in the [room] lines above. Do not claim progress that has not happened.',
    '',
    'TEAMWORK CONTRACT',
    'Make handoffs explicit. If a teammate asks or assigns you something, answer that first and say what you are unblocking for them.',
    'Help the room finish the user goal: create or claim concrete tickets, attach evidence, start workers when ready, and yield when waiting.',
    'If you disagree, name the risk and offer the next useful move instead of reopening the whole plan.',
    '',
    VONTRA_STYLE_PROMPT,
    '',
    'SHARED BOARD: coordinate through it so you never duplicate a teammate\'s work:',
    'ACTION: TOOL list_tickets: see every ticket and who is working on what.',
    'ACTION: TOOL claim_ticket {"title":"..."}: claim a ticket BEFORE you start it, so teammates know it is taken.',
    'ACTION: TOOL create_ticket {"title":"...","body":"...","priority":"high","risk":"medium","allowedCapabilities":["read"],"requiredEvidence":["files inspected","verification result"],"acceptanceCriteria":["..."]}: add an execution ticket to the board.',
    'ACTION: TOOL add_ticket_evidence {"title":"...","kind":"file","summary":"What you found","ref":"src/file.ts"}: attach evidence to a claimed ticket.',
    'ACTION: TOOL complete_ticket {"title":"...","evidence":[{"kind":"review","summary":"Reviewed X and found Y"}]}: mark your claimed ticket complete after attaching required evidence.',
    'ACTION: TOOL left_to_work: see the remaining room tickets, active workers, risks, and known file refs.',
    'ACTION: TOOL memory_write {"name":"decisions","content":"...","mode":"append-bullet"}: update the room\'s live shared memory (decisions, plan, blockers). Teammates see updates on their next turn.',
    'ACTION: TOOL start_ticket_work {"title":"...","phase":"build","paths":["src/x.ts"]}: ask the Room Supervisor to start a ticket worker thread in the room worktree. Use phase "review" for reviewer passes. If a file overlap is intentional, include "mergeDecision":"why this overlap is safe and how results will be merged".',
    'ACTION: TOOL delegate {"task":"..."}: hand a focused subtask to a sub-agent and get its result. Add "targetSeat":"SeatName" to run it on that seat\'s provider/model, or "providerId"/"accountId"/"model" for an explicit route; omit them to use your own.',
    'Always claim before working; never touch a ticket already claimed by someone else.',
    'Only start ticket workers for claimed or ready tickets. Declare paths when you know them so the supervisor can prevent overlapping edits.',
    'Do not ACTION: FINAL while room tickets are still open.',
    '',
    'WORKSPACE: inspect the shared project safely:',
    'ACTION: TOOL read_file {"path":"src/x.ts"}: read a file.',
    'ACTION: TOOL search {"query":"..."}: find relevant code.',
    'ACTION: TOOL list_dir {"path":"src"}: inspect folders.',
    'Room agents are read-only by default. Do not attempt edit_file, write_file, or run_command unless the user opens an approved work thread for that ticket.',
    'When you know the change needed, create/claim a ticket and explain the proposed edit for an approved thread to execute.',
    '',
    'TEAM ROSTER',
    rosterText,
  ].filter(Boolean).join('\n')
}

export interface Action { verb: string; text: string; raw: string; prose?: string }

// Lenient gate: missing action becomes SPEAK, empty body becomes YIELD.
export function gateAction(output: string): Action {
  const raw = (output ?? '').trim()
  if (!raw) return { verb: 'YIELD', text: '', raw }
  const lines = raw.split('\n')
  const actionLine = [...lines].reverse().find((l) => l.trim().toUpperCase().startsWith('ACTION:'))
  const prose = lines.filter((l) => l !== actionLine).join('\n').trim()
  if (!actionLine) return { verb: 'SPEAK', text: prose || raw, raw }
  const body = actionLine.trim().slice(actionLine.trim().indexOf(':') + 1).trim()
  const parts = body.split(/\s+/)
  const verb = (parts[0] || '').toUpperCase()
  const text = parts.slice(1).join(' ').trim()
  if (!VERBS.has(verb)) return { verb: 'SPEAK', text: prose || raw, raw }
  if (!text && verb !== 'YIELD') return { verb: 'YIELD', text: '', raw }
  // Keep the prose so peers receive the model's actual reasoning, not just the one-line ACTION tagline.
  return { verb, text, raw, prose }
}

export function renderForRoom(ev: RoomEventItem, compact = false): string {
  const label = ev.kind.toUpperCase()
  // Addressed asks/assigns carry a short ref (#id8) so a teammate can answer with
  // "RE: #<id>"; replies show the ref they answer (re #id8).
  const ref = (ev.kind === 'ask' || ev.kind === 'assign') ? ` #${ev.id.slice(0, 8)}` : ''
  const reply = ev.replyTo ? ` re #${ev.replyTo.slice(0, 8)}` : ''
  if (compact) return `[room] ${ev.seat} (${label}${reply}): ${ev.text}`
  const head = ev.target && ev.target !== 'all' && ev.target !== ev.seat
    ? `[room] ${ev.seat} (${label}, to ${ev.target}${ref}${reply}):`
    : `[room] ${ev.seat} (${label}${ref}${reply}):`
  return `${head}\n${ev.text}`
}

export function signature(raw: string): string {
  return raw.split('\n').filter((l) => !l.trim().toUpperCase().startsWith('ACTION:')).join(' ').trim().toLowerCase().slice(0, 400)
}

export function blockerSignature(blockers: string[]): string {
  return blockers.map((blocker) => blocker.replace(/\s+/g, ' ').trim().toLowerCase()).filter(Boolean).join('\n')
}

export function oneLine(text: string, max = 180): string {
  const compact = text.replace(/\s+/g, ' ').trim()
  return compact.length > max ? `${compact.slice(0, Math.max(0, max - 3))}...` : compact
}

export function roomTeamCue(room: Room, seat: Seat): string {
  const recent = room.events.filter((event) => !['task', 'notice', 'user'].includes(event.kind)).slice(-10)
  const directHandoffs = recent.filter((event) => (event.kind === 'ask' || event.kind === 'assign') && event.target === seat.name)
  const latestDirect = directHandoffs[directHandoffs.length - 1]
  const latestOtherHandoff = recent.filter((event) =>
    (event.kind === 'ask' || event.kind === 'assign') &&
    !!event.target &&
    event.target !== 'all' &&
    event.target !== seat.name
  ).slice(-1)[0]
  const latestClaim = recent.filter((event) => event.kind === 'claim').slice(-1)[0]
  const latestUserSteer = room.events
    .filter((event) => event.kind === 'user' && (event.target === 'all' || event.target === seat.name))
    .slice(-1)[0]
  const cue: string[] = []

  if (latestUserSteer) {
    cue.push(`USER STEERING: "${oneLine(latestUserSteer.text, 220)}". Treat this as the user's latest direction and respond to it before lower-priority room chatter.`)
  }

  if (latestDirect) {
    cue.push(`TEAM CUE: ${latestDirect.seat} handed work to you: "${oneLine(latestDirect.text, 220)}". Answer or unblock that first, naming ${latestDirect.seat}.`)
  } else if (latestOtherHandoff) {
    cue.push(`TEAM CUE: ${latestOtherHandoff.seat} handed work to ${latestOtherHandoff.target}. Let that teammate respond unless you can unblock with new evidence.`)
  }

  if (latestClaim) {
    cue.push(`Latest claim: ${latestClaim.seat} claimed "${oneLine(latestClaim.text, 120)}". Build on it and do not duplicate it.`)
  }

  cue.push('Work like a teammate: reference who you are helping, answer direct asks before new topics, create or claim the next useful ticket, start workers when ready, and ACTION: YIELD if you are waiting.')
  return cue.join(' ')
}

export function escapeReg(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}
