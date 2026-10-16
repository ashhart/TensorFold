// Loop Mode — builds a normalized RunContract (Definition of Done) from each
// surface's inputs (solo goal+criteria, room, or a kanban ticket). Pure.

import type { RunContract, LoopAutonomy, LoopContractSource, LoopMode } from './loop-types'
import { DEFAULT_MAX_CYCLES, HARD_MAX_CYCLES } from './loop-types'

const DEFAULT_FORBIDDEN = [
  'Do not stub, fake, or skip tests to make them pass.',
  'Do not disable type checks, linters, or assertions to get green.',
  'Do not claim work is done without running it.',
]
const DEFAULT_INTERRUPT = [
  'Pause only for missing secrets, product decisions, external outages, or unsafe/destructive actions.',
]

export interface BuildContractInput {
  source: LoopContractSource
  mode: Exclude<LoopMode, 'off'>
  goal: string
  criteria?: string[]
  verify?: string[]
  deliverables?: string[]
  autonomy?: LoopAutonomy
  maxCycles?: number
  ticket?: { acceptanceCriteria?: string[]; requiredEvidence?: string[]; allowedCapabilities?: string[]; risk?: string }
}

export function buildRunContract(input: BuildContractInput): RunContract {
  const acceptanceCriteria = (input.ticket?.acceptanceCriteria ?? input.criteria ?? []).filter(Boolean)
  const requiredEvidence = (input.ticket?.requiredEvidence ?? ['test']).filter(Boolean)
  const maxCycles = Math.min(HARD_MAX_CYCLES, input.maxCycles ?? DEFAULT_MAX_CYCLES[input.mode])
  return {
    source: input.source,
    goal: input.goal.trim(),
    deliverables: (input.deliverables ?? []).filter(Boolean),
    acceptanceCriteria: acceptanceCriteria.length ? acceptanceCriteria : ['The goal is fully and correctly implemented.'],
    verificationCommands: (input.verify ?? []).filter(Boolean),
    forbiddenShortcuts: [...DEFAULT_FORBIDDEN],
    requiredEvidence,
    interruptPolicy: [...DEFAULT_INTERRUPT],
    autonomy: input.autonomy ?? 'medium',
    maxCycles,
  }
}
