import type { Automation, AutomationDeliveryTarget, AutomationSourceScope } from '../electron/agent/types'
import { normalizeAutomationTimeZone } from './automation-time'

export interface PersonalRoutineInput {
  kind: 'morning-brief' | 'meeting-prep' | 'follow-up'
  botId: string
  providerId: string
  providerAccountId?: string | null
  model: string
  sourceScope: AutomationSourceScope[]
  timeZone: string
  atHour?: number
  atMinute?: number
  deliveryTarget?: AutomationDeliveryTarget
}

const ROUTINES = {
  'morning-brief': { name: 'Morning brief', prompt: 'Prepare a concise personal brief from only the supplied selected-source items. Identify today’s priorities, unanswered requests and time-sensitive items, citing the source item IDs. If there is no selected activity, say so; do not invent it.' },
  'meeting-prep': { name: 'Meeting preparation', prompt: 'Prepare a short meeting preparation note using only supplied selected-source items. Summarize purpose, participants, open questions and supporting context, citing source item IDs. State missing information explicitly.' },
  'follow-up': { name: 'Follow-up suggestions', prompt: 'Use only supplied selected-source items to suggest unanswered requests and useful follow-ups, citing source item IDs. Draft suggestions for the user’s review; do not contact anybody, send messages or invent commitments.' },
} as const

/** A small original recipe over the existing scheduler, never a second bot runtime. */
export function buildPersonalRoutine(input: PersonalRoutineInput): Partial<Automation> {
  const recipe = ROUTINES[input.kind]
  if (!recipe || !input.botId?.trim() || !input.providerId?.trim() || !input.model?.trim()) throw new Error('Choose a bot with a saved model before creating a routine.')
  if (!Array.isArray(input.sourceScope) || input.sourceScope.length > 32) throw new Error('Choose at most 32 exact connected sources.')
  const timeZone = normalizeAutomationTimeZone(input.timeZone)
  if (!timeZone) throw new Error('Choose an explicit timezone for this daily routine.')
  const atHour = input.atHour ?? 9, atMinute = input.atMinute ?? 0
  if (!Number.isInteger(atHour) || atHour < 0 || atHour > 23 || !Number.isInteger(atMinute) || atMinute < 0 || atMinute > 59) throw new Error('Choose a valid daily time.')
  return {
    name: recipe.name, prompt: recipe.prompt, routineKind: input.kind,
    botId: input.botId, providerId: input.providerId, providerAccountId: input.providerAccountId ?? null, model: input.model,
    sourceScope: input.sourceScope.map(scope => ({ ...scope })), mode: 'chat', cwd: null,
    scheduleKind: 'daily', atHour, atMinute, timeZone, enabled: false,
    deliverTo: 'inbox', previewOnly: true,
    ...(input.deliveryTarget ? { deliveryTarget: { ...input.deliveryTarget } } : {}),
    misfire: 'coalesce', requiredRisks: [], acknowledgedRisks: [], completionPolicy: { type: 'none', version: 0 },
  }
}
