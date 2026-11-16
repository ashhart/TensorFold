import { randomUUID } from 'node:crypto'
import type { Automation } from '../../agent/types'
import { store } from '../state'
import { nextDailyAutomationRun, normalizeAutomationTimeZone } from '../../../shared/automation-time'

export function mkAutomation(p: Partial<Automation>): Automation {
  return {
    id: randomUUID(),
    name: p.name || 'Untitled automation',
    prompt: p.prompt || '',
    scheduleKind: p.scheduleKind || 'manual',
    everyMinutes: p.everyMinutes ?? 60,
    atHour: p.atHour ?? 9,
    atMinute: p.atMinute ?? 0,
    ...(p.timeZone !== undefined ? { timeZone: normalizeAutomationTimeZone(p.timeZone) } : {}),
    providerId: p.providerId ?? store.settings?.activeProviderId ?? null,
    providerAccountId: p.providerAccountId ?? store.settings?.activeProviderAccountId ?? null,
    model: p.model ?? store.settings?.activeModel ?? null,
    mode: p.mode || 'chat',
    cwd: p.cwd ?? store.settings?.defaultCwd ?? null,
    deliverTo: p.deliverTo || 'inbox',
    ...(p.deliveryTarget ? { deliveryTarget: { ...p.deliveryTarget } } : {}),
    ...(p.previewOnly !== undefined ? { previewOnly: p.previewOnly } : {}),
    ...(p.botId ? { botId: p.botId } : {}),
    ...(p.sourceScope ? { sourceScope: p.sourceScope.map(scope => ({ ...scope })) } : {}),
    ...(p.routineKind ? { routineKind: p.routineKind } : {}),
    enabled: p.enabled ?? false,
    lastRunAt: null,
    nextRunAt: null,
    lastStatus: null,
    runs: [],
    // Industrial fields (VST-038) carried through on create so a template / NL-built draft
    // keeps its misfire policy, risk gating, and completion policy.
    ...(p.misfire ? { misfire: p.misfire } : {}),
    ...(p.requiredRisks ? { requiredRisks: p.requiredRisks } : {}),
    ...(p.acknowledgedRisks ? { acknowledgedRisks: p.acknowledgedRisks } : {}),
    ...(p.completionPolicy ? { completionPolicy: p.completionPolicy } : {}),
    createdAt: Date.now(),
    updatedAt: Date.now(),
  }
}

export function computeNextRun(a: Automation, from = Date.now()): number | null {
  if (!a.enabled || a.scheduleKind === 'manual') return null
  if (a.scheduleKind === 'interval') {
    const step = Math.max(1, a.everyMinutes) * 60_000
    const base = a.lastRunAt ?? from
    let next = base + step
    if (next <= from) next = from + step
    return next
  }
  return nextDailyAutomationRun(a.atHour, a.atMinute, Math.max(from, a.lastRunAt ?? from), a.timeZone)
}

