import type { PlanStep } from './types'
import { openPlanSteps as openSteps, planProgress as progressOf } from '../../shared/plan-progress'

export type { PlanProgress } from '../../shared/plan-progress'

// Todo tracker: pure decision logic for the two behaviors that keep a plan honest during long runs.
// loop.ts owns the I/O; everything here is deterministic and independently tested.

/** Mutating tools counted toward the mid-run stale-todo nudge (reads never nudge). */
export const TODO_MUTATING_TOOLS = new Set(['run_command', 'write_file', 'edit_file', 'apply_patch'])

export const MID_RUN_NUDGE_MUTATION_THRESHOLD = 12
export const MID_RUN_NUDGE_MAX_PER_RUN = 2
export const STOP_REMINDER_MAX_PER_RUN = 3

export const planProgress = progressOf
export const openPlanSteps = openSteps

/** Maps raw update_plan args onto PlanSteps; tolerant of the legacy {text, status:'pending'|'done'} shape. */
export function normalizePlanSteps(raw: unknown): PlanStep[] {
  if (!Array.isArray(raw)) return []
  const steps: PlanStep[] = []
  for (const item of raw) {
    const source: Record<string, unknown> = item && typeof item === 'object' ? item as Record<string, unknown> : {}
    const text = String(source.text ?? item ?? '').trim().slice(0, 200)
    if (!text) continue
    const rawStatus = source.status
    // Legacy Codex shape carried only a done boolean; keep honoring it.
    const doneFlag = source.done === true
    const status: PlanStep['status'] =
      rawStatus === 'in_progress' || rawStatus === 'active' ? 'in_progress'
        : rawStatus === 'done' || rawStatus === 'completed' ? 'done'
          : rawStatus === 'blocked' ? 'blocked'
            : rawStatus === 'dropped' || rawStatus === 'abandoned' ? 'dropped'
              : doneFlag ? 'done'
                : 'pending'
    const blocker = status === 'blocked' && typeof source.blocker === 'string' && source.blocker.trim()
      ? source.blocker.trim().slice(0, 160)
      : undefined
    const depth = source.subtask === true || source.depth === 1 ? 1 : 0
    steps.push({
      id: `s${steps.length}`,
      text,
      done: status === 'done',
      status,
      ...(blocker ? { blocker } : {}),
      ...(depth ? { depth: 1 } : {}),
    })
  }
  return steps.slice(0, 64)
}

/** One-line tree glyph per step, for the update_plan timeline output. */
export function renderPlanTree(plan: PlanStep[]): string {
  return plan.map((step) => {
    const indent = step.depth ? '  ' : ''
    const marker = step.done || step.status === 'done' ? '✓'
      : step.status === 'in_progress' ? '▸'
        : step.status === 'blocked' ? '■'
          : step.status === 'dropped' ? '×'
            : '○'
    const blocker = step.status === 'blocked' && step.blocker ? ` — ${step.blocker}` : ''
    return `${indent}${marker} ${step.text}${blocker}`
  }).join('\n')
}

/**
 * Mid-run nudge: after many mutating tool calls without touching the plan, remind the
 * model to keep the checklist current (max 2 per run). Returns the guard message.
 */
export function midRunTodoNudge(mutationsSincePlanTouch: number, plan: PlanStep[], nudgesSoFar: number): string | null {
  if (nudgesSoFar >= MID_RUN_NUDGE_MAX_PER_RUN) return null
  if (mutationsSincePlanTouch < MID_RUN_NUDGE_MUTATION_THRESHOLD) return null
  const open = openPlanSteps(plan)
  if (!open.length) return null
  const noun = open.length === 1 ? 'item' : 'items'
  return `${open.length} plan ${noun} still open. If you finished work since your last update_plan, mark it done now so progress stays visible; if a step is blocked or no longer needed, say so with blocked/dropped instead of leaving it pending. Otherwise keep working.`
}

/**
 * Stop reminder: the model produced a final no-tool answer while plan items remain
 * open. Skips when the assistant is asking the user a question (their turn), when the
 * question mark is the only "open" thing, or once the reminder budget is spent.
 */
export function stopTodoReminder(plan: PlanStep[], remindersSoFar: number, assistantText: string): { message: string; open: PlanStep[] } | null {
  if (remindersSoFar >= STOP_REMINDER_MAX_PER_RUN) return null
  const open = openPlanSteps(plan)
  if (!open.length) return null
  if (assistantEndsWithUserQuestion(assistantText)) return null
  const list = open.slice(0, 12).map((step) => `${step.depth ? '  - ' : '- '}${step.text}${step.status === 'blocked' && step.blocker ? ` (blocked: ${step.blocker})` : ''}`).join('\n')
  const message = `You stopped with ${open.length} open plan item${open.length === 1 ? '' : 's'}:\n${list}\n\nContinue working on these items, or update the plan to mark them done, dropped, or blocked before finishing. (Reminder ${remindersSoFar + 1}/${STOP_REMINDER_MAX_PER_RUN})`
  return { message, open }
}

function assistantEndsWithUserQuestion(text: string): boolean {
  const trimmed = text.trim()
  if (!trimmed) return false
  const lastLine = trimmed.split('\n').filter(Boolean).pop() ?? ''
  if (!/[?？]$/.test(lastLine.trim())) return false
  return /\b(you|your|we|our|would you|could you|should i|do you|shall)\b/i.test(lastLine) || /[^\x00-\x7F]/.test(lastLine)
}
