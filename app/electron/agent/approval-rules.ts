import type { ApprovalRule, ApprovalRuleDecision, ApprovalRuleScope } from './types'

export const APPROVAL_RULE_TTL_MS = 8 * 60 * 60_000
export const APPROVAL_RULE_MAX_USES = 20

export interface ApprovalRuleContext {
  projectId?: string | null
  workspace?: string | null
}

export interface ConsumeApprovalRuleInput extends ApprovalRuleContext {
  scope: ApprovalRuleScope
  key: string
  now?: number
}

function clean(value: string | null | undefined): string | null {
  const next = String(value ?? '').trim()
  return next || null
}

export function boundedApprovalRuleFields(
  context: ApprovalRuleContext,
  now = Date.now(),
  options: { ttlMs?: number; maxUses?: number } = {},
): Pick<ApprovalRule, 'projectId' | 'workspace' | 'expiresAt' | 'maxUses' | 'useCount'> {
  const projectId = clean(context.projectId)
  const workspace = clean(context.workspace)
  if (!projectId && !workspace) throw new Error('Saved approval rules require a project or workspace scope.')
  const ttlMs = Math.max(60_000, Math.min(24 * 60 * 60_000, Math.floor(options.ttlMs ?? APPROVAL_RULE_TTL_MS)))
  const maxUses = Math.max(1, Math.min(100, Math.floor(options.maxUses ?? APPROVAL_RULE_MAX_USES)))
  return { projectId, workspace, expiresAt: now + ttlMs, maxUses, useCount: 0 }
}

export function approvalRuleIsActive(rule: ApprovalRule, context: ApprovalRuleContext, now = Date.now()): boolean {
  if (rule.duration !== 'always') return false
  const projectId = clean(rule.projectId)
  const workspace = clean(rule.workspace)
  // Backward compatibility is deliberately fail-closed: legacy unscoped/unbounded rules remain
  // readable/revocable in settings but never auto-resolve a new approval.
  if (!projectId && !workspace) return false
  if (!Number.isFinite(rule.expiresAt) || Number(rule.expiresAt) <= now) return false
  if (!Number.isInteger(rule.maxUses) || Number(rule.maxUses) < 1) return false
  if (!Number.isInteger(rule.useCount) || Number(rule.useCount) < 0 || Number(rule.useCount) >= Number(rule.maxUses)) return false
  if (projectId && projectId !== clean(context.projectId)) return false
  if (workspace && workspace !== clean(context.workspace)) return false
  return true
}

export function consumeApprovalRule(
  rules: ApprovalRule[],
  input: ConsumeApprovalRuleInput,
): { rules: ApprovalRule[]; rule: ApprovalRule | null } {
  const now = input.now ?? Date.now()
  const index = rules.findIndex((rule) =>
    rule.scope === input.scope && rule.key === input.key && approvalRuleIsActive(rule, input, now))
  if (index < 0) return { rules, rule: null }
  const consumed: ApprovalRule = {
    ...rules[index],
    useCount: Number(rules[index].useCount) + 1,
    lastUsedAt: now,
    updatedAt: now,
  }
  const next = rules.slice()
  next[index] = consumed
  return { rules: next, rule: consumed }
}

export function approvalRuleIdentity(
  scope: ApprovalRuleScope,
  key: string,
  context: ApprovalRuleContext,
): string {
  return [scope, key, clean(context.projectId) ?? '', clean(context.workspace) ?? ''].join('\u0000')
}

export function approvalRuleDecision(rule: ApprovalRule | null): ApprovalRuleDecision | null {
  return rule?.decision === 'allow' || rule?.decision === 'deny' ? rule.decision : null
}
