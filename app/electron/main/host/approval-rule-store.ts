import type { ApprovalRule, ApprovalRuleDecision, ApprovalRuleScope } from '../../agent/types'
import {
  approvalRuleIdentity,
  boundedApprovalRuleFields,
  consumeApprovalRule,
  type ApprovalRuleContext,
} from '../../agent/approval-rules'

export type ApprovalRuleDetails = Partial<Pick<
  ApprovalRule,
  'label' | 'explanation' | 'legacyTool' | 'projectId' | 'workspace'
>>

export interface DurableApprovalRuleStoreDeps {
  getRules(): ApprovalRule[]
  setRules(rules: ApprovalRule[]): void
  persist(label: string): Promise<void>
  now?: () => number
  newId: () => string
  onSkipped?: (input: { scope: ApprovalRuleScope; key: string; reason: 'missing_project_or_workspace_scope' }) => void | Promise<void>
  onChanged?: (input: {
    action: 'created' | 'updated'
    scope: ApprovalRuleScope
    key: string
    previous: Pick<ApprovalRule, 'decision' | 'duration' | 'projectId' | 'workspace' | 'expiresAt' | 'maxUses' | 'useCount'> | null
    next: ApprovalRule
  }) => void | Promise<void>
}

/**
 * Serializes the read-modify-persist cycle for bounded approval rules.
 *
 * In-memory mutation alone is not authority: a matched rule is returned only after its
 * incremented use count is durable. Remembering a rule rejects on persistence failure so an
 * `allow_always` caller cannot continue as though a policy exists when it does not.
 */
export function createDurableApprovalRuleStore(deps: DurableApprovalRuleStoreDeps) {
  let mutationTail: Promise<void> = Promise.resolve()

  function serialize<T>(mutation: () => Promise<T>): Promise<T> {
    const run = mutationTail.then(mutation, mutation)
    mutationTail = run.then(() => undefined, () => undefined)
    return run
  }

  function find(
    scope: ApprovalRuleScope,
    key: string,
    context: ApprovalRuleContext = {},
  ): Promise<ApprovalRule | null> {
    return serialize(async () => {
      const now = deps.now?.() ?? Date.now()
      const priorRules = deps.getRules()
      const consumed = consumeApprovalRule(priorRules, { scope, key, ...context, now })
      if (!consumed.rule) return null

      deps.setRules(consumed.rules)
      try {
        await deps.persist('approvalRules.consume')
      } catch {
        // Restore only the approval-rule slice. The host setter deliberately preserves any
        // unrelated settings changes that landed while the critical write was in flight.
        deps.setRules(priorRules)
        return null
      }
      return consumed.rule
    })
  }

  function remember(
    scope: ApprovalRuleScope,
    key: string,
    decision: ApprovalRuleDecision,
    duration: 'always' = 'always',
    details: ApprovalRuleDetails = {},
  ): Promise<ApprovalRule | null> {
    return serialize(async () => {
      const now = deps.now?.() ?? Date.now()
      let bounds: ReturnType<typeof boundedApprovalRuleFields>
      try {
        bounds = boundedApprovalRuleFields({ projectId: details.projectId, workspace: details.workspace }, now)
      } catch {
        await deps.onSkipped?.({ scope, key, reason: 'missing_project_or_workspace_scope' })
        return null
      }

      const identity = approvalRuleIdentity(scope, key, bounds)
      const rules = deps.getRules()
      const index = rules.findIndex((item) => approvalRuleIdentity(item.scope, item.key, item) === identity)
      const existing = index >= 0 ? rules[index]! : null
      const previous = existing ? {
        decision: existing.decision,
        duration: existing.duration,
        projectId: existing.projectId ?? null,
        workspace: existing.workspace ?? null,
        expiresAt: existing.expiresAt ?? null,
        maxUses: existing.maxUses ?? null,
        useCount: existing.useCount ?? null,
      } : null
      const next: ApprovalRule = {
        ...(existing ?? { id: deps.newId(), createdAt: now }),
        scope,
        key,
        decision,
        duration,
        label: details.label ?? existing?.label,
        explanation: details.explanation ?? existing?.explanation,
        legacyTool: details.legacyTool ?? existing?.legacyTool,
        ...bounds,
        updatedAt: now,
        lastUsedAt: null,
      }
      const approvalRules = rules.slice()
      if (index >= 0) approvalRules[index] = next
      else approvalRules.push(next)
      deps.setRules(approvalRules)

      try {
        await deps.persist('approvalRules.remember')
      } catch (error) {
        deps.setRules(rules)
        throw error
      }

      await deps.onChanged?.({ action: previous ? 'updated' : 'created', scope, key, previous, next })
      return next
    })
  }

  return { find, remember }
}
