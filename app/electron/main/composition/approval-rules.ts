import { randomUUID } from 'node:crypto'
import type { ApprovalRuleContext } from '../../agent/approval-rules'
import { store } from '../state'
import { persistSettingsCritical } from '../host/persistence'
import { auditEvent } from '../host/audit'
import { createDurableApprovalRuleStore, type ApprovalRuleDetails } from '../host/approval-rule-store'

export function createApplicationApprovalRules() {
  const approvalRuleStore = createDurableApprovalRuleStore({
    getRules: () => store.settings.approvalRules ?? [],
    setRules: (approvalRules) => { store.settings = { ...store.settings, approvalRules } },
    persist: (label) => persistSettingsCritical(label, store.settings),
    newId: () => randomUUID(),
    onSkipped: ({ scope, key, reason }) => auditEvent('approval.rule_skipped', { scope, key, reason }),
    onChanged: ({ action, scope, key, previous, next }) => auditEvent('approval.rule_changed', {
      action,
      scope,
      key,
      previous,
      next: {
        decision: next.decision,
        duration: next.duration,
        projectId: next.projectId,
        workspace: next.workspace,
        expiresAt: next.expiresAt,
        maxUses: next.maxUses,
        useCount: next.useCount,
      },
    }),
  })

  function findApprovalRule(scope: 'tool' | 'connector', key: string, context: ApprovalRuleContext = {}) {
    return approvalRuleStore.find(scope, key, context)
  }

  function rememberApprovalRule(
    scope: 'tool' | 'connector',
    key: string,
    decision: 'allow' | 'deny',
    duration: 'always' = 'always',
    details: ApprovalRuleDetails = {},
  ) {
    return approvalRuleStore.remember(scope, key, decision, duration, details)
  }

  return { findApprovalRule, rememberApprovalRule }
}
