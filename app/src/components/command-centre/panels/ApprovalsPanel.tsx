import clsx from 'clsx'
import { formatDuration, relativeTime, shortPath } from '../../../lib/format'
import type { ApprovalRule } from '../../../lib/types'
import { Button } from '../../ui'
import { ApprovalRowItem, EmptyState, Panel, StatusIcon } from '../ui'
import { approvalRuleIcon, approvalRuleTitle } from '../attention'
import { approvalsShouldStick, type ApprovalGroup, type ApprovalRow } from '../model'
import { toneDot } from '../styles'

interface ApprovalsPanelProps {
  pendingApprovals: number
  approvalGroups: ApprovalGroup[]
  approvalRows: ApprovalRow[]
  savedPolicies: ApprovalRule[]
  revokePolicy: (rule: ApprovalRule) => Promise<void>
  revokingRuleId: string | null
}

export function ApprovalsPanel(props: ApprovalsPanelProps) {
  const { pendingApprovals, approvalGroups, approvalRows, savedPolicies, revokePolicy, revokingRuleId } = props
  const stickApprovals = approvalsShouldStick(pendingApprovals)

  return (
    <>
      <div id="command-centre-approvals" className={clsx(stickApprovals && 'sticky top-0 z-10')}>
        <Panel title="Approvals" icon="shield" action={pendingApprovals ? `${pendingApprovals} waiting` : 'clear'}>
          {approvalRows.length === 0 ? (
            <EmptyState icon="check" title="No approvals waiting" detail="Tool calls, outbound connector actions, and ticket reviews will land here." />
          ) : (
            <div className="divide-y divide-line">
              {approvalGroups.map((group) => (
                <div key={group.id}>
                  <div className="flex items-center gap-2 bg-card-2 px-3 py-2">
                    <span className={clsx('h-1.5 w-1.5 rounded-full', toneDot(group.tone))} />
                    <div className="min-w-0 flex-1 truncate text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">{group.title}</div>
                    <span className="shrink-0 rounded bg-ink-5 px-1.5 py-0.5 text-[11px] font-medium text-ink-4">{group.detail}</span>
                  </div>
                  <div className="divide-y divide-line">
                    {group.rows.map((row) => <ApprovalRowItem key={row.id} row={row} />)}
                  </div>
                </div>
              ))}
            </div>
          )}
        </Panel>
      </div>

      <Panel title="Saved Policies" icon="shield" action={savedPolicies.length ? `${savedPolicies.length} scoped` : 'none'}>
        <div className="divide-y divide-line">
          {savedPolicies.length === 0 ? (
            <EmptyState icon="shield" title="No saved policies" detail="Using Allow scoped creates a policy for this project or workspace, valid for up to 8 hours or 20 uses." />
          ) : savedPolicies.map((rule) => (
            <div key={rule.id} className="px-3 py-3">
              <div className="flex items-start gap-3">
                <StatusIcon icon={approvalRuleIcon(rule)} tone={rule.decision === 'deny' ? 'critical' : 'good'} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-start gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-[13px] font-semibold text-ink">{rule.label || approvalRuleTitle(rule)}</div>
                      <div className="mt-0.5 flex flex-wrap items-center gap-1.5 text-[12px] text-ink-4">
                        <span className="rounded bg-ink-5 px-1.5 py-0.5">{rule.scope}</span>
                        <span className="rounded bg-ink-5 px-1.5 py-0.5">{rule.decision}</span>
                        <span className="rounded bg-ink-5 px-1.5 py-0.5">{rule.projectId ? `project ${rule.projectId}` : rule.workspace ? `workspace ${shortPath(rule.workspace)}` : 'legacy unscoped · inactive'}</span>
                        <span className="rounded bg-ink-5 px-1.5 py-0.5">{Number.isInteger(rule.maxUses) ? `${rule.useCount ?? 0}/${rule.maxUses} uses` : 'legacy unlimited · inactive'}</span>
                        <span className="rounded bg-ink-5 px-1.5 py-0.5">{Number.isFinite(rule.expiresAt) ? (Number(rule.expiresAt) <= Date.now() ? `expired ${relativeTime(Number(rule.expiresAt))}` : `expires in ${formatDuration(Number(rule.expiresAt) - Date.now())}`) : 'legacy no expiry · inactive'}</span>
                        <span>{rule.lastUsedAt ? `last used ${relativeTime(rule.lastUsedAt)}` : `saved ${relativeTime(rule.createdAt)}`}</span>
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="xs"
                      icon="trash"
                      onClick={() => void revokePolicy(rule)}
                      disabled={revokingRuleId === rule.id}
                    >
                      {revokingRuleId === rule.id ? 'Revoking' : 'Revoke'}
                    </Button>
                  </div>
                  <div className="mt-1.5 line-clamp-2 text-[13px] leading-snug text-ink-2">{rule.explanation || approvalRuleTitle(rule)}</div>
                  <div className="mt-1 truncate font-mono text-[12px] text-ink-4">{rule.key}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Panel>
    </>
  )
}
