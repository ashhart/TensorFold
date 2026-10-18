import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { authoredText } from '../../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../../lib/store'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import clsx from 'clsx'
import { Icon } from '../../lib/icons'
import type { GitHygieneFinding, GitHygieneReport } from '../../lib/types'

const HYGIENE_TINT: Record<string, string> = { critical: 'text-danger', high: 'text-danger', medium: 'text-warn', low: 'text-ink-3' }

export function GitHygieneModal({
  report,
  pendingMessage,
  busy,
  error,
  onSanitize,
  onCommitAnyway,
  onClose,
}: {
  report: GitHygieneReport
  pendingMessage: string | null
  busy: boolean
  error: string | null
  onSanitize: () => void
  onCommitAnyway: () => void
  onClose: () => void
}) {
  const tx = useAuthoredText()
  const autoFixable = report.findings.filter((f) => f.autoFixable && f.source === 'working_tree').length
  const grouped = report.findings.reduce<Record<string, GitHygieneFinding[]>>((acc, finding) => {
    const key = `${finding.severity}:${finding.source}`
    ;(acc[key] ??= []).push(finding)
    return acc
  }, {})
  const currentBlockers = report.findings.filter((finding) => finding.source !== 'git_history' && ['critical', 'high', 'medium'].includes(finding.severity))
  const historyBlockers = report.findings.filter((finding) => finding.source === 'git_history' && ['critical', 'high', 'medium'].includes(finding.severity))
  const sourceLabel: Record<string, string> = { get working_tree() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Working tree") }, get git_history() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Git history") }, get commit_message() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Commit message") } }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/55 p-6">
      <div className="flex h-[min(760px,calc(100vh-56px))] w-[min(980px,calc(100vw-48px))] flex-col overflow-hidden rounded-2xl border border-line bg-card shadow-pop">
        <div className="flex items-center gap-3 border-b border-line px-4 py-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-warn/15 text-warn">
            <Icon name="shield" size={17} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-[13px] font-semibold text-ink">{tx("Git hygiene preflight")}</div>
            <div className="mt-0.5 flex flex-wrap items-center gap-2 font-mono text-[12px]">
              <span className="text-ink-3">{report.scannedFiles} {tx("files")}</span>
              <span className="text-ink-3">{report.scannedCommits} {tx("commits")}</span>
              <span className="text-danger">{report.summary.critical} {tx("critical")}</span>
              <span className="text-danger">{report.summary.high} {tx("high")}</span>
              <span className="text-warn">{report.summary.medium} {tx("medium")}</span>
              <span className="text-ink-3">{report.summary.low} {tx("low")}</span>
            </div>
          </div>
          <button onClick={onClose} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink" aria-label={tx("Close hygiene scan")}>
            <Icon name="x" size={15} />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-4 select-text">
          <div className={clsx('mb-3 rounded-xl border px-3.5 py-3 text-[13px] leading-snug', report.blocked ? 'border-danger/30 bg-danger/10 text-danger' : 'border-line bg-card-2 text-ink-2')}>
            {report.blocked
              ? tx("Resolve or handle current findings before committing: {count}.", { count: currentBlockers.length })
              : historyBlockers.length
                ? tx("History findings to clean up before publishing: {count}. Current files are clear for a local commit.", { count: historyBlockers.length })
                : report.findings.length
                ? tx("Only low current findings remain. Review them before committing.")
                : tx("No sensitive findings detected in the working tree, commit message, or Git history scan.")}
          </div>

          {report.notes.length > 0 && (
            <div className="mb-3 rounded-xl border border-line bg-card-2/50 px-3.5 py-2 text-[13px] text-ink-3">
              {report.notes.map((note) => <div key={note}>{note}</div>)}
            </div>
          )}

          {report.findings.length === 0 ? (
            <div className="rounded-xl border border-dashed border-line py-10 text-center text-[13px] text-ink-3">{tx("Clean scan.")}</div>
          ) : (
            <div className="space-y-3">
              {Object.entries(grouped).map(([key, findings]) => {
                const [severity, source] = key.split(':')
                return (
                  <div key={key} className="overflow-hidden rounded-xl border border-line bg-card-2/40">
                    <div className="flex items-center justify-between border-b border-line px-3.5 py-2">
                      <div className="text-[13px] font-semibold text-ink">{sourceLabel[source] ?? source}</div>
                      <div className={clsx('text-[12px] font-semibold uppercase tracking-[0.05em]', HYGIENE_TINT[severity])}>{severity} · {findings.length}</div>
                    </div>
                    <div className="divide-y divide-line">
                      {findings.slice(0, 80).map((finding) => (
                        <div key={finding.id} className="px-3.5 py-2.5">
                          <div className="flex items-center gap-2 text-[13px]">
                            <span className={clsx('font-semibold', HYGIENE_TINT[finding.severity])}>{finding.kind.replace(/_/g, ' ')}</span>
                            {finding.file && <span className="truncate text-ink-2">{finding.file}{finding.line ? `:${finding.line}` : ''}</span>}
                            {finding.commit && <span className="font-mono text-ink-4">{finding.commit.slice(0, 8)}</span>}
                            {finding.autoFixable && <span className="rounded bg-ink-5 px-1.5 py-0.5 text-[11px] text-ink-3">{tx("auto")}</span>}
                          </div>
                          <div className="mt-1 font-mono text-[12px] leading-snug text-ink-3">{finding.excerpt || finding.match}</div>
                          <div className="mt-1 text-[13px] leading-snug text-ink-3">{finding.suggestion}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {error && <div className="border-t border-line px-4 py-2 text-[13px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</div>}
        <div className="flex items-center justify-between gap-3 border-t border-line px-4 py-3">
          <div className="text-[13px] text-ink-3">
            {pendingMessage ? tx("Pending commit: {value0}", { value0: pendingMessage }) : tx("Manual scan only")}
          </div>
          <div className="flex gap-2">
            <button onClick={onSanitize} disabled={busy || autoFixable === 0} className="rounded-lg border border-line px-3 py-1.5 text-[13px] font-medium text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink disabled:opacity-40">
              {busy ? tx("Working…") : tx("Sanitize obvious{value0}", { value0: autoFixable ? ` (${autoFixable})` : '' })}
            </button>
            {pendingMessage && (
              <button onClick={onCommitAnyway} disabled={busy || report.blocked} className="rounded-lg bg-accent px-3 py-1.5 text-[13px] font-medium text-accent-ink transition-colors hover:bg-accent-strong disabled:opacity-40">
                {tx("Commit")}</button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
