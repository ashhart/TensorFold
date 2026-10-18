import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useState } from 'react'
import clsx from 'clsx'
import { useStore } from '../../lib/store'
import { Icon } from '../../lib/icons'
import { basename } from '../../lib/format'
import { api } from '../../lib/ipc'
import type { GitHygieneReport, GitStatus, Thread, WorkspaceLocation } from '../../lib/types'
import { PanelSection } from './primitives'
import { GitHygieneModal } from './GitHygieneModal'

type WorktreeRow = { path: string; head: string | null; branch: string | null; detached: boolean; bare: boolean }

export function GitToolsSection({
  thread,
  cwd,
  workspace,
  remote,
  git,
  onChanged,
}: {
  thread: Thread
  cwd: string
  workspace: WorkspaceLocation | null
  remote: boolean
  git: GitStatus | null
  onChanged: () => void
}) {
  const tx = useAuthoredText()
  const [committing, setCommitting] = useState(false)
  const [hygieneOpen, setHygieneOpen] = useState(false)
  const [hygieneBusy, setHygieneBusy] = useState(false)
  const [hygieneError, setHygieneError] = useState<string | null>(null)
  const [hygieneReport, setHygieneReport] = useState<GitHygieneReport | null>(null)
  const [pendingCommitMessage, setPendingCommitMessage] = useState<string | null>(null)
  const setThreadWorkspace = useStore((s) => s.setThreadWorkspace)
  const [wtOpen, setWtOpen] = useState(false)
  const [wt, setWt] = useState<{ branches: string[]; worktrees: WorktreeRow[]; current: string | null } | null>(null)
  const [wtBusy, setWtBusy] = useState(false)
  const [wtError, setWtError] = useState<string | null>(null)
  const [featureName, setFeatureName] = useState('')

  useEffect(() => {
    if (!wtOpen || !cwd) return
    let alive = true
    setWtBusy(true)
    api.getWorktrees({ cwd, workspace })
      .then((d) => { if (alive) setWt(d) })
      .catch(() => { if (alive) setWt(null) })
      .finally(() => { if (alive) setWtBusy(false) })
    return () => { alive = false }
  }, [wtOpen, cwd, workspace])

  const reloadWorktrees = () => {
    if (!cwd) return
    api.getWorktrees({ cwd, workspace }).then(setWt).catch(() => {})
  }

  const switchBranch = async (branch: string) => {
    if (!cwd || wtBusy || branch === (wt?.current ?? git?.branch)) return
    setWtBusy(true); setWtError(null)
    try {
      const r = await api.checkoutBranch({ cwd, workspace, branch })
      if (!r.ok) { setWtError(r.error || tx("Could not switch branch")); return }
      onChanged(); reloadWorktrees()
    } catch (e: any) { setWtError(e?.message || tx("Could not switch branch")) }
    finally { setWtBusy(false) }
  }

  const switchWorktree = async (target: string) => {
    if (!target || target === cwd) return
    await setThreadWorkspace(thread.id, null, target)
    setWtOpen(false)
    onChanged()
  }

  const createFeature = async (mode: 'worktree' | 'branch') => {
    const name = featureName.trim()
    if (!name || !cwd || wtBusy) return
    setWtBusy(true); setWtError(null)
    try {
      const r = await api.createWorktree({ cwd, workspace, name, mode })
      if (!r.ok) { setWtError(r.error || tx("Could not create feature")); return }
      setFeatureName('')
      if (mode === 'worktree' && r.path) {
        await setThreadWorkspace(thread.id, null, r.path)
        setWtOpen(false)
      }
      onChanged(); reloadWorktrees()
    } catch (e: any) { setWtError(e?.message || tx("Could not create feature")) }
    finally { setWtBusy(false) }
  }

  const runHygieneScan = async (message: string | null = null) => {
    if (!cwd || hygieneBusy) return null
    setHygieneBusy(true)
    setHygieneError(null)
    try {
      const report = await api.scanGitHygiene(cwd, message ?? '', workspace)
      setHygieneReport(report)
      setPendingCommitMessage(message)
      setHygieneOpen(true)
      return report
    } catch (e: any) {
      setHygieneError(e?.message ?? tx("Git hygiene scan failed."))
      setHygieneOpen(true)
      return null
    } finally {
      setHygieneBusy(false)
    }
  }

  const commitWithMessage = async (message: string) => {
    setCommitting(true)
    setHygieneError(null)
    try {
      await api.commit(cwd, message, workspace)
      setHygieneOpen(false)
      setPendingCommitMessage(null)
      setHygieneReport(null)
      onChanged()
    } catch (e: any) {
      setHygieneError(e?.message ?? tx("Commit failed."))
    } finally {
      setCommitting(false)
    }
  }

  const commit = async () => {
    if (committing || !git || git.files.length === 0 || remote) return
    const msg = prompt(tx("Commit message"), thread.goal || thread.title)
    if (!msg) return
    const report = await runHygieneScan(msg)
    if (!report || report.findings.length > 0) return
    await commitWithMessage(msg)
  }

  const sanitizeFindings = async () => {
    if (!hygieneReport || hygieneBusy) return
    setHygieneBusy(true)
    setHygieneError(null)
    try {
      await api.sanitizeGitHygiene(cwd, hygieneReport.findings, workspace)
      onChanged()
      const report = await api.scanGitHygiene(cwd, pendingCommitMessage ?? '', workspace)
      setHygieneReport(report)
    } catch (e: any) {
      setHygieneError(e?.message ?? tx("Could not sanitize findings."))
    } finally {
      setHygieneBusy(false)
    }
  }

  return (
    <>
      <div className="mx-4 border-t border-line" />
      <PanelSection title={tx("Git tools")}>
        <div className="space-y-1">
          <div className="flex items-center gap-2.5 rounded-lg px-2 py-2 text-[13px]">
            <Icon name="file" size={15} className="text-ink-3" />
            <span className="flex-1 font-medium text-ink-2">{tx("Changes")}</span>
            {git && (git.files.length > 0
              ? <span className="flex items-center gap-1.5 font-mono text-[13px]"><span className="text-positive">+{git.additions}</span><span className="text-danger">−{git.deletions}</span></span>
              : <span className="text-[13px] text-ink-4">{tx("clean")}</span>)}
          </div>
          <button onClick={() => setWtOpen((v) => !v)} className="flex w-full items-center gap-2.5 rounded-lg px-2 py-2 text-left text-[13px] transition-colors hover:bg-ink-5">
            <Icon name="branch" size={15} className="text-ink-3" />
            <span className="flex-1 truncate font-medium text-ink-2">{git?.branch || wt?.current || tx("no branch")}</span>
            <Icon name="chevronDown" size={13} className={clsx('text-ink-4 transition-transform', wtOpen && 'rotate-180')} />
          </button>
          {wtOpen && (
            <div className="space-y-2 rounded-lg border border-line bg-card-2 p-2">
              {wtError && <div className="rounded-md bg-danger/10 px-2 py-1 text-[12px] leading-snug text-danger">{wtError}</div>}
              <div>
                <div className="mb-1 px-1 text-[12px] font-semibold uppercase tracking-wide text-ink-4">{tx("Branches")}</div>
                {wt?.branches.length ? (
                  <div className="scroll-thin max-h-40 space-y-0.5 overflow-y-auto">
                    {wt.branches.map((b) => {
                      const active = b === (wt.current ?? git?.branch)
                      return (
                        <button key={b} onClick={() => switchBranch(b)} disabled={wtBusy || active} className={clsx('flex w-full items-center gap-1.5 rounded-md px-1.5 py-1 text-left text-[13px] transition-colors disabled:cursor-default', active ? 'text-ink' : 'text-ink-2 hover:bg-ink-5')}>
                          <Icon name={active ? 'check' : 'branch'} size={12} className={active ? 'text-accent' : 'text-ink-4'} />
                          <span className="flex-1 truncate">{b}</span>
                        </button>
                      )
                    })}
                  </div>
                ) : <div className="px-1.5 py-1 text-[12px] text-ink-4">{wtBusy ? tx("Loading…") : tx("No branches")}</div>}
              </div>
              {wt && wt.worktrees.length > 1 && (
                <div>
                  <div className="mb-1 px-1 text-[12px] font-semibold uppercase tracking-wide text-ink-4">{tx("Worktrees")}</div>
                  <div className="scroll-thin max-h-40 space-y-0.5 overflow-y-auto">
                    {wt.worktrees.map((w) => {
                      const here = w.path === cwd
                      return (
                        <button key={w.path} onClick={() => switchWorktree(w.path)} disabled={here} title={w.path} className={clsx('flex w-full items-center gap-1.5 rounded-md px-1.5 py-1 text-left text-[13px] transition-colors disabled:cursor-default', here ? 'text-ink' : 'text-ink-2 hover:bg-ink-5')}>
                          <Icon name={here ? 'check' : 'folder'} size={12} className={here ? 'text-accent' : 'text-ink-4'} />
                          <span className="flex-1 truncate">{w.branch || (w.detached ? tx("detached") : basename(w.path))}</span>
                          {here && <span className="text-[11px] text-ink-4">{tx("here")}</span>}
                        </button>
                      )
                    })}
                  </div>
                </div>
              )}
              <div className="border-t border-line pt-2">
                <div className="mb-1 px-1 text-[12px] font-semibold uppercase tracking-wide text-ink-4">{tx("New feature")}</div>
                <input value={featureName} onChange={(e) => setFeatureName(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && featureName.trim() && !remote) createFeature('worktree') }} placeholder={tx("feature name")} className="mb-1.5 w-full rounded-md border border-line bg-card px-2 py-1 text-[13px] text-ink placeholder:text-ink-4 focus:border-ink-3 focus:outline-none" />
                <div className="flex gap-1.5">
                  <button onClick={() => createFeature('worktree')} disabled={!featureName.trim() || wtBusy || remote} title={remote ? tx("Worktrees are local-only for now — use Branch here.") : tx("Create an isolated worktree (separate folder) and switch this chat to it.")} className="flex flex-1 items-center justify-center gap-1 rounded-md bg-ink px-2 py-1.5 text-[13px] font-medium text-bg transition-opacity hover:opacity-90 disabled:opacity-40">
                    <Icon name="plus" size={12} /> {tx("Create worktree")}</button>
                  <button onClick={() => createFeature('branch')} disabled={!featureName.trim() || wtBusy} title={tx("Create the branch here and switch to it in place.")} className="flex flex-1 items-center justify-center gap-1 rounded-md border border-line px-2 py-1.5 text-[13px] font-medium text-ink-2 transition-colors hover:bg-ink-5 disabled:opacity-40">
                    <Icon name="branch" size={12} /> {tx("Branch here")}</button>
                </div>
              </div>
            </div>
          )}
          <button onClick={commit} disabled={!git || git.files.length === 0 || committing || remote} title={remote ? tx("Commit runs on the local repo — use the agent for remote workspaces.") : undefined} className="flex w-full items-center gap-2.5 rounded-lg px-2 py-2 text-left text-[13px] font-medium text-ink-2 transition-colors hover:bg-ink-5 disabled:opacity-40 disabled:hover:bg-transparent">
            <Icon name="check" size={15} className="text-ink-3" />
            <span className="flex-1">{committing ? tx("Committing…") : tx("Commit")}</span>
            <Icon name="dots" size={14} className="text-ink-4" />
          </button>
          <button onClick={() => runHygieneScan(null)} disabled={!cwd || hygieneBusy || remote} title={remote ? tx("Hygiene scans the local repo — use the agent for remote workspaces.") : undefined} className="flex w-full items-center gap-2.5 rounded-lg px-2 py-2 text-left text-[13px] font-medium text-ink-2 transition-colors hover:bg-ink-5 disabled:opacity-40 disabled:hover:bg-transparent">
            <Icon name="shield" size={15} className="text-ink-3" />
            <span className="flex-1">{hygieneBusy ? tx("Scanning…") : tx("Git hygiene scan")}</span>
            <Icon name="dots" size={14} className="text-ink-4" />
          </button>
        </div>
      </PanelSection>
      {hygieneOpen && (hygieneReport ? (
          <GitHygieneModal
            report={hygieneReport}
            pendingMessage={pendingCommitMessage}
            busy={hygieneBusy || committing}
            error={hygieneError}
            onSanitize={sanitizeFindings}
            onCommitAnyway={() => pendingCommitMessage && commitWithMessage(pendingCommitMessage)}
            onClose={() => setHygieneOpen(false)}
          />
        ) : (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/55 p-6">
            <div className="w-[min(460px,calc(100vw-48px))] rounded-2xl border border-line bg-card p-4 shadow-pop">
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-danger/15 text-danger"><Icon name="shield" size={17} /></div>
                <div className="min-w-0 flex-1">
                  <div className="text-[13px] font-semibold text-ink">{tx("Git hygiene scan failed")}</div>
                  <div className="mt-1 text-[13px] leading-snug text-ink-3">{hygieneError || tx("The scan could not complete.")}</div>
                </div>
              </div>
              <div className="mt-4 flex justify-end">
                <button onClick={() => setHygieneOpen(false)} className="rounded-lg border border-line bg-card-2 px-3 py-1.5 text-[13px] font-medium text-ink-2 transition-colors hover:bg-ink-5">{tx("Close")}</button>
              </div>
            </div>
          </div>
        ))}
    </>
  )
}
