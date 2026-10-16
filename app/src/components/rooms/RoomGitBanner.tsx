import { useEffect, useState } from 'react'
import { api } from '../../lib/ipc'
import { Icon } from '../../lib/icons'
import { Button } from '../ui'
import type { GitRepositoryState, Project } from '../../lib/types'

export function RoomGitBanner({ project, running, onReady }: { project: Project | null | undefined; running: boolean; onReady: () => void }) {
  const [state, setState] = useState<GitRepositoryState | null>(null)
  const [busy, setBusy] = useState(false)
  const [initBusy, setInitBusy] = useState(false)
  const [error, setError] = useState('')
  const remoteProject = project?.executionTarget === 'orchestrator'

  useEffect(() => {
    let live = true
    setState(null)
    setError('')
    if (!project?.cwd || remoteProject) return
    setBusy(true)
    api.inspectGitRepository(project.cwd)
      .then((next) => { if (live) setState(next) })
      .catch((e) => { if (live) setError(e?.message ?? 'Could not inspect Git repository.') })
      .finally(() => { if (live) setBusy(false) })
    return () => { live = false }
  }, [project?.cwd, remoteProject])

  if (!project?.cwd || remoteProject || (state?.exists && state.hasCommits)) return null

  const initialize = async () => {
    if (!project.cwd || initBusy) return
    setInitBusy(true)
    setError('')
    try {
      const next = await api.initGitRepository(project.cwd)
      setState(next)
      if (next.exists && next.hasCommits) onReady()
    } catch (e: any) {
      setError(e?.message ?? 'Could not initialize Git.')
    } finally {
      setInitBusy(false)
    }
  }

  return (
    <div className="rounded-xl border border-warn/35 bg-warn/10 px-3.5 py-3">
      <div className="flex items-start gap-2.5">
        <Icon name={busy ? 'clock' : 'warning'} size={16} className="mt-0.5 shrink-0 text-warn" />
        <div className="min-w-0 flex-1">
          <div className="text-[13px] font-medium text-ink">{busy ? 'Checking Git...' : state?.exists ? 'Initial commit needed' : 'Initialize this repo'}</div>
          <div className="mt-0.5 text-[13px] leading-snug text-ink-3">
            Rooms use isolated Git worktrees. This folder needs Git and an initial commit before the room can start.
          </div>
          {error && <div className="mt-2 rounded-lg border border-danger/30 bg-danger/10 px-2.5 py-1.5 text-[13px] text-danger">{error}</div>}
        </div>
        <Button size="sm" variant="primary" icon="branch" onClick={initialize} disabled={running || busy || initBusy}>
          {initBusy ? 'Initializing...' : 'Initialize Git'}
        </Button>
      </div>
    </div>
  )
}
