import { useAuthoredText } from '../../../lib/preferences/useAuthoredText'
import type { DoctorSummary } from '../../../lib/readiness-doctor'
import type { Automation } from '../../../lib/types'
import { CommandStatusSummary, Metric } from '../ui'
import { scheduleTime } from '../brief'
import type {
  AttentionRow,
  CommandStatusSummary as CommandStatusSummaryModel,
  ConnectorHealthSignal,
  LiveJob,
  RunningWorkSummary,
  Tone,
} from '../model'

interface StatusMetricsProps {
  statusSummary: CommandStatusSummaryModel
  statusExpanded: boolean
  toggleStatusExpanded: () => void
  attentionRows: AttentionRow[]
  unreadSignals: number
  pendingApprovals: number
  liveJobs: LiveJob[]
  runningWorkSummary: RunningWorkSummary
  connectedCount: number
  connectorRows: { length: number }
  connectorActionCount: number
  gapCount: number
  connectorHealthRows: Array<{ connector: unknown; health: ConnectorHealthSignal }>
  nextBrief: number | null
  primaryBrief: Automation | null
  missionRows: { length: number }
  activeMissions: number
  readinessSummary: DoctorSummary
  readinessTone: Tone
}

export function StatusMetrics(props: StatusMetricsProps) {
  const tx = useAuthoredText()
  const {
    statusSummary,
    statusExpanded,
    toggleStatusExpanded,
    attentionRows,
    unreadSignals,
    pendingApprovals,
    liveJobs,
    runningWorkSummary,
    connectedCount,
    connectorRows,
    connectorActionCount,
    gapCount,
    connectorHealthRows,
    nextBrief,
    primaryBrief,
    missionRows,
    activeMissions,
    readinessSummary,
    readinessTone,
  } = props

  return (
    <CommandStatusSummary summary={statusSummary} expanded={statusExpanded} onToggle={toggleStatusExpanded}>
      <div className="grid grid-cols-2 gap-2 lg:grid-cols-4 min-[1500px]:grid-cols-7">
        <Metric icon="bell" label={tx("Attention")} value={String(attentionRows.length)} detail={`${unreadSignals} unread signals`} tone={attentionRows.some((r) => r.tone === 'critical') ? 'critical' : 'normal'} />
        <Metric icon="shield" label={tx("Approvals")} value={String(pendingApprovals)} detail={pendingApprovals ? 'Waiting on you' : 'Queue clear'} tone={pendingApprovals ? 'high' : 'good'} />
        <Metric icon="terminal" label={tx("Live Jobs")} value={String(liveJobs.length)} detail={`${runningWorkSummary.statusCounts.queued} queued / ${runningWorkSummary.statusCounts.stalled + runningWorkSummary.statusCounts.failed + runningWorkSummary.statusCounts.dead_letter} blocked`} tone={runningWorkSummary.statusCounts.failed || runningWorkSummary.statusCounts.dead_letter ? 'critical' : runningWorkSummary.statusCounts.stalled ? 'high' : liveJobs.length ? 'normal' : 'good'} />
        <Metric icon="plug" label={tx("Connectors")} value={`${connectedCount}/${connectorRows.length || 0}`} detail={connectorActionCount ? `${connectorActionCount} actions` : `${gapCount} gaps`} tone={connectorHealthRows.some((row) => row.health.tone === 'critical') ? 'critical' : connectorActionCount ? 'high' : gapCount ? 'high' : 'good'} />
        <Metric icon="clock" label={tx("Next Brief")} value={nextBrief ? scheduleTime(nextBrief) : 'None'} detail={primaryBrief?.name ?? 'No daily run'} tone={nextBrief ? 'normal' : 'low'} />
        <Metric icon="spark" label={tx("Missions")} value={String(activeMissions)} detail={`${missionRows.length} total`} tone={activeMissions ? 'normal' : 'low'} />
        <Metric icon="check" label={tx("Readiness")} value={readinessSummary.error ? 'Blocked' : readinessSummary.warning ? 'Review' : readinessSummary.checking ? 'Checking' : 'Ready'} detail={`${readinessSummary.ok}/${readinessSummary.total || 0} ready`} tone={readinessTone} />
      </div>
    </CommandStatusSummary>
  )
}
