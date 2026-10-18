import { useAuthoredText } from '../../../lib/preferences/useAuthoredText'
import type { Automation } from '../../../lib/types'
import { Icon } from '../../../lib/icons'
import { Button } from '../../ui'
import { Panel } from '../ui'
import { briefSchedule, scheduleTime } from '../brief'
import type { CommandCentreSetView } from '../model'

interface DailyBriefPanelProps {
  primaryBrief: Automation | null
  runAutomation: (id: string) => Promise<void>
  setView: CommandCentreSetView
  dailyBriefLines: string[]
  briefAutomations: Automation[]
}

export function DailyBriefPanel(props: DailyBriefPanelProps) {
  const tx = useAuthoredText()
  const { primaryBrief, runAutomation, setView, dailyBriefLines, briefAutomations } = props

  return (
    <Panel title={tx("Daily Brief")} icon="sun" action={primaryBrief?.enabled ? 'enabled' : 'manual'}>
      <div className="px-3 py-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="truncate text-[13px] font-semibold text-ink">{primaryBrief?.name ?? tx("No daily brief configured")}</div>
            <div className="mt-0.5 text-[13px] text-ink-3">
              {primaryBrief ? briefSchedule(primaryBrief) : tx("Create one from Automations")}
            </div>
          </div>
          <Button variant="default" size="xs" icon="play" onClick={() => primaryBrief ? void runAutomation(primaryBrief.id) : setView({ name: 'automations' })} disabled={primaryBrief?.runs[0]?.status === 'running'}>
            {tx("Run")}</Button>
        </div>

        <div className="mt-3 space-y-2">
          {dailyBriefLines.length ? dailyBriefLines.map((line, index) => (
            <div key={`${line}-${index}`} className="flex gap-2 text-[13px] leading-snug text-ink-2">
              <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-ink-4" />
              <span className="line-clamp-2">{line}</span>
            </div>
          )) : (
            <div className="rounded-lg border border-line bg-card-2 px-3 py-2.5 text-[13px] leading-snug text-ink-3">
              {tx("Waiting for the first brief run.")}</div>
          )}
        </div>

        {briefAutomations.length > 1 && (
          <div className="mt-3 grid grid-cols-1 gap-1.5">
            {briefAutomations.slice(1).map((automation) => (
              <button key={automation.id} onClick={() => void runAutomation(automation.id)} className="flex items-center gap-2 rounded-lg border border-line bg-card-2 px-2.5 py-2 text-left text-[13px] text-ink-2 transition-colors hover:border-line-strong hover:text-ink">
                <Icon name="clock" size={13} className="text-ink-3" />
                <span className="min-w-0 flex-1 truncate">{automation.name}</span>
                <span className="shrink-0 text-ink-4">{automation.nextRunAt ? scheduleTime(automation.nextRunAt) : tx("manual")}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </Panel>
  )
}
