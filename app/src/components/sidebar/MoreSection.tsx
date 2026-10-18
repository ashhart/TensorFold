import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import type { View } from '../../lib/store'
import type { Settings } from '../../lib/types'
import { AssistantSourceRow, NavRow, SectionTitle } from './primitives'
import { SOURCE_LABEL, connectedSources, sourceDetail } from './source-detail'

// The surfaces that used to be separate sidebar modes, kept as one compact group. The Assistant
// row expands to its connected sources while the Assistant is open (they filter its inbox).
export function MoreSection({ view, open, onToggle, settings, assistantSource, needsMe, reviewTicketCount, runningAutomations, pickSource, setView }: {
  view: View
  open: boolean
  onToggle: () => void
  settings: Settings | null
  assistantSource: string
  needsMe: number
  reviewTicketCount: number
  runningAutomations: number
  pickSource: (id: string) => void
  setView: (view: View) => void
}) {
  const tx = useAuthoredText()
  const inAssistant = view.name === 'assistant'
  const sources = inAssistant ? connectedSources(settings) : []
  return (
    <>
      <SectionTitle onClick={onToggle} open={open}>{tx("More")}</SectionTitle>
      {open && (
        <div>
          <NavRow icon="mail" label={tx("Assistant")} active={inAssistant && assistantSource === 'all'} badge={needsMe ? String(needsMe) : undefined} onClick={() => pickSource('all')} />
          {inAssistant && sources.map((id) => (
            <AssistantSourceRow key={id} id={id} label={SOURCE_LABEL[id]} detail={sourceDetail(settings, id)} active={assistantSource === id} onClick={() => pickSource(id)} />
          ))}
          <NavRow icon="layers" label={tx("Rooms")} active={view.name === 'projects'} onClick={() => setView({ name: 'projects' })} />
          <NavRow icon="clock" label={tx("Automations")} active={view.name === 'automations'} badge={runningAutomations ? String(runningAutomations) : undefined} onClick={() => setView({ name: 'automations' })} />
          <NavRow icon="plug" label={tx("Plugins")} active={view.name === 'plugins'} onClick={() => setView({ name: 'plugins' })} />
          <NavRow icon="bell" label={tx("Command Centre")} active={view.name === 'commandCentre'} badge={reviewTicketCount ? String(reviewTicketCount) : undefined} onClick={() => setView({ name: 'commandCentre' })} />
        </div>
      )}
    </>
  )
}
