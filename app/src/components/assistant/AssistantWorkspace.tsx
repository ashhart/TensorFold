import { useRef, useState, type ReactNode } from 'react'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { GoogleWorkspaceConnector } from '../settings/connectors/GoogleWorkspaceConnector'
import { useGoogleWorkspace } from '../settings/connectors/google/useGoogleWorkspace'
import { AssistantConnections } from './AssistantConnections'
import { AssistantOverview } from './AssistantOverview'

const panels = ['overview', 'inbox', 'google', 'connections'] as const
type Panel = typeof panels[number]
const labels = { overview: 'Overview', inbox: 'Inbox', google: 'Google Workspace', connections: 'Connections' }

export function AssistantWorkspace({ children }: { children: ReactNode }) {
  const tx = useAuthoredText()
  const google = useGoogleWorkspace()
  const [panel, setPanel] = useState<Panel>('inbox')
  const [visited, setVisited] = useState<Partial<Record<Panel, boolean>>>({ inbox: true })
  const buttons = useRef<Array<HTMLButtonElement | null>>([])
  const show = (next: Panel) => { setVisited(old => ({ ...old, [next]: true })); setPanel(next) }
  const accounts = google.snapshot?.accounts ?? []
  const reconnects = accounts.filter(account => account.status === 'needs-reauthorization').length
  const reviews = google.snapshot?.reviews.length ?? 0
  return <div className="flex h-full min-h-0 flex-col bg-canvas">
    <div role="tablist" aria-label={tx('Assistant workspace')} className="flex shrink-0 items-center gap-1 overflow-x-auto border-b border-line bg-card px-3 py-1.5">
      {panels.map((id, index) => <button key={id} ref={node => { buttons.current[index] = node }}
        id={`assistant-tab-${id}`} role="tab" aria-selected={panel === id} aria-controls={`assistant-panel-${id}`}
        aria-label={tx(labels[id])} tabIndex={panel === id ? 0 : -1}
        onClick={() => show(id)} onKeyDown={event => {
          const next = event.key === 'Home' ? 0 : event.key === 'End' ? panels.length - 1
            : event.key === 'ArrowRight' ? (index + 1) % panels.length
              : event.key === 'ArrowLeft' ? (index + panels.length - 1) % panels.length : null
          if (next === null) return
          event.preventDefault(); show(panels[next]); buttons.current[next]?.focus()
        }} className={`shrink-0 rounded-lg px-3 py-1.5 text-[13px] focus-visible:outline-2 focus-visible:outline-accent ${panel === id ? 'bg-accent/10 font-medium text-accent' : 'text-ink-2 hover:bg-surface'}`}>
        {tx(labels[id])}{id === 'google' && (reconnects || reviews) ? <span aria-hidden="true" className="ml-2 rounded bg-warn/15 px-1.5 text-[11px] text-warn">{reconnects + reviews}</span> : null}
      </button>)}
    </div>
    <div id="assistant-panel-overview" role="tabpanel" aria-labelledby="assistant-tab-overview" hidden={panel !== 'overview'} className={panel === 'overview' ? 'min-h-0 flex-1 overflow-y-auto' : 'hidden'}>
      {visited.overview && <AssistantOverview google={google} onGoogle={() => show('google')} onInbox={() => show('inbox')} />}
    </div>
    <div id="assistant-panel-inbox" role="tabpanel" aria-labelledby="assistant-tab-inbox" hidden={panel !== 'inbox'} className={panel === 'inbox' ? 'min-h-0 flex-1' : 'hidden'}>{children}</div>
    <div id="assistant-panel-connections" role="tabpanel" aria-labelledby="assistant-tab-connections" hidden={panel !== 'connections'} className={panel === 'connections' ? 'min-h-0 flex-1 overflow-y-auto' : 'hidden'}>
      {visited.connections && <AssistantConnections onGoogle={() => show('google')} />}
    </div>
    <div id="assistant-panel-google" role="tabpanel" aria-labelledby="assistant-tab-google" hidden={panel !== 'google'} className={panel === 'google' ? 'min-h-0 flex-1 overflow-y-auto p-3 sm:p-5' : 'hidden'}>
      {visited.google && <div className="mx-auto max-w-[1000px]"><GoogleWorkspaceConnector googleState={google} compact /></div>}
    </div>
  </div>
}
