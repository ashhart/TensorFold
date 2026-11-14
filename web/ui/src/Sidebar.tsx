import type { EngineInfo } from './api'
import type { ConversationIndex } from './conversations'
import type { Ctx } from './ctx'
import { ConversationList } from './screens/ChatList'
import { linkTo } from './route'

export function newChat() {
  location.hash = '#chat'
  dispatchEvent(new CustomEvent('tfs:new-conversation'))
}

export default function Sidebar({ engine, conversations, toast }: {
  engine: EngineInfo | null; conversations: ConversationIndex; toast: Ctx['toast']
}) {
  const { convId, convs, q, setQ, more, loadMore, refreshList } = conversations
  const status = engine?.status ?? 'unreachable'
  const name = engine?.model_name ?? engine?.model
  return (
    <aside className="workspace-sidebar" aria-label="Sidebar">
      <button className="sidebar-new" onClick={newChat}><span aria-hidden="true">＋</span> New chat</button>
      <ConversationList convs={convs} convId={convId} q={q} setQ={setQ} more={more} loadMore={loadMore}
        open={(id) => { location.hash = linkTo('chat', id) }} refresh={() => refreshList()} toast={toast}
        onDeleted={(id) => { if (id === convId) newChat() }} />
      <div className="sidebar-bottom">
        <a href="#monitor" className="engine-status"><span className={`dot ${status === 'running' ? '' : status === 'starting' ? 'warn' : 'off'}`} />
          <span>{status === 'running' ? `${name ?? 'Model'} ready` : status === 'starting' ? 'Loading model…' : status === 'unauthorized' ? 'Engine key refused' : status === 'failed' ? 'Engine stopped with a problem' : 'No model loaded'}</span>
        </a>
        <a href="#models"><span aria-hidden="true">◇</span> Models</a>
        <a href="#settings"><span aria-hidden="true">⚙</span> Settings</a>
      </div>
    </aside>
  )
}
