import { useEffect, useState } from 'react'
import { gb, num } from './api'
import type { Ctx } from './ctx'
import { ModelLogo } from './brandLogo'
import { modelTab } from './navigation'
import Models from './screens/Models'
import Teach from './screens/Teach'
import Connect from './screens/Connect'

const TABS = [ ['mine', 'My models', '#models'], ['discover', 'Discover', '#models/discover'], ['learning', 'Learning', '#models/learning'], ['api', 'API', '#models/api'] ] as const

export default function ModelWorkspace(ctx: Ctx) {
  const [tab, setTab] = useState(() => modelTab(location.hash))
  useEffect(() => {
    const on = () => { if (!location.hash.startsWith('#fleet')) setTab(modelTab(location.hash)) }
    addEventListener('hashchange', on)
    return () => removeEventListener('hashchange', on)
  }, [])
  const engine = ctx.engine
  const ready = engine?.status === 'running'
  const name = engine?.model_name ?? engine?.model
  return <section className="model-workspace">
    <div className="head"><div><h1>Models</h1><p>Choose a model and get started.</p></div>
      <div className="actions"><a className="btn" href="#fleet">Compute and clusters</a><a className="btn" href="#settings/storage">Model storage</a><a className="btn" href="#models/discover">Downloads</a></div>
    </div>
    <div className="card engine-card">
      <div className="engine-card-head"><span className="engine-mark"><ModelLogo names={[name, engine?.model]} size="lg" /></span>
        <div><div className="small muted">Engine</div><h2>{name ?? 'No model loaded'}</h2><div className="small muted">{ready ? 'Ready for conversations' : engine?.status === 'starting' ? 'Loading model…' : engine?.status === 'unauthorized' ? 'API key refused' : engine?.status === 'failed' ? 'Stopped with a problem' : 'Choose a model below'}</div></div>
        <a className="btn" href="#monitor">Details</a>
      </div>
      <div className="engine-facts"><div><span>State</span><b>{engine?.status ?? 'Connecting'}</b></div><div><span>Memory</span><b>{engine?.memory?.active != null ? gb(engine.memory.active) : '—'}</b></div><div><span>Context</span><b>{engine?.context_length ? `${num(engine.context_length)} tokens` : '—'}</b></div></div>
    </div>
    <nav className="model-tabs" aria-label="Models"><div>{TABS.map(([id, name, href]) => <a key={id} href={href} aria-current={tab === id ? 'page' : undefined}>{name}</a>)}</div></nav>
    {tab === 'mine' || tab === 'discover' ? <Models {...ctx} discover={tab === 'discover'} /> : tab === 'learning' ? <Teach {...ctx} /> : <Connect {...ctx} />}
  </section>
}
