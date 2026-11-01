import { useEffect, useState, type ReactNode } from 'react'
import { api, gb, type ConversationRow, type FleetMachine, type LivingView, type Model } from '../api'
import type { Ctx } from '../ctx'
import { linkTo } from '../route'
import { ago } from './MonitorUsage'
import { FirstRun } from './SetupChecks'
import { HardwareLogo, ModelLogo } from '../brandLogo'
import { hardwareOf } from '../brand'
import { whyNotTeach } from '../teachable'

const ICON = {
  chat: 'M4 5h16v11H9l-5 4z',
  teach: 'M12 3l10 5-10 5L2 8zM6 10.5v5c0 1.5 3 3 6 3s6-1.5 6-3v-5M21 8v6',
  apps: 'M10 14a4 4 0 0 0 5.6 0l3-3a4 4 0 0 0-5.6-5.6L11.5 7M14 10a4 4 0 0 0-5.6 0l-3 3a4 4 0 0 0 5.6 5.6l1.5-1.6',
  bench: 'M4.9 19a10 10 0 1 1 14.2 0M12 14l4-5M12 14a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3',
  monitor: 'M3 12h4l3-8 4 16 3-8h4',
  fleet: 'M3 6h7v6H3zM14 6h7v6h-7zM8.5 14h7v6h-7zM6.5 12v2M17.5 12v2M12 12v2',
}

/** Home: what the AI is doing, in words, the one action for that state, and the three things a person came for. */
export default function Home({ machine, engine, settings, saveSettings, refreshEngine, refreshSettings, toast, stale }: Ctx) {
  const [models, setModels] = useState<Model[] | null>(null)
  const [catalogErr, setCatalogErr] = useState<string | null>(null)   // said, rather than "Reading…" for ever
  const [convs, setConvs] = useState<ConversationRow[]>([])
  const [living, setLiving] = useState<LivingView | null>(null)
  const [busy, setBusy] = useState(false)
  const [fleet, setFleet] = useState<FleetMachine[]>([])
  useEffect(() => {
    api.catalog().then((c) => { setModels(c.models); setCatalogErr(null) }).catch((e) => setCatalogErr(String((e as Error).message ?? e)))
    api.conversations().then((r) => setConvs(r.conversations.slice(0, 5))).catch(() => {})
    api.living().then(setLiving).catch(() => {})
  }, [engine?.status, engine?.model])
  // a remote engine's machine is described by its own TensorFold Web, when Fleet knows it
  const remote = engine?.managed === false
  useEffect(() => { if (remote) api.fleet().then((f) => setFleet(f.machines)).catch(() => {}) }, [remote])

  const st = engine?.status ?? 'unreachable'
  const name = engine?.model_name ?? engine?.model ?? 'the AI'
  const running = st === 'running'
  // the machines that run the AI: this one, or the remote engine's, by its Fleet entry or else its host's short name;
  // a list, so that an engine spread over several machines can name each and show each maker's logo
  const host = remote ? shortHost(engine?.engine_url) : machine?.hostname || null
  const peer = remote && host ? fleet.find((m) => m.reachable && m.hostname && shortHost(`http://${m.hostname}`) === host) : null
  const rigs: Rig[] = !host ? [] : !remote ? [{ host, chip: machine?.chip ?? null, backend: machine?.backend ?? null, ram: machine?.ram_bytes ?? null }]
    : [{ host, chip: peer?.chip ?? null, backend: peer?.backend ?? null, ram: peer?.ram_bytes ?? null }]
  const installed = models?.filter((m) => m.installed && m.runs_here) ?? []
  const recommended = models?.find((m) => m.recommended) ?? null
  const last = settings?.model ? installed.find((m) => m.id === settings.model) ?? null : null
  const toStart = last ?? installed[0] ?? null
  const start = async (m: Model) => {
    setBusy(true)
    try {
      const fraction = m.verdict?.level === 'tight' ? m.verdict.budget_fraction : settings?.budget_fraction ?? null
      await api.start({ model: m.id, budget_fraction: fraction })
      await Promise.all([refreshEngine(), refreshSettings()])
      toast(`Starting ${m.name}`)
    } catch (e) { toast(String(e), 'bad') } finally { setBusy(false) }
  }

  // the state in one sentence, and the one button that makes sense now
  let line: ReactNode
  let action: ReactNode = null
  if (running) {
    const n = engine?.connections ?? 0
    line = <><b>{name}</b> is ready{n ? `, answering ${n} ${n === 1 ? 'conversation' : 'conversations'}` : ' and idle'}.</>
    action = <a className="btn" href="#models">Check other models</a>      // Chat has its tile below; Models has none
  } else if (st === 'starting') {
    line = <><b>{name}</b> is starting. It takes about fifteen seconds; the first start takes longer.</>
  } else if (st === 'unauthorized') {
    line = <>An AI answers at the address in Settings, but it refuses the key this TensorFold Web holds.</>
    action = <a className="btn" href="#settings">Open Settings</a>
  } else if (st === 'failed' || st === 'error') {
    line = <>{name === 'the AI' ? 'The AI' : name} stopped with a problem.</>
    action = <a className="btn" href="#monitor">See what happened</a>
  } else if (engine?.managed === false) {
    line = <>No AI answers at the address in Settings yet.</>
    action = <a className="btn" href="#settings">Open Settings</a>
  } else if (toStart) {
    line = <>Nothing is running. <b>{toStart.name}</b> is on this machine, ready to start.</>
    action = <button className="btn primary" aria-busy={busy} disabled={busy || stale} onClick={() => start(toStart)}>{busy ? 'Starting…' : `Start ${toStart.name}`}</button>
  } else if (recommended) {
    const size = (recommended.size_bytes ?? 0) + (recommended.drafter_bytes ?? 0)
    line = <>No AI is installed yet. <b>{recommended.name}</b> fits this machine{size ? `: ${gb(size)} to download` : ''}.</>
    action = <a className="btn primary" href={linkTo('models', recommended.id)}>Get {recommended.name}</a>
  } else if (models) {
    line = <>No AI is installed yet.</>
    action = <a className="btn primary" href="#models">Choose one</a>
  } else if (catalogErr) {
    line = <>The list of AIs could not be read ({catalogErr}).</>
    action = <button className="btn" onClick={() => api.catalog().then((c) => { setModels(c.models); setCatalogErr(null) }).catch((e) => setCatalogErr(String((e as Error).message ?? e)))}>Try again</button>
  } else {
    line = <>Reading this machine…</>
  }

  const copies = living?.copies ?? []
  const noTeach = whyNotTeach(engine, living)             // the tile stays, greyed, and says why
  // two counts, said apart: the AIs this machine's memory holds, and those its engine serves (on a Spark, TensorFold's
  // CUDA engine serves Nemotron only today, while four of the catalog fit its memory)
  const inMemory = models?.filter((m) => m.verdict && m.verdict.level !== 'no') ?? null
  const fits = inMemory?.length ?? null
  const runsHere = inMemory?.filter((m) => m.runs_here).length ?? null
  return (
    <section>
      <div className="head"><div><h1>Home</h1><p>{remote ? <>Your AI runs on {host ?? 'another machine'}; this TensorFold Web talks to it.</> : 'Your AI runs on this machine and keeps everything here.'}</p></div></div>
      <FirstRun settings={settings} saveSettings={saveSettings} toast={toast} />
      <div className="card home-state" role="status">
        <div><span className={`dot ${running ? '' : st === 'starting' ? 'warn' : st === 'failed' || st === 'error' || st === 'unauthorized' ? 'bad' : 'off'}`} /></div>
        <div className="home-lines">
          <div className="home-line">{(running || st === 'starting') && <ModelLogo names={[engine?.model_name, engine?.model]} size="lg" />}<span>{line}</span></div>
          {(running || st === 'starting' || st === 'failed' || st === 'error') && rigs.length > 0 && <RigLine rigs={rigs} />}
        </div>
        {action && <div className="actions">{action}</div>}
      </div>

      <div className="home-actions">
        <a className="tile" href="#chat">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d={ICON.chat} /></svg>
          <b>Chat</b><span>Ask it anything. Everything you write stays on this machine.</span>
        </a>
        {noTeach ? <div className="tile off" aria-disabled="true" title={noTeach}>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d={ICON.teach} /></svg>
          <b>Teach it something</b><span>{noTeach}</span>
        </div> : <a className="tile" href="#teach">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d={ICON.teach} /></svg>
          <b>Teach it something</b><span>Tell it a fact about you or your work; it remembers it from then on.</span>
        </a>}
        <a className="tile" href="#connect">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d={ICON.apps} /></svg>
          <b>Use it in an app</b><span>Claude Code, Open WebUI, LibreChat and others, with settings to copy.</span>
        </a>
        <a className="tile" href="#bench">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d={ICON.bench} /></svg>
          <b>Bench</b><span>Measure its speed, check the speed-up is exact, compare models, settings and machines.</span>
        </a>
        <a className="tile" href="#monitor">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d={ICON.monitor} /></svg>
          <b>Monitor</b><span>What the AI is doing now: memory, speed, conversations served, the engine's log.</span>
        </a>
        <a className="tile" href="#fleet">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d={ICON.fleet} /></svg>
          <b>Fleet</b><span>This machine and the others, each with its AI, watched and opened from here.</span>
        </a>
      </div>

      <div className="two">
        <div className="card">
          <div className="membar-head"><b>Last conversations</b><a className="small" href="#chat">All of them</a></div>
          {convs.length ? <ul className="home-list">{convs.map((c) => <li key={c.id}><a href={linkTo('chat', c.id)} title={c.title}>{c.title}</a><span className="small muted">{ago(c.updated)}</span></li>)}</ul>
            : <p className="small muted">None yet. {running ? <a href="#chat">Start one.</a> : 'Start the AI, then say hello.'}</p>}
        </div>
        <div className={`card${noTeach ? ' off' : ''}`}>
          <div className="membar-head"><b>What it has learned</b>{!noTeach && <a className="small" href="#teach">Teach</a>}</div>
          {copies.length > 0 && <ul className="home-list">{copies.map((c) => <li key={c.name}><a href="#teach">{c.name}</a><span className="small muted">{c.taught} {c.taught === 1 ? 'fact' : 'facts'} kept{living?.served === c.name ? ', running now' : ''}</span></li>)}</ul>}
          {noTeach ? <p className="small muted">{noTeach}</p>
            : !copies.length && <p className="small muted">Nothing yet. Teach keeps what you tell it in a copy of the AI, so the original stays as it was.</p>}
        </div>
      </div>

      {machine && <p className="small muted home-machine">{machine.chip}, {gb(machine.ram_bytes)} of memory{fits != null ? `: ${fits} of the ${models?.length ?? 0} AIs in the catalog fit its memory${runsHere != null && runsHere !== fits ? `, ${runsHere} of them ${runsHere === 1 ? 'runs' : 'run'} on the engine installed here` : ''}` : ''}. <a href="#models">See them</a>.</p>}
    </section>
  )
}

/** The first label of an engine address's host (spark for http://spark.example.ts.net:8080). */
function shortHost(url: string | undefined): string | null {
  try {
    const h = url ? new URL(url).hostname : ''
    return /^[\d.]+$|:/.test(h) ? h : h.split('.')[0] || null  // an IP address stays whole
  } catch { return null }
}

type Rig = { host: string; chip: string | null; backend: string | null; ram: number | null }

/** The hardware under the model, on its own line: each maker's logo once, then each machine with its chip and memory. */
function RigLine({ rigs }: { rigs: Rig[] }) {
  const makers = [...new Map(rigs.map((r) => [hardwareOf(r.chip, r.backend)?.id ?? r.host, r])).values()]
  return (
    <div className="home-rig">
      <span className="home-rig-logos">{makers.map((r) => <HardwareLogo key={r.host} names={[r.chip, r.backend]} />)}</span>
      <span>On {rigs.map((r, i) => <span key={r.host}>{i ? (i === rigs.length - 1 ? ' and ' : ', ') : ''}<b>{r.host}</b>
        {r.chip || r.ram ? ` (${[chipName(r.chip), r.ram ? `${gb(r.ram)} of memory` : null].filter(Boolean).join(', ')})` : ''}</span>)}</span>
    </div>
  )
}

/** A chip as a person names it: the board in brackets left out (NVIDIA GB10 for "NVIDIA GB10 (GX10)"). */
function chipName(chip: string | null): string | null {
  return chip ? chip.replace(/\s*\([^)]*\)\s*$/, '') : null
}
