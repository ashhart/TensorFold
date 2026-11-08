import { useEffect, useState } from 'react'
import { api, gb, num, type Verdict } from '../api'
import type { Ctx } from '../ctx'
import SettingsAccess from './SettingsAccess'
import SettingsBrands from './SettingsBrands'
import SettingsStorage from './SettingsStorage'
import { Diagnostic, FirstRun } from './SetupChecks'
import { before, displayEngineVersion } from '../version'
import { subOf } from '../route'

const CONTEXTS = [null, 8192, 16384, 32768, 65536, 131072]
const SECTIONS = [['engine', 'Models & engine'], ['memory', 'Memory'], ['conversations', 'Conversations'], ['access', 'Access'], ['brands', 'Brands'], ['checks', 'Setup checks'], ['storage', 'Storage'], ['bench', 'Benchmark costs'], ['about', 'About']] as const
const settingsPage = () => { const s = subOf('settings'); return SECTIONS.some(([id]) => id === s) ? s! : 'engine' }
const INSTALL = 'curl -fsSL https://tensorfold.dev/install.sh | sh'

export default function SettingsScreen({ machine, engine, settings, saveSettings, toast, refreshBrands }: Ctx) {
  const [page, setPage] = useState(settingsPage)
  useEffect(() => { const on = () => setPage(settingsPage()); addEventListener('hashchange', on); return () => removeEventListener('hashchange', on) }, [])
  const [verdicts, setVerdicts] = useState<(Verdict & { fraction: number })[]>([])
  const [url, setUrl] = useState(settings?.engine_url ?? '')
  const [key, setKey] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const [flags, setFlags] = useState<Awaited<ReturnType<typeof api.engineFlags>> | null>(null)
  const [extra, setExtra] = useState((settings?.extra_args ?? []).join(' '))
  useEffect(() => { api.engineFlags().then(setFlags).catch(() => setFlags(null)) }, [])
  const [dragging, setDragging] = useState<number | null>(null)          // the slider while the hand is on it
  useEffect(() => { setExtra((settings?.extra_args ?? []).join(' ')) }, [settings?.extra_args])
  useEffect(() => { setUrl(settings?.engine_url ?? '') }, [settings?.engine_url])
  useEffect(() => { if (settings?.model) api.verdicts(settings.model).then((r) => setVerdicts(r.verdicts)).catch(() => setVerdicts([])) }, [settings?.model])
  if (!settings || !machine) return <section><div className="head"><div><h1>Settings</h1></div></div></section>
  const engineVersion = displayEngineVersion(machine.tensorfold_version, flags?.version)
  const cap = machine.budget.capacity_bytes ?? machine.ram_bytes          // what the engine's backend can draw on
  const memoryName = machine.backend === 'cuda' && !machine.unified_memory ? "the card's memory" : machine.unified_memory ? 'unified memory' : 'RAM'
  const limitVar = machine.backend === 'cuda' ? 'TENSORFOLD_CUDA_MEMORY_LIMIT_GB' : 'TENSORFOLD_MEMORY_LIMIT_GB'
  const fraction = settings.budget_fraction ?? machine.budget.default_fraction
  // Offer a budget slider only when the backend admits an adjustable range.
  const adjustable = machine.budget.raised_fraction - machine.budget.default_fraction > 0.001
  const current = verdicts.find((v) => Math.abs(v.fraction - (dragging ?? fraction)) < 0.001)
  const modelName = settings.model?.split('/').pop()
  const hint = !settings.model ? 'Pick a model in Models to see what each budget allows it.'
    : current ? (current.level === 'no' ? `At ${gb(current.budget_bytes)}, ${modelName} does not run: its weights and one prompt chunk leave no usable context.`
      : `At ${gb(current.budget_bytes)}, ${modelName} ${current.level === 'tight' ? 'runs with' : 'runs comfortably with'} about ${num(current.context_tokens)} tokens of context, and ${gb(current.left_for_apps_bytes)} stay free for your apps.`)
    : 'The size of this model is not known yet.'
  const save = async (patch: Parameters<typeof saveSettings>[0], done = 'Saved') => { try { await saveSettings(patch); setErr(null); toast(done) } catch (e) { setErr(String(e)) } }
  const saveKey = async () => { if (!key.trim()) return; await save({ api_key: key.trim() }, 'Engine key saved'); setKey('') }
  return (
    <section className="settings-shell">
      <nav className="settings-nav" aria-label="Settings sections"><a className="back-link" href="#chat">← Back to app</a><h2>Settings</h2>{SECTIONS.map(([id, label]) => <a key={id} href={`#settings/${id}`} aria-current={page === id ? 'page' : undefined}>{label}</a>)}</nav>
      <div className="settings-content"><div className="head"><div><h1>{SECTIONS.find(([id]) => id === page)?.[1]}</h1><p>Manage TensorFold Web on this machine.</p></div></div>
      {err && <div className="notice bad">{err}</div>}
      <div className="settings">
        <div className="section" hidden={page !== 'memory'}>
          <h2>Memory the AI may use</h2>
          <div style={{ padding: '4px 0 14px' }}>
            {adjustable ? <>
            <div className="budget"><span>{gb(cap * machine.budget.default_fraction)}<br /><span className="small">default, {Math.round(machine.budget.default_fraction * 100)}%</span></span><b>{gb(cap * fraction)}</b><span style={{ textAlign: 'right' }}>{gb(cap * machine.budget.raised_fraction)}<br /><span className="small">most</span></span></div>
            <input type="range" min={Math.round(machine.budget.default_fraction * 100)} max={Math.round(machine.budget.raised_fraction * 100)} step={5} value={Math.round((dragging ?? fraction) * 100)}
                   onChange={(e) => setDragging(Number(e.target.value) / 100)}
                   onPointerUp={() => { if (dragging != null) { save({ budget_fraction: dragging }); setDragging(null) } }}
                   onKeyUp={() => { if (dragging != null) { save({ budget_fraction: dragging }); setDragging(null) } }}
                   onBlur={() => { if (dragging != null) { save({ budget_fraction: dragging }); setDragging(null) } }}
                   aria-label="Memory budget" />
            <p className="hint">{hint}</p>
            {machine.tensorfold_version && !machine.tensorfold_version.includes('Python')
              ? <p className="hint">This budget applies to the frozen Python line; native memory admission is in <a href="#monitor">Details</a>.</p>
              : <p className="hint">Next start uses <code>{limitVar}</code> for {memoryName} ({gb(cap)}), subject to engine admission.</p>}
            </> : <>
            <p className="budget-fixed"><b>{gb(cap * machine.budget.default_fraction)}</b> of the {gb(cap)} of {memoryName}</p>
            <p className="hint">TensorFold sizes its memory at startup; <a href="#monitor">Details</a> shows the admitted budget.</p>
            {settings.model && <p className="hint">{hint}</p>}
            </>}
            {machine.budget.ceiling_bytes && machine.budget.ceiling_bytes < cap * machine.budget.raised_fraction && (
              <p className="hint">GPU ceiling: <b>{gb(machine.budget.ceiling_bytes)}</b>; <code>sudo sysctl iogpu.wired_limit_mb={Math.round((cap * machine.budget.raised_fraction) / 1024 ** 2)}</code> raises it until restart.</p>
            )}
          </div>
        </div>

        <div className="section" hidden={page !== 'conversations'}>
          <h2>Conversations</h2>
          <div className="srow"><div className="l"><b>Thinking</b><span>Default for new chats; apps can override it.</span></div>
            <div className="r"><select className="sel" aria-label="Thinking" value={settings.thinking} onChange={(e) => save({ thinking: e.target.value as typeof settings.thinking })}><option value="default">Model default</option><option value="on">On</option><option value="off">Off</option></select></div></div>
          <div className="srow"><div className="l"><b>Longest conversation</b><span>Auto lets TensorFold size it from memory.</span></div>
            <div className="r"><select className="sel" aria-label="Longest conversation" value={settings.context ?? ''} onChange={(e) => save({ context: e.target.value ? Number(e.target.value) : null })}>{CONTEXTS.map((c) => <option key={String(c)} value={c ?? ''}>{c ? `${num(c)} tokens` : 'Auto'}</option>)}</select></div></div>
          {flags?.flags['--cortheon'] && <div className="srow"><div className="l"><b>Web research (Cortheon)</b><span>Run web research on each chat at the next start, unless its request opts out.
            </span><span>Search queries go to DuckDuckGo; visited sites see this machine's address.</span>
            <span>Fetched pages can steer the model.</span>
            {settings.cortheon && settings.share && <span className="bad"> The engine is shared on the network: every app that reaches it gets web research.</span>}
            {before(engineVersion, [1, 0, 11]) && <span className="bad"> Update with <code>{INSTALL}</code> before using Cortheon in {engineVersion}, which lacks its 1.0.11 security fixes.</span>}</div>
            <div className="r"><label className="toggle"><input type="checkbox" aria-label="Web research (Cortheon)" checked={!!settings.cortheon} onChange={(e) => save({ cortheon: e.target.checked }, e.target.checked ? 'Web research on at the next start' : 'Web research off at the next start')} /></label></div></div>}
          <div className="srow"><div className="l"><b>Appearance</b><span>Follows the system by default.</span></div>
            <div className="r"><select className="sel" aria-label="Appearance" value={settings.theme} onChange={(e) => save({ theme: e.target.value as typeof settings.theme })}><option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option></select></div></div>
        </div>

        <div hidden={page !== 'access'}><SettingsAccess toast={toast} /></div>
        <div hidden={page !== 'brands'}><SettingsBrands toast={toast} refreshBrands={refreshBrands} /></div>

        <div className="section" hidden={page !== 'bench'}>
          <h2>Bench</h2>
          <div className="srow"><div className="l"><b>Electricity price</b><span>Price per kWh for benchmark cost estimates; leave empty to hide costs.</span></div>
            <div className="r"><div className="field"><input type="number" min={0} max={10} step={0.01} defaultValue={settings.energy_price ?? ''} key={`p${settings.energy_price}`} style={{ width: 90 }}
              onBlur={(e) => { const v = e.target.value.trim(); const n = v ? Number(v) : null; if (n !== settings.energy_price) save({ energy_price: n }, 'Price saved') }} aria-label="Electricity price per kWh" placeholder="0.25" />
              <input type="text" defaultValue={settings.currency} key={`c${settings.currency}`} maxLength={3} style={{ width: 60 }}
              onBlur={(e) => { const v = e.target.value.trim().toUpperCase(); if (v && v !== settings.currency) save({ currency: v }, 'Currency saved') }} aria-label="Currency" /></div></div></div>
        </div>

        <div className="section" hidden={page !== 'engine'}>
          <h2>Engine</h2>
          <div className="srow"><div className="l"><b>TensorFold {engineVersion}</b><span>{machine.tensorfold_version ? (machine.tensorfold_version.includes('Python') ? `Frozen Python engine · ${machine.backend_note}` : `Native engine · ${machine.backend_note}`) : <>Install it with <code>{INSTALL}</code> (Apple Silicon Mac or NVIDIA Linux), or connect a remote engine below.</>}</span></div></div>
          <div className="srow"><div className="l"><b>External engine</b><span>Connect to an engine Web did not start, on this or another machine.</span></div>
            <div className="r"><label className="toggle"><input type="checkbox" aria-label="External engine" checked={settings.remote} onChange={(e) => save({ remote: e.target.checked })} /></label></div></div>
          <div className="srow"><div className="l" style={{ flex: 1 }}><b>Address</b><span>Where TensorFold Web reaches the engine.</span>
            <div className="field"><input type="text" value={url} onChange={(e) => setUrl(e.target.value)} onBlur={() => url !== settings.engine_url && save({ engine_url: url.replace(/\/$/, '') }, 'Address saved')} placeholder="http://engine.local:8080" aria-label="Engine address" /></div></div></div>
          {flags?.binary && <div className="srow"><div className="l" style={{ flex: 1 }}><b>Extra engine flags</b><span>Use TensorFold {engineVersion}'s supported flags at the next start.</span>
            <div className="field"><input type="text" list="serve-flags" value={extra} onChange={(e) => setExtra(e.target.value)} placeholder="--keep-warm --parallel 4" aria-label="Extra engine flags" autoComplete="off" spellCheck={false}
              onBlur={() => { const args = extra.trim() ? extra.trim().split(/\s+/) : []; if (args.join(' ') !== (settings.extra_args ?? []).join(' ')) save({ extra_args: args }, 'Flags saved for the next start') }} />
              <datalist id="serve-flags">{Object.entries(flags.flags).filter(([, f]) => !f.managed).map(([n, f]) => <option key={n} value={n}>{f.choices ? f.choices.join(' | ') : f.value ? 'takes a value' : ''}</option>)}</datalist></div></div></div>}
          <div className="srow"><div className="l" style={{ flex: 1 }}><b>Engine API key</b><span>{settings.api_key_set ? 'Web sends this key on every call and uses it at engine start; API shows it.' : 'Enter the remote engine’s key here, or generate one in API for a Web-managed engine.'}</span>{engine?.key_pending_restart && <span>The running engine keeps its previous key until restart.</span>}
            <div className="field"><input type="password" value={key} onChange={(e) => setKey(e.target.value)} onBlur={saveKey} onKeyDown={(e) => { if (e.key === 'Enter') saveKey() }} placeholder={settings.api_key_set ? 'type a new one to replace it' : 'paste the key a remote engine expects'} autoComplete="off" aria-label="Engine API key" />{settings.api_key_set && <button className="btn" onClick={() => save({ api_key: null }, 'Engine key forgotten')}>Forget</button>}</div></div></div>
        </div>

        <div hidden={page !== 'checks'}><FirstRun settings={settings} saveSettings={saveSettings} toast={toast} /><Diagnostic settings={settings} saveSettings={saveSettings} toast={toast} /></div>
        <div hidden={page !== 'storage'}><SettingsStorage toast={toast} /></div>

        <div hidden={page !== 'about'} className="card"><h2>TensorFold Web {machine.studio_version}</h2><p>Native engine {engineVersion}.</p><p className="small muted">{machine.os} {machine.os_version} · {machine.chip} · {gb(machine.ram_bytes)} {machine.unified_memory ? 'unified memory' : 'RAM'}</p><p className="small muted">Model discovery and downloads contact Hugging Face only when requested.</p></div>
      </div></div>
    </section>
  )
}
