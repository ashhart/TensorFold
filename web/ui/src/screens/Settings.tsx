import { useEffect, useState } from 'react'
import { api, gb, num, type Verdict } from '../api'
import type { Ctx } from '../ctx'
import SettingsAccess from './SettingsAccess'
import SettingsBrands from './SettingsBrands'
import SettingsStorage from './SettingsStorage'
import { Diagnostic } from './SetupChecks'
import { before } from '../version'

const CONTEXTS = [null, 8192, 16384, 32768, 65536, 131072]
const INSTALL = 'curl -fsSL https://tensorfold.dev/install.sh | sh'

export default function SettingsScreen({ machine, engine, settings, saveSettings, toast, refreshBrands }: Ctx) {
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
  const cap = machine.budget.capacity_bytes ?? machine.ram_bytes          // what the engine's backend can draw on
  const memoryName = machine.backend === 'cuda' && !machine.unified_memory ? "the card's memory" : machine.unified_memory ? 'unified memory' : 'RAM'
  const limitVar = machine.backend === 'cuda' ? 'TENSORFOLD_CUDA_MEMORY_LIMIT_GB' : 'TENSORFOLD_MEMORY_LIMIT_GB'
  const fraction = settings.budget_fraction ?? machine.budget.default_fraction
  // a slider only where the budget has a range: the Python line on a Mac (70 to 85%); the native engine and a GB10
  // size their memory themselves, so there it is a figure to read, not a setting
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
    <section>
      <div className="head"><div><h1>Settings</h1><p>The defaults are safe. The one worth knowing about on a 32 GB machine is how much memory the AI may use.</p></div></div>
      {err && <div className="notice bad">{err}</div>}
      <div className="settings">
        <div className="section">
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
              ? <p className="hint">The native engine installed here sizes its memory itself and does not take this budget; it applies to the frozen Python line only. What the native engine admits at startup is in Monitor.</p>
              : <p className="hint">Shares of {memoryName} ({gb(cap)}). Above the default, TensorFold Web passes <code>{limitVar}</code> to the engine; the engine's own admission at startup is what counts, and Monitor shows it. Changes apply at the next start.</p>}
            </> : <>
            <p className="budget-fixed"><b>{gb(cap * machine.budget.default_fraction)}</b> of the {gb(cap)} of {memoryName}</p>
            <p className="hint">TensorFold sizes its memory itself on this machine and keeps the rest for the system and your apps: there is nothing to set here. What the engine admitted at its last start is in Monitor.</p>
            {settings.model && <p className="hint">{hint}</p>}
            </>}
            {machine.budget.ceiling_bytes && machine.budget.ceiling_bytes < cap * machine.budget.raised_fraction && (
              <p className="hint">macOS lets the GPU use at most <b>{gb(machine.budget.ceiling_bytes)}</b> on this Mac, whatever the budget asks. On a Mac used mostly for TensorFold, <code>sudo sysctl iogpu.wired_limit_mb={Math.round((cap * machine.budget.raised_fraction) / 1024 ** 2)}</code> raises it to {gb(cap * machine.budget.raised_fraction)} until the next restart; large models then get a usable context.</p>
            )}
          </div>
        </div>

        <div className="section">
          <h2>Conversations</h2>
          <div className="srow"><div className="l"><b>Thinking</b><span>Default for new chats. Apps can override it.</span></div>
            <div className="r"><select className="sel" aria-label="Thinking" value={settings.thinking} onChange={(e) => save({ thinking: e.target.value as typeof settings.thinking })}><option value="default">Model default</option><option value="on">On</option><option value="off">Off</option></select></div></div>
          <div className="srow"><div className="l"><b>Longest conversation</b><span>Auto lets TensorFold size it from memory.</span></div>
            <div className="r"><select className="sel" aria-label="Longest conversation" value={settings.context ?? ''} onChange={(e) => save({ context: e.target.value ? Number(e.target.value) : null })}>{CONTEXTS.map((c) => <option key={String(c)} value={c ?? ''}>{c ? `${num(c)} tokens` : 'Auto'}</option>)}</select></div></div>
          {flags?.flags['--cortheon'] && <div className="srow"><div className="l"><b>Web research (Cortheon)</b><span>TensorFold works through Cortheon's reasoning stages and searches the web before it answers, for every chat request any app sends; a request can still turn it off. Applies at the next start.
            {' '}<b>What leaves this machine:</b> the search queries the model writes from the conversation (to DuckDuckGo), and each page it reads, fetched from this machine's address. A web page can steer the model: use it where that risk is acceptable.
            {settings.cortheon && settings.share && <span className="bad"> The engine is shared on the network: every app that reaches it gets web research.</span>}
            {before(flags.version, [1, 0, 11]) && <span className="bad"> TensorFold {flags.version} predates 1.0.11, Cortheon's security release: a web page can hide orders from its instruction filter, hold a request for minutes or stop the engine. Update with <code>{INSTALL}</code> first.</span>}</span></div>
            <div className="r"><label className="toggle"><input type="checkbox" aria-label="Web research (Cortheon)" checked={!!settings.cortheon} onChange={(e) => save({ cortheon: e.target.checked }, e.target.checked ? 'Web research on at the next start' : 'Web research off at the next start')} /></label></div></div>}
          <div className="srow"><div className="l"><b>Appearance</b><span>Follows the system by default.</span></div>
            <div className="r"><select className="sel" aria-label="Appearance" value={settings.theme} onChange={(e) => save({ theme: e.target.value as typeof settings.theme })}><option value="system">System</option><option value="light">Light</option><option value="dark">Dark</option></select></div></div>
        </div>

        <div style={{ marginTop: 14 }}><SettingsAccess toast={toast} /></div>
        <div style={{ marginTop: 14 }}><SettingsBrands toast={toast} refreshBrands={refreshBrands} /></div>

        <details className="more" style={{ marginTop: 18 }}>
        <summary>Advanced: the engine, its checks, the Bench's price of electricity, storage</summary>
        <div className="section">
          <h2>Bench</h2>
          <div className="srow"><div className="l"><b>Electricity price</b><span>Per kWh. Bench results and reports then show what a run cost in electricity, and per million tokens: a figure to set beside a paid API's price. Measured for the whole machine on a Mac, the GPU only on NVIDIA and AMD (an estimate). Empty: no cost shown.</span></div>
            <div className="r"><div className="field"><input type="number" min={0} max={10} step={0.01} defaultValue={settings.energy_price ?? ''} key={`p${settings.energy_price}`} style={{ width: 90 }}
              onBlur={(e) => { const v = e.target.value.trim(); const n = v ? Number(v) : null; if (n !== settings.energy_price) save({ energy_price: n }, 'Price saved') }} aria-label="Electricity price per kWh" placeholder="0.25" />
              <input type="text" defaultValue={settings.currency} key={`c${settings.currency}`} maxLength={3} style={{ width: 60 }}
              onBlur={(e) => { const v = e.target.value.trim().toUpperCase(); if (v && v !== settings.currency) save({ currency: v }, 'Currency saved') }} aria-label="Currency" /></div></div></div>
        </div>

        <div className="section">
          <h2>Engine</h2>
          <div className="srow"><div className="l"><b>TensorFold {machine.tensorfold_version ?? 'not installed'}</b><span>{machine.tensorfold_version ? (machine.tensorfold_version.includes('Python') ? `The frozen Python line, for the models not yet on the native engine. ${machine.backend_note}` : `The native engine. ${machine.backend_note}`) : <>Install it with <code>{INSTALL}</code> (Apple Silicon Mac or NVIDIA Linux), or connect a remote engine below.</>}</span></div></div>
          <div className="srow"><div className="l"><b>Remote engine</b><span>Drive a TensorFold on another machine, such as a DGX Spark. TensorFold Web then only watches it.</span></div>
            <div className="r"><label className="toggle"><input type="checkbox" aria-label="Remote engine" checked={settings.remote} onChange={(e) => save({ remote: e.target.checked })} /></label></div></div>
          <div className="srow"><div className="l" style={{ flex: 1 }}><b>Address</b><span>Where TensorFold Web reaches the engine.</span>
            <div className="field"><input type="text" value={url} onChange={(e) => setUrl(e.target.value)} onBlur={() => url !== settings.engine_url && save({ engine_url: url.replace(/\/$/, '') }, 'Address saved')} placeholder="http://spark.local:8080" aria-label="Engine address" /></div></div></div>
          {flags?.binary && <div className="srow"><div className="l" style={{ flex: 1 }}><b>Extra engine flags</b><span>Passed to <code>tensorfold serve</code> at the next start, checked against the flags TensorFold {flags.version} offers. TensorFold Web sets the address, context, thinking and key itself.</span>
            <div className="field"><input type="text" list="serve-flags" value={extra} onChange={(e) => setExtra(e.target.value)} placeholder="--keep-warm --parallel 4" aria-label="Extra engine flags" autoComplete="off" spellCheck={false}
              onBlur={() => { const args = extra.trim() ? extra.trim().split(/\s+/) : []; if (args.join(' ') !== (settings.extra_args ?? []).join(' ')) save({ extra_args: args }, 'Flags saved for the next start') }} />
              <datalist id="serve-flags">{Object.entries(flags.flags).filter(([, f]) => !f.managed).map(([n, f]) => <option key={n} value={n}>{f.choices ? f.choices.join(' | ') : f.value ? 'takes a value' : ''}</option>)}</datalist></div></div></div>}
          <div className="srow"><div className="l" style={{ flex: 1 }}><b>Engine API key</b><span>{settings.api_key_set ? 'Set. TensorFold Web sends it on every call and starts its own engine with it; Connect shows it.' : 'None. An engine this TensorFold Web starts needs no key unless you generate one in Connect; a remote engine started with one needs it entered here.'}{engine?.key_pending_restart ? ' The running engine still uses its previous key until it restarts.' : ''}</span>
            <div className="field"><input type="password" value={key} onChange={(e) => setKey(e.target.value)} onBlur={saveKey} onKeyDown={(e) => { if (e.key === 'Enter') saveKey() }} placeholder={settings.api_key_set ? 'type a new one to replace it' : 'paste the key a remote engine expects'} autoComplete="off" aria-label="Engine API key" />{settings.api_key_set && <button className="btn" onClick={() => save({ api_key: null }, 'Engine key forgotten')}>Forget</button>}</div></div></div>
        </div>

        <div style={{ marginTop: 14 }}><Diagnostic settings={settings} saveSettings={saveSettings} toast={toast} /></div>
        <div style={{ marginTop: 14 }}><SettingsStorage toast={toast} /></div>
        </details>

        <p className="about">TensorFold Web {machine.studio_version} on {machine.os} {machine.os_version}, {machine.chip}, {gb(machine.ram_bytes)} {machine.unified_memory ? 'unified memory' : 'RAM'}. Settings live in TensorFold Web's configuration folder on this machine. TensorFold Web itself only reaches the engine and, to read model sizes and download models, Hugging Face.</p>
      </div>
    </section>
  )
}
