import { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import '../../src/styles/index.css'
import { createRecoveryFixture, SCENARIOS, type Scenario } from './fixture-api'

// Install the synthetic preload before importing any production IPC consumer.
const fixture = createRecoveryFixture()
;(window as any).tensorfoldStudio = fixture.api
const nativeFetch = window.fetch.bind(window)
window.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
  const url = new URL(typeof input === 'string' ? input : input instanceof URL ? input.href : input.url, location.href)
  if (url.origin !== location.origin) return Promise.reject(new Error('Synthetic fixture blocks external network requests.'))
  return nativeFetch(input, init)
}) as typeof fetch
window.open = () => null
// Keep writing-assistant extensions out of this isolated test's editors.
const optOutEditors = () => document.querySelectorAll('input, textarea, [contenteditable]').forEach(element => {
  element.setAttribute('data-gramm', 'false')
  element.setAttribute('data-gramm_editor', 'false')
  element.setAttribute('data-enable-grammarly', 'false')
})
new MutationObserver(optOutEditors).observe(document.getElementById('root')!, { childList: true, subtree: true })
const { GoogleWorkspaceConnector } = await import('../../src/components/settings/connectors/GoogleWorkspaceConnector')

const requested = new URLSearchParams(location.search).get('scenario')
const initial: Scenario = SCENARIOS.find(item => item.value === requested)?.value ?? 'multiple'
fixture.reset(initial)
function FixturePage() {
  const [scenario, setScenario] = useState<Scenario>(initial)
  const [generation, setGeneration] = useState(0)
  const [summary, setSummary] = useState(fixture.summary)
  useEffect(() => fixture.subscribe(() => setSummary(fixture.summary())), [])
  const reset = (next: Scenario) => {
    fixture.reset(next); setScenario(next); setGeneration(value => value + 1)
    const url = new URL(location.href); url.searchParams.set('scenario', next); history.replaceState(null, '', url)
  }
  const control = 'rounded-lg border border-line-strong bg-card px-3 py-2 text-[13px] font-medium text-ink hover:bg-card-2 disabled:opacity-50'
  return <main className="mx-auto h-full max-w-5xl space-y-5 overflow-y-auto p-4 text-ink sm:p-7">
    <section aria-label="Synthetic fixture controls" className="space-y-3 rounded-xl border border-info/30 bg-card p-4 shadow-sm">
      <div className="flex flex-wrap items-center gap-2"><h1 className="text-lg font-semibold">Google account recovery</h1><span className="rounded-full bg-accent-soft px-2.5 py-1 text-xs font-semibold text-accent">SYNTHETIC FIXTURE · NO REAL ACCOUNTS</span></div>
      <p className="text-[13px] leading-relaxed text-ink-2">The connector, hooks, controls and styles below are the real app; this page supplies an in-memory preload with no Google network access or credentials.</p>
      <div className="flex flex-wrap items-end gap-2">
        <label className="text-[13px] text-ink-2">Scenario<select aria-label="Fixture scenario" value={scenario} onChange={event => reset(event.target.value as Scenario)} className="mt-1 block rounded-lg border border-line-strong bg-card-2 px-3 py-2 text-ink">{SCENARIOS.map(item => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
        <button className={control} onClick={() => reset(scenario)}>Reset scenario</button>
        <button className={control} onClick={fixture.armExpiry}>Expire next Gmail read</button>
        <button className={control} onClick={fixture.finishSignIn}>Finish fake sign-in</button>
        <button className={control} onClick={() => document.documentElement.classList.toggle('dark')}>Toggle theme</button>
      </div>
      <p role="status" className="text-[13px] text-ink-2">{summary.notice}</p>
      <p className="break-words font-mono text-xs text-ink-3">{summary.accounts} · saved drafts: {summary.drafts} · next-read expiry: {summary.expireNextRead ? 'armed' : 'off'} · fake sign-in: {summary.waiting ? 'waiting' : 'idle'}</p>
      <details className="text-[13px] text-ink-2"><summary className="cursor-pointer font-medium">Draft preservation check</summary><p className="mt-2 leading-relaxed">Choose Expired on next Gmail read, enter a recipient, subject and message in Write an email, then press Search; confirm the unsaved fields remain with Reconnect needed and Save local draft disabled, reconnect the same account, use Finish fake sign-in above, wait for polling and save the retained draft.</p></details>
    </section>
    <GoogleWorkspaceConnector key={generation} />
  </main>
}
createRoot(document.getElementById('root')!).render(<FixturePage />)
