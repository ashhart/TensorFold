import { shortPath } from '../../../lib/format'
import { BrandIcon } from '../../../lib/brand'
import { Icon } from '../../../lib/icons'
import { Button, Input, Toggle } from '../../ui'
import { Section, type SettingsSectionContext } from '../shared'
import { ConnectorStatusChip } from './ConnectorStatusChip'

export function BrowserConnector({ ctx }: { ctx: SettingsSectionContext }) {
  const {
    draft,
    setDraft,
    browserBusy,
    browserTabUrl,
    setBrowserTabUrl,
    browserTabTitle,
    setBrowserTabTitle,
    browserTabExcerpt,
    setBrowserTabExcerpt,
    browserHistoryPath,
    setBrowserHistoryPath,
    connectorStatus,
    addBrowserCurrentTabSource,
    addBrowserHistorySource,
    updateBrowserSource,
    removeBrowserSource,
    syncBrowser,
  } = ctx
  const connectorStatusMap = connectorStatus ?? {}

  return (
    <Section title="Browser connector" desc="Save explicit current-tab snapshots or a user-provided JSON history export for source-backed browser reads and summaries.">
      <div className="space-y-3">
        <div className="rounded-xl border border-line bg-card px-3.5 py-3 shadow-card">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <div className="text-[13px] font-medium text-ink">{draft.browser.sources.length} source{draft.browser.sources.length === 1 ? '' : 's'} configured</div>
              <div className="mt-0.5 text-[13px] text-ink-3">{draft.browser.lastSyncAt ? `Last sync ${new Date(draft.browser.lastSyncAt).toLocaleString()}` : 'Not synced yet'}</div>
              <ConnectorStatusChip status={connectorStatusMap.browser} />
            </div>
            <Button variant="primary" size="sm" icon="browser" onClick={syncBrowser} disabled={!!browserBusy || draft.browser.sources.length === 0}>{browserBusy === 'sync' ? 'Syncing' : 'Sync browser'}</Button>
          </div>

          <div className="mt-3 grid gap-2 lg:grid-cols-[1.3fr_.8fr]">
            <div className="space-y-2 rounded-lg border border-line bg-card-2 p-2.5">
              <Input value={browserTabUrl} onChange={(e) => setBrowserTabUrl(e.target.value)} placeholder="https://example.com/current-page" className="h-8" />
              <div className="grid gap-2 sm:grid-cols-2">
                <Input value={browserTabTitle} onChange={(e) => setBrowserTabTitle(e.target.value)} placeholder="Page title" className="h-8" />
                <Input value={browserTabExcerpt} onChange={(e) => setBrowserTabExcerpt(e.target.value)} placeholder="Short excerpt or note" className="h-8" />
              </div>
              <div className="flex justify-end">
                <Button variant="default" size="sm" icon="plus" onClick={addBrowserCurrentTabSource} disabled={!!browserBusy || !browserTabUrl.trim()}>{browserBusy === 'add-tab' ? 'Adding' : 'Add current tab'}</Button>
              </div>
            </div>
            <div className="space-y-2 rounded-lg border border-line bg-card-2 p-2.5">
              <Input value={browserHistoryPath} onChange={(e) => setBrowserHistoryPath(e.target.value)} placeholder="/path/to/history-export.json" className="h-8" />
              <div className="flex justify-end">
                <Button variant="default" size="sm" icon="fileText" onClick={addBrowserHistorySource} disabled={!!browserBusy || !browserHistoryPath.trim()}>{browserBusy === 'add-history' ? 'Adding' : 'Add history export'}</Button>
              </div>
            </div>
          </div>
        </div>

        {draft.browser.sources.length === 0 ? (
          <div className="rounded-xl border border-dashed border-line px-3.5 py-5 text-center text-[13px] text-ink-3">
            Browser sources stay empty until you explicitly save a current tab or provide a local history export file.
          </div>
        ) : draft.browser.sources.map((source) => {
          const target = source.kind === 'history_export' ? (source.path ?? '') : (source.url ?? '')
          return (
            <div key={source.id} className="rounded-xl border border-line bg-card p-3.5 shadow-card">
              <div className="flex items-start gap-3">
                <BrandIcon brand="browser" size={32} />
                <div className="min-w-0 flex-1">
                  <Input
                    value={source.label}
                    onChange={(e) => setDraft((d) => ({ ...d, browser: { ...d.browser, sources: d.browser.sources.map((s) => s.id === source.id ? { ...s, label: e.target.value } : s) } }))}
                    onBlur={(e) => updateBrowserSource(source.id, { label: e.target.value })}
                    className="h-8"
                  />
                  <div className="mt-1 truncate text-[13px] text-ink-3" title={target}>{source.kind === 'history_export' ? shortPath(target, 5) : target}</div>
                  {source.lastError && <div className="mt-1 text-[13px] text-danger">{source.lastError}</div>}
                </div>
                <Toggle label={`Enable ${source.label || "browser source"}`} checked={source.enabled} onChange={(enabled) => updateBrowserSource(source.id, { enabled })} />
                <button onClick={() => removeBrowserSource(source.id)} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-danger/10 hover:text-danger" aria-label="Remove browser source">
                  <Icon name="trash" size={15} />
                </button>
              </div>
              <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
                <span className="rounded-lg border border-line bg-card-2 px-2.5 py-1.5 text-[13px] text-ink-3">{source.kind === 'history_export' ? 'History export JSON' : 'Current tab snapshot'}</span>
                <span className="text-[12px] text-ink-3">{source.lastSyncAt ? `Synced ${new Date(source.lastSyncAt).toLocaleString()}` : 'Not scanned'}</span>
              </div>
            </div>
          )
        })}
      </div>
    </Section>
  )
}
