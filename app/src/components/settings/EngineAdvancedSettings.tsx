import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useState } from 'react'
import { api } from '../../lib/ipc'
import { Icon } from '../../lib/icons'
import { formatBytes, useEngine } from '../../lib/engine'
import { Button, Input } from '../ui'
import { Section, SettingsPane } from './shared'
import { ENGINE_ENV_PRESETS, isDeniedEngineEnvKey, normalizeEngineEnvOverrides } from '../../../shared/engine-advanced-env'
import type { EngineEnvOverrides } from '../../lib/types'

// Advanced engine environment: the variables a power user passes through to every
// tensorfold-native process (load-limit raises, offline mode, engine diagnostics).
// The main process re-validates and audits every save; the renderer only edits the map.

const NAME_PATTERN = /^[A-Z][A-Z0-9_]{0,63}$/

type Row = { name: string; value: string }

const toRows = (overrides: EngineEnvOverrides): Row[] => Object.entries(overrides).map(([name, value]) => ({ name, value }))

export function EngineAdvancedSettings() {
  const tx = useAuthoredText()
  const { snapshot: engine } = useEngine()
  const [saved, setSaved] = useState<EngineEnvOverrides>({})
  const [rows, setRows] = useState<Row[]>([])
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => {
    let mounted = true
    api.engineEnvOverrides().then((next) => {
      if (!mounted) return
      setSaved(next); setRows(toRows(next))
    }).catch((e) => { if (mounted) setError(String((e as Error)?.message ?? e)) })
    return () => { mounted = false }
  }, [])

  const changed = JSON.stringify(rows) !== JSON.stringify(toRows(saved))
  const invalidName = rows.some((row) => !NAME_PATTERN.test(row.name))
  const incomplete = rows.some((row) => !row.value.length)
  const duplicate = new Set(rows.map((row) => row.name)).size !== rows.length
  const added = new Set(rows.map((row) => row.name))

  const patchRow = (index: number, patch: Partial<Row>) => setRows((prev) => prev.map((row, i) => (i === index ? { ...row, ...patch } : row)))
  const addRow = (name = '') => setRows((prev) => [...prev, { name, value: '' }])
  const removeRow = (index: number) => setRows((prev) => prev.filter((_row, i) => i !== index))

  const save = async () => {
    setError('')
    const map: EngineEnvOverrides = {}
    for (const row of rows) map[row.name] = row.value
    try {
      // Normalise locally for the same error text the main process would give, then persist there.
      normalizeEngineEnvOverrides(map)
      const next = await api.engineSetEnvOverrides(map)
      setSaved(next); setRows(toRows(next))
    } catch (e) { setError(String((e as Error)?.message ?? e)) } finally { setSaving(false) }
  }

  const limitBytes = engine?.status.memoryLimitBytes ?? null

  return (
    <SettingsPane>
      <Section title={tx('Engine variables')} desc={tx('Environment variables passed to the built-in TensorFold engine process. Use these to raise a model load limit or adjust engine behaviour without leaving the app.')}>
        {limitBytes != null && (
          <p className="mb-3 text-[13px] text-ink-3">
            {tx('Current app-side load limit:')} <span className="font-medium text-ink-2">{formatBytes(limitBytes)}</span>
            {engine?.status.model ? ` · ${engine.status.model}` : ''}
          </p>
        )}
        {rows.length > 0 ? (
          <div className="space-y-3">
            {rows.map((row, index) => (
              <div key={index} className="flex items-start gap-2">
                <div className="min-w-0 flex-1 space-y-1.5">
                  <Input
                    value={row.name}
                    spellCheck={false}
                    placeholder="VARIABLE_NAME"
                    className="font-mono"
                    onChange={(e) => patchRow(index, { name: e.target.value.toUpperCase().replace(/[^A-Z0-9_]/g, '') })}
                  />
                  <Input
                    value={row.value}
                    spellCheck={false}
                    placeholder="value"
                    className="font-mono"
                    onChange={(e) => patchRow(index, { value: e.target.value })}
                  />
                  {isDeniedEngineEnvKey(row.name) && <p className="text-[12px] text-red-500">{tx('{value0} is managed by TensorFold or the system.', { value0: row.name })}</p>}
                </div>
                <Button variant="ghost" size="xs" icon="trash" onClick={() => removeRow(index)} aria-label={tx('Remove {value0}', { value0: row.name || `#${index + 1}` })} />
              </div>
            ))}
          </div>
        ) : (
          <p className="rounded-lg border border-dashed border-line px-3 py-4 text-[13px] text-ink-3">{tx('No variables set. The engine runs with its defaults.')}</p>
        )}
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <Button variant="default" size="sm" icon="plus" onClick={() => addRow()}>{tx('Add variable')}</Button>
          <Button variant="primary" size="sm" disabled={!changed || saving || invalidName || incomplete || duplicate} onClick={() => { setSaving(true); void save() }}>{tx('Save')}</Button>
          {changed && <span className="text-[12px] text-ink-3">{tx('Unsaved changes')}</span>}
        </div>
        {duplicate && <p className="mt-2 text-[12px] text-red-500">{tx('Each variable can appear only once.')}</p>}
        {error && <p className="mt-2 text-[13px] text-red-500">{error}</p>}
        <p className="mt-3 flex items-start gap-1.5 text-[12px] leading-relaxed text-ink-3">
          <Icon name="sliders" size={13} className="mt-0.5 shrink-0" />
          <span>{engine?.status.state === 'ready' ? tx('Saved values reach each engine process at its next launch; the loaded model keeps the values it started with until it reloads.') : tx('Saved values take effect the next time a model loads.')}</span>
        </p>
        <div className="mt-4 rounded-xl border border-line bg-card shadow-card">
          <div className="border-b border-line px-3.5 py-2.5 text-[13px] font-medium text-ink-2">{tx('Common variables')}</div>
          <div className="space-y-2.5 px-3.5 py-3">
            {ENGINE_ENV_PRESETS.map((preset) => (
              <div key={preset.key} className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="font-mono text-[12px] font-medium text-ink">{preset.key}</div>
                  <div className="mt-0.5 text-[12px] leading-snug text-ink-3">{tx(preset.desc)}</div>
                </div>
                <Button variant="ghost" size="xs" icon="plus" disabled={added.has(preset.key)} onClick={() => addRow(preset.key)} aria-label={tx('Add {value0}', { value0: preset.key })} />
              </div>
            ))}
          </div>
        </div>
        <p className="mt-3 flex items-start gap-1.5 text-[12px] leading-relaxed text-ink-3">
          <Icon name="shield" size={13} className="mt-0.5 shrink-0" />
          <span>{tx('A load limit above the 70% default leaves less memory for everything else on this Mac. Raise it only when the engine reported that the model just barely did not fit.')}</span>
        </p>
      </Section>
    </SettingsPane>
  )
}