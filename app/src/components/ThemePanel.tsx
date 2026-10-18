import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import { useStore } from '../lib/store'
import { swatchTextColor } from '../lib/theme'
import { highlight } from '../lib/highlight'
import { Icon } from '../lib/icons'
import { Menu, MenuItem, Segmented, Toggle } from './ui'
import { PRESETS, PRESET_LABELS } from '../../shared/theme-presets'
import type { Settings, ThemePalette, ThemePreset } from '../lib/types'
import clsx from 'clsx'

function normHex(v: string): string {
  let h = v.trim().replace(/^#/, '')
  if (h.length === 3) h = h.split('').map((c) => c + c).join('')
  if (!/^[0-9a-fA-F]{6}$/.test(h)) return '#000000'
  return '#' + h
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 border-t border-line py-2.5 first:border-t-0">
      <span className="text-[13px] text-ink">{label}</span>
      <div className="flex items-center gap-2.5">{children}</div>
    </div>
  )
}

function ColorRow({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <Row label={label}>
      <label
        className="no-drag relative flex h-8 min-w-[150px] cursor-pointer items-center gap-2 rounded-full border border-line px-3"
        style={{ background: value, color: swatchTextColor(value) }}
      >
        <span className="h-3.5 w-3.5 rounded-full border" style={{ borderColor: swatchTextColor(value), opacity: 0.45 }} />
        <span className="flex-1 text-right font-mono text-[13px] tracking-tight tabular-nums">{value.toUpperCase()}</span>
        <input type="color" value={normHex(value)} onChange={(e) => onChange(e.target.value)} className="swatch-input absolute inset-0 h-full w-full opacity-0" />
      </label>
    </Row>
  )
}

function FontRow({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <Row label={label}>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        spellCheck={false}
        className="no-drag h-8 w-[230px] truncate rounded-lg border border-line bg-card-2 px-2.5 text-[13px] text-ink-3 outline-none focus:border-accent focus:text-ink"
      />
    </Row>
  )
}

function PresetDropdown({ accent, preset, onPick }: { accent: string; preset: ThemePreset; onPick: (p: Exclude<ThemePreset, 'custom'>) => void }) {
  const tx = useAuthoredText()
  const label = preset === 'custom' ? tx("Custom") : PRESET_LABELS[preset]
  return (
    <Menu
      width={180}
      align="end"
      trigger={(open) => (
        <button className={clsx('flex h-8 items-center gap-2 rounded-lg border border-line px-2 text-[13px] transition-colors', open ? 'bg-ink-5 text-ink' : 'text-ink-2 hover:bg-ink-5')}>
          <span className="flex h-5 w-5 items-center justify-center rounded-md text-[11px] font-bold" style={{ background: accent, color: swatchTextColor(accent) }}>{"Aa"}</span>
          <span className="font-medium">{label}</span>
          <Icon name="chevronDown" size={13} className="text-ink-3" />
        </button>
      )}
    >
      {(close) => (
        <>
          {(Object.keys(PRESET_LABELS) as Array<Exclude<ThemePreset, 'custom'>>).map((p) => (
            <MenuItem key={p} active={preset === p} onClick={() => { onPick(p); close() }}>{PRESET_LABELS[p]}</MenuItem>
          ))}
        </>
      )}
    </Menu>
  )
}

function ThemeCard({
  title, palette, preset, onChange, onPreset, onImport, onCopy,
}: {
  title: string; palette: ThemePalette; preset: ThemePreset
  onChange: (patch: Partial<ThemePalette>) => void
  onPreset: (p: Exclude<ThemePreset, 'custom'>) => void
  onImport: () => void; onCopy: () => void
}) {
  const tx = useAuthoredText()
  return (
    <div className="rounded-2xl border border-line bg-card-2/50 p-1.5">
      <div className="flex items-center justify-between px-2.5 pb-1.5 pt-1">
        <span className="text-[13px] font-medium text-ink-2">{title}</span>
        <div className="flex items-center gap-2.5 text-[13px] text-ink-3">
          <button onClick={onImport} className="transition-colors hover:text-ink">{tx("Import")}</button>
          <button onClick={onCopy} className="transition-colors hover:text-ink">{tx("Copy theme")}</button>
          <PresetDropdown accent={palette.accent} preset={preset} onPick={onPreset} />
        </div>
      </div>
      <div className="rounded-xl bg-card px-3 shadow-card">
        <ColorRow label={tx("Accent")} value={palette.accent} onChange={(v) => onChange({ accent: v })} />
        <ColorRow label={tx("Background")} value={palette.background} onChange={(v) => onChange({ background: v })} />
        <ColorRow label={tx("Foreground")} value={palette.foreground} onChange={(v) => onChange({ foreground: v })} />
        <FontRow label={tx("UI font")} value={palette.uiFont} onChange={(v) => onChange({ uiFont: v })} />
        <FontRow label={tx("Code font")} value={palette.codeFont} onChange={(v) => onChange({ codeFont: v })} />
        <Row label={tx("Translucent sidebar")}><Toggle label={tx("Translucent sidebar")} checked={palette.translucent} onChange={(v) => onChange({ translucent: v })} /></Row>
        <Row label={tx("Contrast")}>
          <input type="range" min={0} max={100} value={palette.contrast} onChange={(e) => onChange({ contrast: +e.target.value })}
            className="h-1.5 w-[180px] cursor-pointer appearance-none rounded-full bg-ink-5 accent-[var(--accent)]" />
          <span className="w-7 text-right font-mono text-[13px] tabular-nums text-ink-2">{palette.contrast}</span>
        </Row>
      </div>
    </div>
  )
}

function diffLines(base: ThemePalette, cur: ThemePalette) {
  const surf = (p: ThemePalette) => (p.translucent ? '"translucent"' : '"solid"')
  return [
    { old: 'const themePreview: ThemeConfig = {', neu: 'const themePreview: ThemeConfig = {', changed: false },
    { old: `  surface: ${surf(base)},`, neu: `  surface: ${surf(cur)},`, changed: base.translucent !== cur.translucent },
    { old: `  accent: "${base.accent.toUpperCase()}",`, neu: `  accent: "${cur.accent.toUpperCase()}",`, changed: base.accent.toLowerCase() !== cur.accent.toLowerCase() },
    { old: `  contrast: ${base.contrast},`, neu: `  contrast: ${cur.contrast},`, changed: base.contrast !== cur.contrast },
    { old: '};', neu: '};', changed: false },
  ]
}

function DiffPreview({ base, cur }: { base: ThemePalette; cur: ThemePalette }) {
  const lines = diffLines(base, cur)
  const Col = ({ side }: { side: 'old' | 'neu' }) => (
    <div className="min-w-0 flex-1">
      {lines.map((l, i) => {
        const text = side === 'old' ? l.old : l.neu
        const tint = l.changed ? (side === 'old' ? 'bg-[var(--del-bg)]' : 'bg-[var(--add-bg)]') : ''
        const sign = l.changed ? (side === 'old' ? '-' : '+') : ' '
        const signColor = l.changed ? (side === 'old' ? 'text-[var(--del-ink)]' : 'text-[var(--add-ink)]') : 'text-transparent'
        return (
          <div key={i} className={clsx('flex items-center', tint)}>
            <span className="w-7 shrink-0 select-none pr-2 text-right text-ink-4">{i + 1}</span>
            <span className={clsx('w-3 shrink-0 select-none text-center', signColor)}>{sign}</span>
            <span className="syn-code overflow-hidden whitespace-pre">{highlight(text)}</span>
          </div>
        )
      })}
    </div>
  )
  return (
    <div className="overflow-hidden rounded-xl border border-line bg-card-2 font-mono text-[13px] leading-[1.7]">
      <div className="flex divide-x divide-line py-1.5">
        <Col side="old" />
        <Col side="neu" />
      </div>
    </div>
  )
}

export function ThemePanel() {
  const tx = useAuthoredText()
  const settings = useStore((s) => s.settings)!
  const save = useStore((s) => s.saveSettings)
  const resolved = useStore((s) => s.resolvedTheme)

  const commit = (patch: Partial<Settings>) => save({ ...settings, ...patch })
  const applyPreset = (p: Exclude<ThemePreset, 'custom'>) => commit({ preset: p, light: { ...PRESETS[p].light }, dark: { ...PRESETS[p].dark } })
  const editPalette = (mode: 'light' | 'dark', patch: Partial<ThemePalette>) =>
    commit({ preset: 'custom', [mode]: { ...settings[mode], ...patch } } as Partial<Settings>)

  const copyTheme = (mode: 'light' | 'dark') => { navigator.clipboard?.writeText(JSON.stringify(settings[mode], null, 2))?.catch(() => {}) }
  const importTheme = (mode: 'light' | 'dark') => {
    const raw = window.prompt(tx("Paste {value0} theme JSON", { value0: mode }))
    if (!raw) return
    try {
      const p = JSON.parse(raw)
      if (p && typeof p.accent === 'string') commit({ preset: 'custom', [mode]: { ...settings[mode], ...p } } as Partial<Settings>)
    } catch { /* ignore malformed */ }
  }

  const previewPalette = resolved === 'dark' ? settings.dark : settings.light
  const previewBase = PRESETS.tensorfold[resolved]

  return (
    <section className="mb-9">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-[15px] font-semibold tracking-[-0.01em] text-ink">{tx("Theme")}</h2>
          <p className="mt-0.5 text-[13px] text-ink-3">{tx("Use light, dark, or match your system.")}</p>
        </div>
        <Segmented
          value={settings.theme}
          onChange={(v) => commit({ theme: v })}
          options={[{ value: 'light', label: tx("Light"), icon: 'sun' }, { value: 'dark', label: tx("Dark"), icon: 'moon' }, { value: 'system', label: tx("System"), icon: 'monitor' }]}
        />
      </div>

      <div className="mt-4">
        <DiffPreview base={previewBase} cur={previewPalette} />
      </div>

      <div className="mt-3 space-y-3">
        <ThemeCard
          title={tx("Light theme")} palette={settings.light} preset={settings.preset}
          onChange={(p) => editPalette('light', p)} onPreset={applyPreset}
          onImport={() => importTheme('light')} onCopy={() => copyTheme('light')}
        />
        <ThemeCard
          title={tx("Dark theme")} palette={settings.dark} preset={settings.preset}
          onChange={(p) => editPalette('dark', p)} onPreset={applyPreset}
          onImport={() => importTheme('dark')} onCopy={() => copyTheme('dark')}
        />
      </div>

      <div className="mt-3 flex items-center justify-between rounded-xl border border-line bg-card px-3.5 py-2.5 shadow-card">
        <div>
          <div className="text-[13px] text-ink">{tx("Use pointer cursors")}</div>
          <div className="text-[13px] text-ink-3">{tx("Show a hand cursor over clickable elements.")}</div>
        </div>
        <Toggle label={tx("Use pointer cursors")} checked={settings.pointerCursors} onChange={(v) => commit({ pointerCursors: v })} />
      </div>

      <div className="mt-3 flex items-center justify-between rounded-xl border border-line bg-card px-3.5 py-2.5 shadow-card">
        <div>
          <div className="text-[13px] text-ink">{tx("Bold plan progress")}</div>
          <div className="text-[13px] text-ink-3">{tx("Render the plan panel with block progress bars and status glyphs.")}</div>
        </div>
        <Toggle label={tx("Bold plan progress")} checked={settings.doomTodos !== false} onChange={(v) => commit({ doomTodos: v })} />
      </div>
    </section>
  )
}
