// Terminal workspace layout (VST-039), renderer side: the pane-grid presets driven by the
// user's choice, persisted per project. Mirrors the tested layout core in
// electron/agent/terminal-multiplex.ts (kept renderer-local so the grid needn't import
// host runtime across the type boundary). The CSS grid template for each preset lives here
// too, so the grid is a pure function of (preset, pane order).

export type LayoutPreset = 'single' | 'two-columns' | 'two-rows' | 'top-main' | 'left-main' | 'four'

// Icons are limited to what the renderer's icon set actually exports (grid/sidebar/monitor/
// file) — the label carries the real distinction.
export const LAYOUT_PRESETS: { value: LayoutPreset; label: string; icon: string; slots: number }[] = [
  { value: 'single', label: 'Single', icon: 'monitor', slots: 1 },
  { value: 'two-columns', label: 'Columns', icon: 'sidebar', slots: 2 },
  { value: 'two-rows', label: 'Rows', icon: 'file', slots: 2 },
  { value: 'top-main', label: 'Top + side', icon: 'grid', slots: 3 },
  { value: 'left-main', label: 'Left + side', icon: 'sidebar', slots: 3 },
  { value: 'four', label: 'Grid of 4', icon: 'grid', slots: 4 },
]

const SLOTS: Record<LayoutPreset, number> = { 'single': 1, 'two-columns': 2, 'two-rows': 2, 'top-main': 3, 'left-main': 3, 'four': 4 }

export function presetSlots(preset: LayoutPreset): number {
  return SLOTS[preset] ?? 1
}

export function isLayoutPreset(value: unknown): value is LayoutPreset {
  return typeof value === 'string' && value in SLOTS
}

// CSS grid style for a preset. `top-main`/`left-main` use named template areas so the first
// pane is the large one and the rest stack alongside; the others are even columns/rows.
export function layoutGridStyle(preset: LayoutPreset): React.CSSProperties {
  switch (preset) {
    case 'single':
      return { gridTemplateColumns: '1fr', gridTemplateRows: '1fr' }
    case 'two-columns':
      return { gridTemplateColumns: '1fr 1fr', gridTemplateRows: '1fr' }
    case 'two-rows':
      return { gridTemplateColumns: '1fr', gridTemplateRows: '1fr 1fr' }
    case 'top-main':
      return { gridTemplateColumns: '1fr 1fr', gridTemplateRows: '2fr 1fr', gridTemplateAreas: '"main main" "a b"' }
    case 'left-main':
      return { gridTemplateColumns: '2fr 1fr', gridTemplateRows: '1fr 1fr', gridTemplateAreas: '"main a" "main b"' }
    case 'four':
      return { gridTemplateColumns: '1fr 1fr', gridTemplateRows: '1fr 1fr' }
    default:
      return { gridTemplateColumns: '1fr', gridTemplateRows: '1fr' }
  }
}

// The grid-area name for pane index `i` under a preset with named areas (main/a/b); '' for
// presets that flow naturally.
export function paneGridArea(preset: LayoutPreset, index: number): string {
  if (preset !== 'top-main' && preset !== 'left-main') return ''
  return index === 0 ? 'main' : index === 1 ? 'a' : index === 2 ? 'b' : ''
}

const STORAGE_KEY = 'vontra.terminal.layout'

// Per-project persisted preset. localStorage keeps it renderer-local (a UI preference, not
// host state) and survives reloads.
export function loadLayoutPreset(projectId: string): LayoutPreset {
  try {
    const map = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')
    const value = map?.[projectId]
    return isLayoutPreset(value) ? value : 'single'
  } catch {
    return 'single'
  }
}

export function saveLayoutPreset(projectId: string, preset: LayoutPreset): void {
  try {
    const map = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')
    map[projectId] = preset
    localStorage.setItem(STORAGE_KEY, JSON.stringify(map))
  } catch { /* private mode: layout resets on reload, harmless */ }
}
