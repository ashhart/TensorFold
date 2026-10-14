import type { ThemePalette, ThemePreset } from './types'

// Shared between main (defaults) and renderer (UI + runtime application).
export const DEFAULT_UI_FONT =
  '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
export const DEFAULT_CODE_FONT =
  'ui-monospace, "SF Mono", SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace'

const f = { uiFont: DEFAULT_UI_FONT, codeFont: DEFAULT_CODE_FONT }

// Every preset is built from the TensorFold brand art: graphite and navy surfaces, the electric
// blue primary and the violet from the fold gradient.
export const PRESETS: Record<Exclude<ThemePreset, 'custom'>, { light: ThemePalette; dark: ThemePalette }> = {
  // The default: near-neutral graphite surfaces with a hint of the brand blue, electric-blue
  // primary for actions and selection. The fold gradient (index.css) is kept for the logo and progress.
  tensorfold: {
    light: { accent: '#3451E0', info: '#5B4BD8', background: '#FBFBFC', foreground: '#17181C', ...f, translucent: false, frosted: false, contrast: 45 },
    dark: { accent: '#5A78FF', info: '#8EA0FF', background: '#111216', foreground: '#ECEDF1', ...f, translucent: false, frosted: false, contrast: 50 },
  },
  // The saturated navy of the brand art (the first beta's default).
  navy: {
    light: { accent: '#2F4BE0', info: '#6B3FD9', background: '#F6F7FC', foreground: '#0B1240', ...f, translucent: false, frosted: false, contrast: 45 },
    dark: { accent: '#4D6BFF', info: '#9A7BFF', background: '#060A24', foreground: '#EEF1FF', ...f, translucent: false, frosted: false, contrast: 55 },
  },
  // Translucent surfaces (native vibrancy blurs the desktop through the window) on the brand violet.
  glass: {
    light: { accent: '#6B3FD9', info: '#3451E0', background: '#F4F3FA', foreground: '#1A1830', ...f, translucent: true, contrast: 44 },
    dark: { accent: '#9A7BFF', info: '#8EA0FF', background: '#141320', foreground: '#ECEAF6', ...f, translucent: true, contrast: 54 },
  },
}

export const PRESET_LABELS: Record<Exclude<ThemePreset, 'custom'>, string> = {
  tensorfold: 'TensorFold',
  navy: 'Navy',
  glass: 'Glass',
}
