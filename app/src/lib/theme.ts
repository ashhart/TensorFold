import type { ThemePalette } from './types'

interface RGB { r: number; g: number; b: number }

function hexToRgb(hex: string): RGB {
  let h = hex.replace('#', '').trim()
  if (h.length === 3) h = h.split('').map((c) => c + c).join('')
  const n = parseInt(h || '000000', 16)
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 }
}
const clamp = (n: number) => Math.max(0, Math.min(255, Math.round(n)))
const toHex = ({ r, g, b }: RGB) => '#' + [r, g, b].map((v) => clamp(v).toString(16).padStart(2, '0')).join('')
function mix(a: RGB, b: RGB, t: number): RGB {
  return { r: a.r + (b.r - a.r) * t, g: a.g + (b.g - a.g) * t, b: a.b + (b.b - a.b) * t }
}
function luminance({ r, g, b }: RGB): number {
  const f = (c: number) => { const s = c / 255; return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4) }
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
}

const BLACK: RGB = { r: 13, g: 13, b: 13 }
const WHITE: RGB = { r: 255, g: 255, b: 255 }

// Derive the full token set from a palette. Surfaces are mixed from the
// background toward the foreground; separation + border weight scale with contrast.
export function buildThemeVars(p: ThemePalette, isDark: boolean): string {
  const fg = hexToRgb(p.foreground)
  const bg = hexToRgb(p.background)
  const c = Math.max(0, Math.min(1, p.contrast / 100))
  const sep = 0.022 + c * 0.06

  // In dark the content is the darkest surface and the chrome and cards lift toward the
  // foreground; in light the content is near-white with cards above it and chrome recessed.
  let chrome: RGB, canvas: RGB, card: RGB, card2: RGB
  if (isDark) {
    canvas = bg
    card = mix(bg, fg, sep * 0.35)
    chrome = mix(bg, fg, sep * 0.7)
    card2 = mix(bg, fg, sep * 0.85)
  } else {
    canvas = bg
    card = mix(bg, WHITE, 0.85)
    chrome = mix(bg, fg, sep * 0.6)
    card2 = mix(bg, fg, sep * 0.45)
  }

  const lineA = (0.05 + c * 0.11).toFixed(3)
  const lineStrongA = (0.09 + c * 0.16).toFixed(3)

  const accent = hexToRgb(p.accent)
  const accentInk = luminance(accent) > 0.55 ? '#0d0d0d' : '#ffffff'
  const accentStrong = toHex(mix(accent, isDark ? WHITE : BLACK, 0.14))
  const accentSoft = toHex(mix(accent, bg, isDark ? 0.84 : 0.88))
  // --info carries toggles, links and focus; primaries use the palette's accent.
  const info = p.info ?? p.accent

  return [
    `--ink:${fg.r} ${fg.g} ${fg.b}`,
    `--accent:${p.accent}`,
    `--accent-strong:${accentStrong}`,
    `--accent-ink:${accentInk}`,
    `--accent-soft:${accentSoft}`,
    `--info:${info}`,
    `--info-ink:${toHex(mix(hexToRgb(info), isDark ? WHITE : BLACK, 0.22))}`,
    `--bg:${toHex(chrome)}`,
    `--canvas:${toHex(canvas)}`,
    `--card:${toHex(card)}`,
    `--card-2:${toHex(card2)}`,
    `--sidebar-surface:${p.frosted === false ? toHex(chrome) : `color-mix(in srgb, ${toHex(chrome)} 72%, transparent)`}`,
    `--line:rgb(${fg.r} ${fg.g} ${fg.b} / ${lineA})`,
    `--line-strong:rgb(${fg.r} ${fg.g} ${fg.b} / ${lineStrongA})`,
    `--font-sans:${p.uiFont}`,
    `--font-mono:${p.codeFont}`,
  ].join(';')
}

// Inject the resolved-mode vars (overrides base :root/.dark) + toggle classes.
export function applyTheme(palette: ThemePalette, isDark: boolean, pointerCursors: boolean): void {
  let el = document.getElementById('vontra-theme') as HTMLStyleElement | null
  if (!el) {
    el = document.createElement('style')
    el.id = 'vontra-theme'
    document.head.appendChild(el)
  }
  el.textContent = `:root,.dark{${buildThemeVars(palette, isDark)}}`
  const root = document.documentElement
  root.classList.toggle('translucent', !!palette.translucent)
  root.classList.toggle('pointer-cursors', !!pointerCursors)
}

export const swatchTextColor = (hex: string) => (luminance(hexToRgb(hex)) > 0.55 ? '#0d0d0d' : '#ffffff')
