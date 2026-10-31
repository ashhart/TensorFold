// The result card to share as an image: drawn from the run alone, 1200 × 675 at twice the pixels, on a canvas.
import { saveBlob, type BenchRun } from './api'
import { secs } from './units'

const W = 1200, H = 675, SCALE = 2
const KIND: Record<string, string> = { quick: 'Quick score', profile: 'Usage profile', prefill: 'Prefill sweep', kl: 'KL divergence' }
const C = { bg: '#FFFFFF', ink: '#1A1C21', ink2: '#5B6270', ink3: '#666C77', line: '#E2E5EA', accent: '#5157E4', soft: '#ECEDFC', good: '#157A4B', bad: '#BB3046' }
const SANS = '"Instrument Sans", -apple-system, "Segoe UI", system-ui, sans-serif'

const n = (x: number | null | undefined) => (x == null ? '—' : x >= 20 ? Math.round(x).toLocaleString('en') : x.toFixed(1))
const k = (size: number) => (size >= 1024 ? `${size / 1024}k` : String(size))
const joules = (x: number | null | undefined) => (x == null ? '—' : x < 1 ? `${Math.round(x * 1000)} mJ` : `${x.toFixed(2)} J`)

type Fig = { label: string; value: string; color?: string }

/** The headline figures of a run, the same ones the report puts first. */
export function cardFigures(run: BenchRun): Fig[] {
  const o = run.summary?.overall
  const hw = run.hardware
  if (run.mode === 'prefill') {
    const pts = (run.summary?.sweep ?? []).filter((p) => !p.error && p.prefill_tps != null)
    if (!pts.length) return []
    const a = pts[0], b = pts[pts.length - 1]
    return [{ label: `Prompt reading at ${k(a.size)}`, value: `${n(a.prefill_tps)} tok/s` }, { label: `at ${k(b.size)}`, value: `${n(b.prefill_tps)} tok/s` },
      { label: `First token at ${k(b.size)}`, value: secs(b.ttft) }]
  }
  if (!o) return []
  const figs: Fig[] = [{ label: 'Writing', value: `${n(o.tps)} tok/s`, color: C.accent }]
  if (o.tps_no_drafts != null) figs.push({ label: 'Without drafts', value: `${n(o.tps_no_drafts)} tok/s` })
  if (o.checked > 0) figs.push({ label: 'Identical answers', value: `${o.exact} of ${o.checked}`, color: o.exact === o.checked ? C.good : C.bad })
  if (hw?.j_per_token != null) figs.push({ label: hw.whole_machine ? 'Energy per token' : 'Energy per token (est.)', value: joules(hw.j_per_token) })
  else figs.push({ label: 'First token', value: secs(o.ttft) })
  return figs.slice(0, 4)
}

function fit(ctx: CanvasRenderingContext2D, text: string, max: number, size: number, weight = 650): number {
  let s = size
  ctx.font = `${weight} ${s}px ${SANS}`
  while (ctx.measureText(text).width > max && s > 22) { s -= 2; ctx.font = `${weight} ${s}px ${SANS}` }
  return s
}

function loadLogo(): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => { const img = new Image(); img.onload = () => resolve(img); img.onerror = () => resolve(null); img.src = '/logo.png' })
}

export async function drawCard(run: BenchRun, names: Record<string, string> = {}): Promise<HTMLCanvasElement> {
  await document.fonts?.ready
  const logo = await loadLogo()
  const canvas = document.createElement('canvas')
  canvas.width = W * SCALE; canvas.height = H * SCALE
  const ctx = canvas.getContext('2d')!
  ctx.scale(SCALE, SCALE)
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H)
  const band = ctx.createLinearGradient(0, 0, W, 0)
  band.addColorStop(0, '#F08C6A'); band.addColorStop(0.55, '#8B5CF6'); band.addColorStop(1, '#3B82F6')
  ctx.fillStyle = band; ctx.fillRect(0, 0, W, 10)

  const L = 72
  if (logo) ctx.drawImage(logo, L, 54, 44, 44)
  ctx.fillStyle = C.ink; ctx.font = `700 22px ${SANS}`; ctx.fillText('TensorFold Web', L + 58, 74)
  ctx.fillStyle = C.ink2; ctx.font = `500 17px ${SANS}`; ctx.fillText(`Bench · ${KIND[run.mode] ?? run.mode}`, L + 58, 97)

  const title = run.model ?? 'model'
  fit(ctx, title, W - 2 * L, 46)
  ctx.fillStyle = C.ink; ctx.fillText(title, L, 186)
  const m = run.machine
  const sub = `${m?.chip ?? 'unknown machine'}${m?.ram_bytes ? `, ${Math.round(m.ram_bytes / 1024 ** 3)} GB` : ''} · ${run.engine ?? 'engine not identified'}`
  fit(ctx, sub, W - 2 * L, 22, 500)
  ctx.fillStyle = C.ink2; ctx.fillText(sub, L, 224)

  const figs = cardFigures(run)
  const colW = (W - 2 * L) / Math.max(3, figs.length)
  figs.forEach((f, i) => {
    const x = L + i * colW
    ctx.fillStyle = C.ink2; ctx.font = `500 19px ${SANS}`; ctx.fillText(f.label, x, 318)
    fit(ctx, f.value, colW - 24, 54, 700)
    ctx.fillStyle = f.color ?? C.ink; ctx.fillText(f.value, x, 384)
  })

  // the suites as small bars under the figures, when the run has several
  const suites = Object.entries(run.summary?.suites ?? {}).filter(([, b]) => b.tps != null)
  if (run.mode !== 'prefill' && suites.length >= 2) {
    const top = Math.max(...suites.map(([, b]) => b.tps ?? 0)) || 1
    const bw = (W - 2 * L) / suites.length
    suites.forEach(([id, b], i) => {
      const h = 70 * ((b.tps ?? 0) / top), x = L + i * bw
      ctx.fillStyle = C.soft; ctx.fillRect(x, 430, bw - 16, 70)
      ctx.fillStyle = C.accent; ctx.fillRect(x, 500 - h, bw - 16, h)
      ctx.fillStyle = C.ink2; ctx.font = `500 14px ${SANS}`
      const label = names[id] ?? id
      ctx.fillText(`${label} ${n(b.tps)}`, x, 524, bw - 16)
    })
  }

  ctx.fillStyle = C.line; ctx.fillRect(L, H - 96, W - 2 * L, 1)
  const sv = run.suites_version
  // thinking as the run recorded it: off only when one of the ways to turn it off was seen to work
  const think = run.thinking ? (run.thinking.off ? 'thinking off' : 'thinking on') : null
  const proto = `Temperature 0, warm-up${think ? `, ${think}` : ''}${sv ? `, suites v${sv.version} ${sv.digest}` : ''}${run.hardware?.throttle?.throttled === false ? ', no slowdown' : run.hardware?.throttle?.throttled ? ', slowed down during the run' : ''}`
  fit(ctx, proto, W - 2 * L - 220, 17, 500)
  ctx.fillStyle = C.ink3; ctx.fillText(proto, L, H - 56)
  ctx.font = `500 17px ${SANS}`; ctx.textAlign = 'right'
  ctx.fillText(new Date(run.created * 1000).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }), W - L, H - 56)
  return canvas
}

export function cardBlob(run: BenchRun, names: Record<string, string> = {}): Promise<Blob> {
  return drawCard(run, names).then((c) => new Promise<Blob>((resolve, reject) => c.toBlob((b) => (b ? resolve(b) : reject(new Error('the card could not be drawn'))), 'image/png')))
}


/** Copies text, with the old way when the page is not a secure context (plain http on the LAN). */
export async function copyText(text: string): Promise<void> {
  if (window.isSecureContext && navigator.clipboard?.writeText) return navigator.clipboard.writeText(text)
  const ta = Object.assign(document.createElement('textarea'), { value: text })
  ta.style.position = 'fixed'; ta.style.opacity = '0'
  document.body.appendChild(ta); ta.select()
  const ok = document.execCommand('copy')
  ta.remove()
  if (!ok) throw new Error('the browser refused to copy')
}

/** Copies the card as an image when the browser allows it; otherwise downloads it. Says which it did. */
export async function shareImage(run: BenchRun, names: Record<string, string> = {}): Promise<'copied' | 'downloaded'> {
  const name = `tensorfold-bench-${run.id}.png`
  if (window.isSecureContext && 'ClipboardItem' in window && navigator.clipboard?.write) {
    try {
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': cardBlob(run, names) })])
      return 'copied'
    } catch { /* refused: fall through to a download */ }
  }
  saveBlob(await cardBlob(run, names), name)
  return 'downloaded'
}
