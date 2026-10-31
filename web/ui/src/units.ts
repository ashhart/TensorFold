// Units that roll over: a figure is shown in the unit that keeps it short, so 0.07 s reads 70 ms, 5,400 s reads
// 1 h 30 min, 0.4 GB reads 410 MB and 2,048 GB reads 2.0 TB. One place for every screen, the report card included.

export const GiB = 1024 ** 3

/** A duration in seconds: ms under a second, one decimal under a minute, then min and s, h and min, days and h. */
export function secs(x: number | null | undefined): string {
  if (x == null || !Number.isFinite(x)) return '—'
  if (x < 1) return `${Math.round(x * 1000)} ms`
  if (x < 59.95) return `${x.toFixed(1)} s`
  return clock(x)
}

/** Elapsed or remaining time, whole seconds: 45 s, 3 min 05 s, 1 h 02 min, 2 d 4 h. */
export function clock(s: number): string {
  const t = Math.max(0, Math.round(s))
  if (t < 60) return `${t} s`
  if (t < 3600) return `${Math.floor(t / 60)} min ${String(t % 60).padStart(2, '0')} s`
  if (t < 86400) return `${Math.floor(t / 3600)} h ${String(Math.floor((t % 3600) / 60)).padStart(2, '0')} min`
  return `${Math.floor(t / 86400)} d ${Math.floor((t % 86400) / 3600)} h`
}

/** Bytes in binary units, written MB, GB and TB as the rest of TensorFold Web does; ``digits`` applies to GB. */
export function bytes(b: number | null | undefined, digits = 0): string {
  if (b == null || !Number.isFinite(b)) return '—'
  const a = Math.abs(b)
  if (a >= 1024 * GiB) return `${(b / (1024 * GiB)).toFixed(a >= 10 * 1024 * GiB ? 0 : 1)} TB`
  if (a >= GiB || a === 0) return `${(b / GiB).toFixed(digits)} GB`
  if (a >= 1024 ** 2) return `${Math.round(b / 1024 ** 2)} MB`
  if (a >= 1024) return `${Math.round(b / 1024)} KB`
  return `${b} B`
}

/** A count kept short: 950, 1.2k, 12k, 3.4M, 1.1B. */
export function count(n: number): string {
  const a = Math.abs(n)
  if (a >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (a >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (a >= 1e4) return `${Math.round(n / 1e3)}k`
  if (a >= 1e3) return `${(n / 1e3).toFixed(1)}k`
  return String(n)
}
