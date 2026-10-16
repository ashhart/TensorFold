import assert from 'node:assert/strict'

// Mirror of the BarVisualization width computation in visualization.tsx.
// Pure logic extracted to verify the NaN/Infinity guard for max === 0.
function barPct(bars: { value: number; max?: number }[]): number[] {
  const rows = bars.length ? bars : [{ value: 0 } as { value: number; max?: number }]
  const max = Math.max(1, ...rows.map((b) => b.max ?? b.value))
  return rows.map((bar) => {
    const denom = bar.max && bar.max > 0 ? bar.max : max
    const ratio = (bar.value / denom) * 100
    return Number.isFinite(ratio) ? Math.max(0, Math.min(100, ratio)) : 0
  })
}

// Regression: bar.max === 0 with value 0 must not produce NaN.
{
  const [pct] = barPct([{ value: 0, max: 0 }])
  assert.equal(Number.isFinite(pct), true, 'pct must be finite for max=0,value=0')
  assert.equal(pct, 0)
}

// bar.max === 0 with positive value must not produce Infinity, falls back to global max.
{
  const [a, b] = barPct([{ value: 5, max: 0 }, { value: 10 }])
  assert.equal(Number.isFinite(a), true)
  // global max is 10 (from second bar's value), so 5/10*100 = 50
  assert.equal(a, 50)
  assert.equal(b, 100)
}

// Normal explicit max still respected.
{
  const [pct] = barPct([{ value: 25, max: 50 }])
  assert.equal(pct, 50)
}

// Value exceeding max is clamped to 100.
{
  const [pct] = barPct([{ value: 200, max: 50 }])
  assert.equal(pct, 100)
}

// Negative max treated as global-max fallback.
{
  const [pct] = barPct([{ value: 4, max: -3 }])
  assert.equal(Number.isFinite(pct), true)
  assert.equal(pct, 100) // global max = value = 4
}

console.log('visualization bar pct: all assertions passed')
