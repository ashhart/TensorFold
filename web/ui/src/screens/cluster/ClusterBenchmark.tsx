import { useState } from 'react'
import type { ClusterBenchmark } from '../../clusterTypes'

type Props = {
  benchmarks: ClusterBenchmark[]
  activeJob: ClusterBenchmark | null
  canRun: boolean
  onRun: (opts: { prompt_tokens: number[]; decode_tokens: number; draft: boolean }) => void
}

const BAR_COLORS = ['#3d6bff', '#8b4dff', '#ff7a5c', '#22d3ee', '#3ccf91', '#f2b234']

export default function ClusterBenchmark({ benchmarks, activeJob, canRun, onRun }: Props) {
  const [promptSet, setPromptSet] = useState<'quick' | 'full'>('quick')
  const [draft, setDraft] = useState(true)

  const doneRuns = benchmarks.filter((b) => b.state === 'done' && (b.points?.length ?? 0) > 0)
  const maxPrefill = Math.max(1, ...doneRuns.map((b) => b.peak_prefill_tps || 0))
  const maxDecode = Math.max(1, ...doneRuns.map((b) => b.avg_decode_tps || 0))

  // Compute automatic 4-node vs 2-node speedup badges if both exist!
  const best2 = doneRuns.filter((b) => b.world === 2).sort((a, b) => (b.avg_decode_tps || 0) - (a.avg_decode_tps || 0))[0]
  const best4 = doneRuns.filter((b) => b.world === 4).sort((a, b) => (b.avg_decode_tps || 0) - (a.avg_decode_tps || 0))[0]
  const decodeGain = best2 && best4 && best2.avg_decode_tps && best4.avg_decode_tps
    ? (best4.avg_decode_tps / best2.avg_decode_tps).toFixed(2)
    : null
  const prefillGain = best2 && best4 && best2.peak_prefill_tps && best4.peak_prefill_tps
    ? (best4.peak_prefill_tps / best2.peak_prefill_tps).toFixed(2)
    : null

  return (
    <section className="cs-panel">
      <div className="cs-panel-head">
        <div>
          <h2 className="cs-panel-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 3v18h18" />
              <rect x="7" y="10" width="3" height="8" rx="1" />
              <rect x="12" y="6" width="3" height="12" rx="1" />
              <rect x="17" y="13" width="3" height="5" rx="1" />
            </svg>
            2-Node vs 4-Node Prefill &amp; Decode Benchmark Arena
            {decodeGain && <span className="cs-badge good">4N vs 2N Decode: {decodeGain}×</span>}
            {prefillGain && <span className="cs-badge master">4N vs 2N Prefill: {prefillGain}×</span>}
          </h2>
          <p className="cs-panel-sub">
            Cold prefill (cache-busted prompts) + sustained decode throughput across cluster topologies
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <div className="cs-seg">
            <button className={promptSet === 'quick' ? 'active' : ''} onClick={() => setPromptSet('quick')}>512 / 2K</button>
            <button className={promptSet === 'full' ? 'active' : ''} onClick={() => setPromptSet('full')}>512 / 2K / 8K</button>
          </div>
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: '0.78rem', cursor: 'pointer' }}>
            <input type="checkbox" checked={draft} onChange={(e) => setDraft(e.target.checked)} />
            MTP Draft
          </label>
          <button
            className="cs-btn primary sm"
            disabled={!canRun || activeJob?.state === 'running'}
            onClick={() => onRun({
              prompt_tokens: promptSet === 'quick' ? [512, 2048] : [512, 2048, 8192],
              decode_tokens: 128,
              draft,
            })}
          >
            {activeJob?.state === 'running'
              ? `Running ${activeJob.steps_done ?? 0}/${activeJob.steps_total ?? 0}…`
              : 'Run Cold Benchmark'}
          </button>
        </div>
      </div>

      {activeJob?.state === 'failed' && (
        <div className="notice bad">Benchmark failed: {activeJob.error}</div>
      )}

      {doneRuns.length === 0 ? (
        <div style={{ padding: '22px 12px', textAlign: 'center', color: 'var(--ink3)', fontSize: '0.84rem' }}>
          No cluster benchmarks recorded yet. Launch a 2-node or 4-node cluster above and click <strong>Run Cold Benchmark</strong> to chart Prefill &amp; Decode scaling!
        </div>
      ) : (
        <div className="cs-bench-grid">
          <div className="cs-bench-chart">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <strong style={{ fontSize: '0.86rem' }}>Decode Throughput (tok/s)</strong>
              <span style={{ fontSize: '0.72rem', color: 'var(--ink3)', fontFamily: 'var(--mono)' }}>Higher is faster</span>
            </div>
            {doneRuns.slice(-6).map((b, i) => {
              const v = b.avg_decode_tps || 0
              const pct = Math.max(8, (v / maxDecode) * 100)
              return (
                <div key={b.id} className="cs-bench-bar-row">
                  <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={`${b.label} · ${b.model}`}>
                    <strong>{b.label}</strong>
                    <div style={{ fontSize: '0.68rem', color: 'var(--ink3)' }}>{b.draft ? 'MTP ON' : 'Plain'} · {b.model}</div>
                  </div>
                  <div className="cs-bench-track">
                    <div className="cs-bench-fill" style={{ width: `${pct}%`, background: BAR_COLORS[i % BAR_COLORS.length] }}>
                      {v.toFixed(1)} tok/s
                    </div>
                  </div>
                  <div style={{ fontFamily: 'var(--mono)', textAlign: 'right', fontSize: '0.76rem' }}>
                    {b.points[0] ? `${b.points[0].ttft_ms}ms` : '—'}
                  </div>
                </div>
              )
            })}
          </div>

          <div className="cs-bench-chart">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <strong style={{ fontSize: '0.86rem' }}>Cold Prefill Throughput (tok/s)</strong>
              <span style={{ fontSize: '0.72rem', color: 'var(--ink3)', fontFamily: 'var(--mono)' }}>Peak prompt rate</span>
            </div>
            {doneRuns.slice(-6).map((b, i) => {
              const v = b.peak_prefill_tps || 0
              const pct = Math.max(8, (v / maxPrefill) * 100)
              return (
                <div key={b.id} className="cs-bench-bar-row">
                  <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={`${b.label} · ${b.model}`}>
                    <strong>{b.label}</strong>
                    <div style={{ fontSize: '0.68rem', color: 'var(--ink3)' }}>
                      {b.points.map((p) => `${p.prompt_tokens}t`).join(' / ')}
                    </div>
                  </div>
                  <div className="cs-bench-track">
                    <div className="cs-bench-fill" style={{ width: `${pct}%`, background: BAR_COLORS[i % BAR_COLORS.length] }}>
                      {v.toFixed(0)} tok/s
                    </div>
                  </div>
                  <div style={{ fontFamily: 'var(--mono)', textAlign: 'right', fontSize: '0.76rem' }}>
                    TP{b.tp}×PP{b.pp}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </section>
  )
}
