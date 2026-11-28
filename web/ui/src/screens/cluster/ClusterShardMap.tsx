import { useState } from 'react'
import type { ClusterModel, ClusterPlan, ClusterSyncJob } from '../../clusterTypes'

type Props = {
  models: ClusterModel[]
  modelPath: string
  mode: 'auto' | 'tp' | 'pp' | 'hybrid'
  context: number
  parallel: number
  plan: ClusterPlan | null
  syncJobs: Record<string, ClusterSyncJob>
  deploying: boolean
  isDeployed: boolean
  onModelChange: (path: string) => void
  onModeChange: (m: 'auto' | 'tp' | 'pp' | 'hybrid') => void
  onContextChange: (ctx: number) => void
  onParallelChange: (p: number) => void
  onSyncModel: () => void
  onDeploy: () => void
  onStop: () => void
}

const fmtGiB = (b: number) => `${(b / 1024 ** 3).toFixed(1)} GiB`

export default function ClusterShardMap({
  models,
  modelPath,
  mode,
  context,
  parallel,
  plan,
  syncJobs,
  deploying,
  isDeployed,
  onModelChange,
  onModeChange,
  onContextChange,
  onParallelChange,
  onSyncModel,
  onDeploy,
  onStop,
}: Props) {
  const [hoverLayer, setHoverLayer] = useState<number | null>(null)
  const [showJson, setShowJson] = useState(false)

  const selectedModel = models.find((m) => m.path === modelPath) ?? models[0] ?? null
  const activeLayer = plan?.layers[hoverLayer ?? 3] ?? plan?.layers[0] ?? null
  const anySyncing = Object.values(syncJobs).some((j) => j.state === 'running')
  const needsSync = Boolean(
    selectedModel?.cluster_presence?.some((p) => p.node_id !== 'master' && (!p.complete || p.shards_present < selectedModel.shards_present)),
  )

  return (
    <section className="cs-panel">
      <div className="cs-panel-head">
        <div>
          <h2 className="cs-panel-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="4" width="18" height="4" rx="1" />
              <rect x="3" y="10" width="8" height="4" rx="1" />
              <rect x="13" y="10" width="8" height="4" rx="1" />
              <rect x="3" y="16" width="5" height="4" rx="1" />
              <rect x="10" y="16" width="5" height="4" rx="1" />
              <rect x="17" y="16" width="4" height="4" rx="1" />
            </svg>
            Heterogeneous Shard &amp; Layer Allocation Engine
          </h2>
          <p className="cs-panel-sub">
            {plan
              ? `${plan.model.name} (${plan.model.num_layers}L${plan.model.has_mtp ? ' + MTP' : ''} · ${plan.total_weights_gib} GiB) → ${plan.resolved_mode.toUpperCase()} (TP=${plan.tp} × PP=${plan.pp}${plan.heterogeneous ? ' · Heterogeneous' : ''}) · ${plan.effective_bandwidth_gbs} GB/s effective bandwidth`
              : 'Select nodes and a model to compute exact FP8 block-128 aligned sharding'}
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <button className="cs-btn sm" onClick={() => setShowJson((v) => !v)}>
            {showJson ? 'Hide rankN.json' : 'Inspect rankN.json'}
          </button>
          {isDeployed ? (
            <>
              <a className="cs-btn sm" href="#chat">Open Chat →</a>
              <button className="cs-btn danger" onClick={onStop}>Stop Cluster</button>
            </>
          ) : (
            <button
              className="cs-btn primary"
              disabled={deploying || !plan || !plan.all_fit}
              onClick={onDeploy}
            >
              {deploying
                ? 'Launching Cluster…'
                : plan
                  ? `Launch ${plan.world}-Node ${plan.resolved_mode.toUpperCase()} (TP=${plan.tp}×PP=${plan.pp})`
                  : 'Launch Cluster'}
            </button>
          )}
        </div>
      </div>

      {/* Controls Row */}
      <div className="cs-plan-controls">
        <div className="cs-field">
          <label>Checkpoint</label>
          <select className="cs-select" value={modelPath} onChange={(e) => onModelChange(e.target.value)}>
            {models.length === 0 && <option value={modelPath}>{modelPath || '~/models/GLM-5.3'}</option>}
            {models.map((m) => (
              <option key={m.path} value={m.path}>
                {m.name} ({m.format} · {m.shards_present}/{m.shards_expected} shards · {fmtGiB(m.bytes_on_disk)})
              </option>
            ))}
          </select>
        </div>

        <div className="cs-field">
          <label>Parallelism Strategy</label>
          <div className="cs-seg">
            {(['auto', 'tp', 'pp', 'hybrid'] as const).map((m) => (
              <button key={m} className={mode === m ? 'active' : ''} onClick={() => onModeChange(m)}>
                {m === 'auto' ? 'Auto' : m === 'tp' ? 'Tensor (TP)' : m === 'pp' ? 'Pipeline (PP)' : 'Hybrid TP×PP'}
              </button>
            ))}
          </div>
        </div>

        <div className="cs-field">
          <label>Context Window</label>
          <select className="cs-select" value={context} onChange={(e) => onContextChange(Number(e.target.value))}>
            <option value={4096}>4,096 tok</option>
            <option value={8192}>8,192 tok</option>
            <option value={16384}>16,384 tok</option>
            <option value={32768}>32,768 tok</option>
            <option value={65536}>65,536 tok</option>
            <option value={131072}>131,072 tok</option>
          </select>
        </div>

        <div className="cs-field">
          <label>Streams</label>
          <select className="cs-select" value={parallel} onChange={(e) => onParallelChange(Number(e.target.value))}>
            <option value={1}>1 stream</option>
            <option value={2}>2 streams</option>
            <option value={4}>4 streams</option>
          </select>
        </div>
      </div>

      {/* Model Download & TB5 Sync Presence Strip */}
      {selectedModel && (selectedModel.download || needsSync || anySyncing) && (
        <div className="cs-sync-strip">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
              <strong style={{ fontSize: '0.84rem' }}>{selectedModel.name} Weight Distribution</strong>
              {selectedModel.download && !selectedModel.download.complete && (
                <span className="cs-badge warn">
                  Master Downloading: {selectedModel.download.pct.toFixed(1)}% · {selectedModel.download.done_files}/{selectedModel.download.total_files} files · {selectedModel.download.rate_mib_s.toFixed(0)} MiB/s
                </span>
              )}
              {selectedModel.complete && <span className="cs-badge good">Master 100% Complete</span>}
            </div>
            <button className="cs-btn sm primary" disabled={anySyncing} onClick={onSyncModel}>
              {anySyncing ? 'Syncing over TB5…' : 'Sync Shards to Slaves over TB5'}
            </button>
          </div>

          {selectedModel.download && !selectedModel.download.complete && (
            <div className="cs-bar" style={{ height: 8 }}>
              <div className="cs-bar-fill" style={{ width: `${Math.min(100, selectedModel.download.pct)}%`, background: 'var(--brand-gradient)' }} />
            </div>
          )}

          <div className="cs-presence-row">
            {(selectedModel.cluster_presence ?? []).map((p) => {
              const sj = syncJobs[p.node_id]
              return (
                <div key={p.node_id} className="cs-presence-pill">
                  <span style={{ color: p.complete ? 'var(--positive)' : 'var(--warn)' }}>{p.complete ? '●' : '◔'}</span>
                  <span>{p.name}:</span>
                  <span>{p.shards_present}/{p.shards_expected}</span>
                  <span style={{ color: 'var(--ink3)' }}>({fmtGiB(p.bytes_on_disk)})</span>
                  {sj?.state === 'running' && (
                    <span style={{ color: 'var(--cs-cyan)' }}>syncing {sj.pct.toFixed(0)}% {sj.rate}</span>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* 79-Layer Interactive Shard Matrix */}
      {plan && (
        <>
          <div className="cs-layer-box">
            <div className="cs-layer-legend">
              <div className="cs-legend-items">
                {plan.ranks.map((r) => (
                  <span key={r.node_id} className="cs-legend-item">
                    <span className="cs-legend-swatch" style={{ background: r.color }} />
                    <strong>R{r.rank} {r.name}</strong>
                    <span style={{ color: 'var(--ink3)', fontFamily: 'var(--mono)' }}>({r.share_pct}%)</span>
                  </span>
                ))}
              </div>
              <div className="cs-legend-items" style={{ fontSize: '0.72rem', color: 'var(--ink3)' }}>
                <span>Top white edge = Dense (L0–{plan.model.num_dense_layers - 1})</span>
                <span>• White dot = Full DSA Indexer</span>
                {plan.model.has_mtp && <span style={{ color: 'var(--cs-amber)' }}>Gold border = MTP Draft (L{plan.model.num_layers})</span>}
              </div>
            </div>

            <div className="cs-layer-grid" role="group" aria-label="Per-layer shard allocation">
              {plan.layers.map((l) => (
                <div
                  key={l.layer}
                  className={`cs-layer-col ${l.kind} ${hoverLayer === l.layer ? 'active' : ''}`}
                  onMouseEnter={() => setHoverLayer(l.layer)}
                >
                  {l.shards.map((s) => (
                    <div
                      key={`${l.layer}-${s.rank}`}
                      className="cs-layer-seg"
                      style={{ height: `${s.pct}%`, background: s.color }}
                    />
                  ))}
                  {l.indexer === 'full' && <span className="cs-layer-indexer-dot" />}
                </div>
              ))}
            </div>

            {activeLayer && (
              <div className="cs-layer-hud">
                <div>
                  <strong>
                    {activeLayer.kind === 'mtp' ? `Layer ${activeLayer.layer} · MTP Speculative Draft` : `Layer ${activeLayer.layer} · ${activeLayer.kind === 'dense' ? 'Dense FFN' : `MoE (${plan.model.n_experts} experts, top-8)`}`}
                  </strong>
                  {' · '}Stage PP{activeLayer.pp_stage}
                  {' · '}DSA Indexer: <strong>{activeLayer.indexer.toUpperCase()}</strong>
                </div>
                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                  {activeLayer.shards.map((s) => (
                    <span key={s.rank} style={{ color: s.color }}>
                      R{s.rank} {s.name}: H[{s.heads}] MoE[{s.moe_ch}] ({s.pct}%)
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Per-Node Stacked Memory & Slice Breakdown */}
          <div className="cs-mem-grid">
            {plan.ranks.map((r) => {
              const cap = Math.max(1, r.ram_total_gib)
              const wPct = Math.min(100, (r.weights_gib / cap) * 100)
              const kvPct = Math.min(100 - wPct, (r.kv_gib / cap) * 100)
              const rtPct = Math.min(100 - wPct - kvPct, (r.runtime_gib / cap) * 100)

              return (
                <div key={r.node_id} className="cs-mem-card" style={{ borderTop: `3px solid ${r.color}` }}>
                  <div className="cs-node-row">
                    <div style={{ fontWeight: 700, fontSize: '0.86rem', display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span>Rank {r.rank} · {r.name}</span>
                      <span className="cs-badge">TP{r.tp_rank}/PP{r.pp_rank}</span>
                    </div>
                    <span className={`cs-badge ${r.fits ? 'good' : 'bad'}`}>
                      {r.fits ? `+${r.headroom_gib} GiB free` : `${r.headroom_gib} GiB OVER`}
                    </span>
                  </div>

                  <div className="cs-stacked-bar" title={`Weights ${r.weights_gib} GiB + KV ${r.kv_gib} GiB + Runtime ${r.runtime_gib} GiB / ${r.ram_total_gib} GiB`}>
                    <div className="cs-stack-weights" style={{ width: `${wPct}%`, background: r.color }} />
                    <div className="cs-stack-kv" style={{ width: `${kvPct}%` }} />
                    <div className="cs-stack-rt" style={{ width: `${rtPct}%` }} />
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', fontFamily: 'var(--mono)', color: 'var(--ink2)' }}>
                    <span>Weights {r.weights_gib}G · KV {r.kv_gib}G</span>
                    <span>{r.total_needed_gib} / {r.load_limit_gib} GiB cap</span>
                  </div>

                  <div className="cs-shard-meta">
                    <div><span>Layers</span>L{r.layer_range[0]}–{r.layer_range[1] - 1} ({r.layer_count})</div>
                    <div><span>Attn Heads</span>H{r.head_range[0]}–{r.head_range[1] - 1} ({r.head_count})</div>
                    <div><span>MoE Ch</span>{r.moe_half_range[0]}–{r.moe_half_range[1] - 1}</div>
                  </div>
                </div>
              )
            })}
          </div>

          {showJson && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 10 }}>
              {plan.ranks.map((r) => (
                <div key={r.node_id}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 700, marginBottom: 4, color: r.color }}>
                    rank{r.rank}.json ({r.name})
                  </div>
                  <pre className="cs-code">{JSON.stringify(r.rank_json, null, 2)}</pre>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </section>
  )
}
