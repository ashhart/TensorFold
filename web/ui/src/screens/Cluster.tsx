import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import type { ClusterPlan, ClusterState } from '../clusterTypes'
import type { Ctx } from '../ctx'
import ClusterBenchmark from './cluster/ClusterBenchmark'
import ClusterSetupModal from './cluster/ClusterSetupModal'
import ClusterShardMap from './cluster/ClusterShardMap'
import ClusterTopology from './cluster/ClusterTopology'

function Sparkline({ values, color }: { values: number[]; color: string }) {
  if (values.length < 2) return null
  const max = Math.max(1, ...values)
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * 220
    const y = 26 - (v / max) * 22
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
  return (
    <svg className="cs-spark" viewBox="0 0 220 28" preserveAspectRatio="none" aria-hidden="true">
      <polyline fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" points={pts} />
    </svg>
  )
}

const fmtGiB = (bytes: number) => `${(bytes / 1024 ** 3).toFixed(bytes >= 1024 ** 4 ? 0 : 1)} GB`

export default function Cluster({ toast, refreshEngine }: Ctx) {
  const [state, setState] = useState<ClusterState | null>(null)
  const [plan, setPlan] = useState<ClusterPlan | null>(null)
  const [selectedIds, setSelectedIds] = useState<string[]>(['master'])
  const [focusedNodeId, setFocusedNodeId] = useState<string | null>(null)
  const [weights, setWeights] = useState<Record<string, number>>({})
  const [modelPath, setModelPath] = useState<string>('')
  const [mode, setMode] = useState<'auto' | 'tp' | 'pp' | 'hybrid'>('auto')
  const [context, setContext] = useState<number>(8192)
  const [parallel, setParallel] = useState<number>(1)
  const [setupOpen, setSetupOpen] = useState(false)
  const [deploying, setDeploying] = useState(false)
  const [showLogs, setShowLogs] = useState(false)
  const initSelectionDone = useRef(false)

  const refreshState = useCallback(async () => {
    try {
      const next = await api.clusterState()
      setState(next)
      if (!initSelectionDone.current && next.nodes.length > 0) {
        initSelectionDone.current = true
        const onlineIds = next.nodes.filter((n) => n.online && n.enabled).map((n) => n.id)
        if (onlineIds.length > 0) setSelectedIds(onlineIds)
      }
      setModelPath((prev) => prev || next.models[0]?.path || '')
    } catch {
      // Transient poll errors handled silently
    }
  }, [])

  // Live 1.5s telemetry loop
  useEffect(() => {
    let alive = true
    let timer: ReturnType<typeof setTimeout>
    const tick = async () => {
      await refreshState()
      if (alive) timer = setTimeout(tick, 1500)
    }
    tick()
    return () => { alive = false; clearTimeout(timer) }
  }, [refreshState])

  // Recompute heterogeneous shard plan whenever selection, weights, model, mode, context or streams change
  useEffect(() => {
    if (!state || selectedIds.length === 0) return
    const handle = setTimeout(() => {
      api.clusterPlan({
        node_ids: selectedIds,
        model_path: modelPath || undefined,
        mode,
        context,
        parallel,
        custom_weights: weights,
      })
        .then(setPlan)
        .catch(() => {})
    }, 80)
    return () => clearTimeout(handle)
  }, [state?.nodes.length, selectedIds, modelPath, mode, context, parallel, weights])

  const onlineIds = useMemo(() => (state?.nodes ?? []).filter((n) => n.online).map((n) => n.id), [state])

  const toggleNode = (id: string) => {
    if (id === 'master') return
    setSelectedIds((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id])
  }

  const applyPreset = (preset: '2tp' | '4tp' | '4hybrid' | '4pp') => {
    if (preset === '2tp') {
      setSelectedIds(onlineIds.slice(0, 2))
      setMode('tp')
      setWeights({})
    } else if (preset === '4tp') {
      setSelectedIds(onlineIds.slice(0, 4))
      setMode('tp')
      setWeights({})
    } else if (preset === '4hybrid') {
      setSelectedIds(onlineIds.slice(0, 4))
      setMode('hybrid')
      setWeights({})
    } else if (preset === '4pp') {
      setSelectedIds(onlineIds.slice(0, 4))
      setMode('pp')
      setWeights({})
    }
  }

  const handleDeploy = async () => {
    setDeploying(true)
    try {
      await api.clusterDeploy({
        node_ids: selectedIds,
        model_path: modelPath || undefined,
        mode,
        context,
        parallel,
        custom_weights: weights,
        prompt_cache_gib: 0,
      })
      toast('Cluster ranks launched across TB5 mesh!')
      await refreshState()
      refreshEngine()
    } catch (e) {
      toast(`Cluster launch failed: ${e}`, 'bad')
    } finally {
      setDeploying(false)
    }
  }

  const handleStop = async () => {
    try {
      await api.clusterStop()
      toast('Cluster stopped across all nodes.')
      await refreshState()
      refreshEngine()
    } catch (e) {
      toast(`Stop failed: ${e}`, 'bad')
    }
  }

  const handleProbeLinks = async () => {
    try {
      await api.clusterProbeLinks()
      toast('Probing all active Thunderbolt 5 RDMA cables…')
      await refreshState()
    } catch (e) {
      toast(`RDMA probe failed: ${e}`, 'bad')
    }
  }

  const handleSyncModel = async () => {
    if (!modelPath) return
    try {
      await api.clusterSyncModel({ model_path: modelPath })
      toast('Syncing model shards to slaves over direct TB5 links…')
      await refreshState()
    } catch (e) {
      toast(`Sync failed: ${e}`, 'bad')
    }
  }

  const handleRemoveNode = async (id: string) => {
    try {
      await api.clusterDeleteNode(id)
      setSelectedIds((prev) => prev.filter((x) => x !== id))
      toast('Removed slave node from cluster.')
      await refreshState()
    } catch (e) {
      toast(`Could not remove node: ${e}`, 'bad')
    }
  }

  const handleRunBenchmark = async (opts: { prompt_tokens: number[]; decode_tokens: number; draft: boolean }) => {
    try {
      await api.clusterBenchmark(opts)
      toast('Cold prefill & decode benchmark started!')
      await refreshState()
    } catch (e) {
      toast(`Benchmark failed: ${e}`, 'bad')
    }
  }

  const summary = state?.summary
  const nodes = state?.nodes ?? []
  const links = state?.links ?? []
  const dep = state?.deployment
  const masterRankState = nodes.find((n) => n.id === 'master')?.rank?.state ?? 'idle'
  const isDeployed = Boolean(dep || masterRankState === 'loading' || masterRankState === 'running')
  const isServing = masterRankState === 'running'
  const isInferencing = Boolean((summary?.decode_tps ?? 0) > 0.1 || (summary?.prefill_tps ?? 0) > 0.1 || (summary?.running_requests ?? 0) > 0)
  const focusedNode = nodes.find((n) => n.id === focusedNodeId) ?? nodes[0] ?? null
  const slavesOnline = nodes.filter((n) => n.role === 'slave' && n.online).length

  return (
    <div className="cs-page">
      {/* Top Hero Command Strip + 1-2-3 Stepper */}
      <header className="cs-hero">
        <div className="cs-hero-top">
          <div className="cs-title-group">
            <div className="cs-logo-mark" aria-hidden="true">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.1">
                <polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5 12 2" />
                <line x1="12" y1="22" x2="12" y2="15.5" />
                <polyline points="22 8.5 12 15.5 2 8.5" />
                <polyline points="2 15.5 12 8.5 22 15.5" />
                <line x1="12" y1="2" x2="12" y2="8.5" />
              </svg>
            </div>
            <div>
              <h1 className="cs-title">
                TensorFold Cluster Studio
                <span className={`cs-live-pill ${isServing ? '' : masterRankState === 'loading' ? 'loading' : 'idle'}`}>
                  <span className="cs-pulse" />
                  {isServing
                    ? `Serving · ${dep?.world ?? 1}N ${dep?.mode?.toUpperCase() ?? ''}`
                    : masterRankState === 'loading'
                      ? 'Loading Shards…'
                      : `${summary?.nodes_online ?? 1} Node${(summary?.nodes_online ?? 1) === 1 ? '' : 's'} Ready`}
                </span>
              </h1>
              <p className="cs-subtitle">
                1-2-3 Master/Slave SSH Orchestration · Full-Mesh Thunderbolt 5 RDMA · Heterogeneous TP / PP / Hybrid Engine
              </p>
            </div>
          </div>

          <div className="cs-hero-actions">
            <button className="cs-btn sm" onClick={() => setShowLogs((v) => !v)}>
              {showLogs ? 'Hide Rank Logs' : 'Rank Console'}
            </button>
            <button className="cs-btn" onClick={() => setSetupOpen(true)}>
              + Setup SSH / Slaves
            </button>
            {isDeployed && (
              <button className="cs-btn danger" onClick={handleStop}>
                Stop Cluster
              </button>
            )}
          </div>
        </div>

        {/* 1-2-3 Guided Workflow Ribbon */}
        <div className="cs-stepper">
          <div className="cs-step done" onClick={() => setFocusedNodeId('master')}>
            <div className="cs-step-num">1</div>
            <div>
              <div className="cs-step-title">Master Node Online</div>
              <div className="cs-step-sub">
                {nodes[0] ? `${nodes[0].name} · ${nodes[0].hardware.chip} · ${fmtGiB(nodes[0].hardware.ram_total_bytes)}` : 'Initializing…'}
              </div>
            </div>
          </div>

          <div className={`cs-step ${slavesOnline > 0 ? 'done' : 'active'}`} onClick={() => setSetupOpen(true)}>
            <div className="cs-step-num">2</div>
            <div>
              <div className="cs-step-title">
                {slavesOnline > 0 ? `${slavesOnline} Slave${slavesOnline > 1 ? 's' : ''} Linked via TB5` : 'Link Slaves via SSH'}
              </div>
              <div className="cs-step-sub">
                {slavesOnline > 0
                  ? `${summary?.links_active ?? 0} active 80G RDMA cables · Click to add more`
                  : <>1-click SSH bootstrap or run <code>tensorfold-web --slave</code></>}
              </div>
            </div>
          </div>

          <div className={`cs-step ${isServing ? 'done' : slavesOnline > 0 ? 'active' : ''}`} onClick={() => { if (!isDeployed && plan?.all_fit) handleDeploy() }}>
            <div className="cs-step-num">3</div>
            <div>
              <div className="cs-step-title">
                {isServing ? `Serving ${dep?.model_name ?? 'Model'}` : 'Split & Launch Model'}
              </div>
              <div className="cs-step-sub">
                {plan ? `${plan.model.name} · TP=${plan.tp} × PP=${plan.pp} (${plan.effective_bandwidth_gbs} GB/s)` : 'Configure sharding below'}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* 4 Live Compute & Throughput Hero KPIs */}
      <section className="cs-kpis">
        <div className="cs-kpi">
          <div className="cs-kpi-head">
            <span>Decode Throughput</span>
            {summary?.mtp_accept_rate != null && (
              <span className="cs-badge good">MTP {summary.mtp_accept_rate}%</span>
            )}
          </div>
          <div className="cs-kpi-val">
            {summary ? summary.decode_tps.toFixed(1) : '0.0'}
            <span className="cs-kpi-unit">tok/s</span>
          </div>
          <Sparkline values={(state?.history ?? []).map((h) => h.decode_tps)} color="#3d6bff" />
          <div className="cs-kpi-foot">
            <span>{summary?.generation_tokens_total?.toLocaleString() ?? 0} gen tok</span>
            <span>{summary?.running_requests ?? 0} active req</span>
          </div>
        </div>

        <div className="cs-kpi">
          <div className="cs-kpi-head">
            <span>Prefill Throughput</span>
            {summary?.ttft_p50_ms != null && (
              <span className="cs-badge master">TTFT {Math.round(summary.ttft_p50_ms)} ms</span>
            )}
          </div>
          <div className="cs-kpi-val">
            {summary ? summary.prefill_tps.toFixed(0) : '0'}
            <span className="cs-kpi-unit">tok/s</span>
          </div>
          <Sparkline values={(state?.history ?? []).map((h) => h.prefill_tps)} color="#8b4dff" />
          <div className="cs-kpi-foot">
            <span>{summary?.prompt_tokens_total?.toLocaleString() ?? 0} prompt tok</span>
            <span>{plan ? `${plan.all_reduce_per_token} syncs/tok` : '—'}</span>
          </div>
        </div>

        <div className="cs-kpi">
          <div className="cs-kpi-head">
            <span>Cluster Unified Memory</span>
            <span className="cs-badge">{summary?.total_bandwidth_gbs ?? 0} GB/s</span>
          </div>
          <div className="cs-kpi-val">
            {summary ? fmtGiB(summary.total_ram_bytes) : '—'}
            <span className="cs-kpi-unit">total</span>
          </div>
          <div className="cs-bar" style={{ height: 8, margin: '8px 0 4px' }}>
            <div
              className="cs-bar-fill"
              style={{
                width: `${summary && summary.total_ram_bytes > 0 ? Math.min(100, (summary.used_ram_bytes / summary.total_ram_bytes) * 100) : 0}%`,
                background: 'var(--brand-gradient)',
              }}
            />
          </div>
          <div className="cs-kpi-foot">
            <span>Used {summary ? fmtGiB(summary.used_ram_bytes) : '—'}</span>
            <span>GPU Alloc {summary ? fmtGiB(summary.total_gpu_alloc_bytes) : '—'}</span>
          </div>
        </div>

        <div className="cs-kpi">
          <div className="cs-kpi-head">
            <span>Compute &amp; TB5 Fabric</span>
            <span className="cs-badge good">{summary?.total_gpu_cores ?? 0} GPU Cores</span>
          </div>
          <div className="cs-kpi-val">
            {summary ? `${Math.round(summary.avg_gpu_util_pct)}%` : '0%'}
            <span className="cs-kpi-unit">GPU · {summary?.total_tflops_fp16 ?? 0} TF</span>
          </div>
          <Sparkline values={(state?.history ?? []).map((h) => h.gpu_util)} color="#22d3ee" />
          <div className="cs-kpi-foot">
            <span>{summary?.links_active ?? 0} × 80G RDMA</span>
            <span>{summary ? `${summary.total_rdma_mbps.toFixed(1)} MB/s live` : '0 MB/s'}</span>
          </div>
        </div>
      </section>

      {/* Interactive Full-Mesh TB5 RDMA Topology */}
      <ClusterTopology
        nodes={nodes}
        links={links}
        plan={plan}
        selectedIds={selectedIds}
        focusedNodeId={focusedNodeId}
        weights={weights}
        probeRunning={Boolean(state?.probe_running)}
        isInferencing={isInferencing}
        onToggleNode={toggleNode}
        onFocusNode={(id) => setFocusedNodeId((cur) => cur === id ? null : id)}
        onWeightChange={(id, w) => setWeights((prev) => ({ ...prev, [id]: w }))}
        onRemoveNode={handleRemoveNode}
        onProbeLinks={handleProbeLinks}
        onOpenSetup={() => setSetupOpen(true)}
        onApplyPreset={applyPreset}
      />

      {/* Heterogeneous Sharding & 79-Layer Architecture Map */}
      <ClusterShardMap
        models={state?.models ?? []}
        modelPath={modelPath}
        mode={mode}
        context={context}
        parallel={parallel}
        plan={plan}
        syncJobs={state?.sync_jobs ?? {}}
        deploying={deploying}
        isDeployed={isDeployed}
        onModelChange={setModelPath}
        onModeChange={setMode}
        onContextChange={setContext}
        onParallelChange={setParallel}
        onSyncModel={handleSyncModel}
        onDeploy={handleDeploy}
        onStop={handleStop}
      />

      {/* Live Rank Console Drawer (Optional) */}
      {showLogs && focusedNode && (
        <section className="cs-panel">
          <div className="cs-panel-head">
            <h2 className="cs-panel-title">Rank Console · {focusedNode.name}</h2>
            <div className="cs-presets">
              {nodes.map((n) => (
                <button
                  key={n.id}
                  className={`cs-chip ${focusedNode.id === n.id ? 'active' : ''}`}
                  onClick={() => setFocusedNodeId(n.id)}
                >
                  {n.name}
                </button>
              ))}
            </div>
          </div>
          <div className="cs-log-box">
            {(focusedNode.rank?.log_tail?.length ?? 0) > 0
              ? focusedNode.rank.log_tail!.join('\n')
              : 'No rank output yet. Launch the cluster above to stream live layer-loading and RDMA logs.'}
          </div>
        </section>
      )}

      {/* 2-Node vs 4-Node Cold Prefill & Decode Benchmark Arena */}
      <ClusterBenchmark
        benchmarks={state?.benchmarks ?? []}
        activeJob={state?.benchmark_job ?? null}
        canRun={isServing}
        onRun={handleRunBenchmark}
      />

      {/* 1-2-3 SSH & Slave Setup Modal */}
      <ClusterSetupModal
        open={setupOpen}
        joinCommand={state?.slave_join_command ?? ''}
        clusterToken={state?.cluster_token ?? ''}
        onClose={() => setSetupOpen(false)}
        onAdded={refreshState}
        toast={toast}
      />
    </div>
  )
}
