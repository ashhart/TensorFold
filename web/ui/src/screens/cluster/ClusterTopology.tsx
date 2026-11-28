import { useMemo } from 'react'
import type { ClusterLink, ClusterNode, ClusterPlan } from '../../clusterTypes'

type Props = {
  nodes: ClusterNode[]
  links: ClusterLink[]
  plan: ClusterPlan | null
  selectedIds: string[]
  focusedNodeId: string | null
  weights: Record<string, number>
  probeRunning: boolean
  isInferencing: boolean
  onToggleNode: (id: string) => void
  onFocusNode: (id: string) => void
  onWeightChange: (id: string, w: number) => void
  onRemoveNode: (id: string) => void
  onProbeLinks: () => void
  onOpenSetup: () => void
  onApplyPreset: (preset: '2tp' | '4tp' | '4hybrid' | '4pp') => void
}

function layoutNodes(nodes: ClusterNode[]): Record<string, { x: number; y: number }> {
  const n = nodes.length
  const out: Record<string, { x: number; y: number }> = {}
  if (n === 1) {
    out[nodes[0].id] = { x: 380, y: 210 }
  } else if (n === 2) {
    out[nodes[0].id] = { x: 185, y: 210 }
    out[nodes[1].id] = { x: 575, y: 210 }
  } else if (n === 3) {
    out[nodes[0].id] = { x: 380, y: 92 }
    out[nodes[1].id] = { x: 175, y: 318 }
    out[nodes[2].id] = { x: 585, y: 318 }
  } else if (n === 4) {
    // Full-mesh 4-node diamond/square with unobstructed diagonals!
    out[nodes[0].id] = { x: 185, y: 102 }
    out[nodes[1].id] = { x: 575, y: 102 }
    out[nodes[2].id] = { x: 185, y: 318 }
    out[nodes[3].id] = { x: 575, y: 318 }
  } else {
    const cx = 380
    const cy = 210
    const rx = 250
    const ry = 135
    nodes.forEach((node, i) => {
      const a = -Math.PI / 2 + (2 * Math.PI * i) / n
      out[node.id] = { x: cx + Math.cos(a) * rx, y: cy + Math.sin(a) * ry }
    })
  }
  return out
}

const fmtGiB = (b: number) => `${(b / 1024 ** 3).toFixed(0)} GB`

export default function ClusterTopology({
  nodes,
  links,
  plan,
  selectedIds,
  focusedNodeId,
  weights,
  probeRunning,
  isInferencing,
  onToggleNode,
  onFocusNode,
  onWeightChange,
  onRemoveNode,
  onProbeLinks,
  onOpenSetup,
  onApplyPreset,
}: Props) {
  const coords = useMemo(() => layoutNodes(nodes), [nodes])
  const rankMap = useMemo(() => {
    const m: Record<string, ClusterPlan['ranks'][number]> = {}
    for (const r of plan?.ranks ?? []) m[r.node_id] = r
    return m
  }, [plan])

  const onlineCount = nodes.filter((n) => n.online).length

  return (
    <section className="cs-panel">
      <div className="cs-panel-head">
        <div>
          <h2 className="cs-panel-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="5" cy="5" r="3" /><circle cx="19" cy="5" r="3" />
              <circle cx="5" cy="19" r="3" /><circle cx="19" cy="19" r="3" />
              <path d="M8 5h8M8 19h8M5 8v8M19 8v8M7.2 7.2l9.6 9.6M16.8 7.2l-9.6 9.6" />
            </svg>
            Thunderbolt 5 RDMA Full-Mesh Topology
          </h2>
          <p className="cs-panel-sub">
            {onlineCount}/{nodes.length} nodes online · {links.filter((l) => l.state === 'active').length} active 80 Gbps RDMA cables · Click any node to inspect or toggle participation
          </p>
        </div>

        <div className="cs-presets">
          {onlineCount >= 2 && (
            <button
              className={`cs-chip ${selectedIds.length === 2 && plan?.tp === 2 ? 'active' : ''}`}
              onClick={() => onApplyPreset('2tp')}
            >
              2-Node TP2
            </button>
          )}
          {onlineCount >= 4 && (
            <>
              <button
                className={`cs-chip ${selectedIds.length === 4 && plan?.tp === 4 ? 'active' : ''}`}
                onClick={() => onApplyPreset('4tp')}
              >
                4-Node TP4
              </button>
              <button
                className={`cs-chip ${selectedIds.length === 4 && plan?.tp === 2 && plan?.pp === 2 ? 'active' : ''}`}
                onClick={() => onApplyPreset('4hybrid')}
              >
                4-Node TP2×PP2
              </button>
              <button
                className={`cs-chip ${selectedIds.length === 4 && plan?.pp === 4 ? 'active' : ''}`}
                onClick={() => onApplyPreset('4pp')}
              >
                4-Node PP4
              </button>
            </>
          )}
          <button className="cs-btn sm" disabled={probeRunning || links.length === 0} onClick={onProbeLinks}>
            {probeRunning ? 'Probing RDMA…' : 'Probe RDMA Mesh'}
          </button>
          <button className="cs-btn sm primary" onClick={onOpenSetup}>
            + Add Slave
          </button>
        </div>
      </div>

      <div className="cs-topo-wrap">
        {/* Interactive Spatial Mesh SVG */}
        <div className="cs-canvas-box">
          <svg className="cs-topo-svg" viewBox="0 0 760 420" role="img" aria-label="Cluster TB5 RDMA Topology">
            <defs>
              <linearGradient id="cs-brand-stroke" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#3d6bff" />
                <stop offset="50%" stopColor="#8b4dff" />
                <stop offset="100%" stopColor="#22d3ee" />
              </linearGradient>
              <filter id="cs-glow" x="-30%" y="-30%" width="160%" height="160%">
                <feGaussianBlur stdDeviation="3.5" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {/* Physical TB5 RDMA Links */}
            {links.map((link) => {
              const a = coords[link.a_id]
              const b = coords[link.b_id]
              if (!a || !b) return null
              const bothSelected = selectedIds.includes(link.a_id) && selectedIds.includes(link.b_id)
              const focused = !focusedNodeId || link.a_id === focusedNodeId || link.b_id === focusedNodeId
              const hot = link.traffic_mbps > 5 || (isInferencing && bothSelected)
              const mx = (a.x + b.x) / 2
              const my = (a.y + b.y) / 2
              // Offset crossing diagonals cleanly along their own vectors so pills never touch
              const isDiag = Math.abs(a.x - b.x) > 200 && Math.abs(a.y - b.y) > 120
              const lx = isDiag ? a.x + (b.x - a.x) * (a.y < b.y ? 0.30 : 0.70) : mx
              const ly = isDiag ? a.y + (b.y - a.y) * (a.y < b.y ? 0.30 : 0.70) : my
              const label = link.bandwidth_gbs
                ? `${link.bandwidth_gbs.toFixed(1)} GB/s${link.latency_us ? ` · ${link.latency_us.toFixed(0)}µs` : ''}`
                : link.traffic_mbps > 0.5
                  ? `${link.traffic_mbps.toFixed(0)} MB/s`
                  : `${link.speed_gbps}G RDMA`

              return (
                <g key={link.id} opacity={focused ? (bothSelected ? 1 : 0.35) : 0.16}>
                  <line
                    x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                    stroke={link.state === 'active' ? 'url(#cs-brand-stroke)' : '#f0525c'}
                    strokeWidth={bothSelected ? 3 : 1.6}
                    strokeOpacity={bothSelected ? 0.55 : 0.3}
                  />
                  {link.state === 'active' && bothSelected && (
                    <line
                      x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                      stroke={hot ? '#22d3ee' : '#8ea0ff'}
                      strokeWidth={hot ? 2.8 : 1.8}
                      className={`cs-link-flow ${hot ? 'fast' : ''}`}
                      filter={hot ? 'url(#cs-glow)' : undefined}
                    />
                  )}
                  <g transform={`translate(${lx}, ${ly})`}>
                    <rect x="-48" y="-11" width="96" height="22" rx="11" fill="var(--card)" stroke="var(--line-strong)" strokeWidth="1" />
                    <text textAnchor="middle" y="4" fontSize="9.5" fontFamily="var(--mono)" fontWeight="600" fill={hot ? '#22d3ee' : 'var(--ink2)'}>
                      {label}
                    </text>
                  </g>
                </g>
              )
            })}

            {/* Node Cards inside Canvas */}
            {nodes.map((node) => {
              const p = coords[node.id]
              if (!p) return null
              const included = selectedIds.includes(node.id)
              const rp = rankMap[node.id]
              const isFocused = focusedNodeId === node.id
              const ramPct = node.memory.pressure_pct || 0
              const gpuPct = node.gpu.util_pct || 0
              const circ = 2 * Math.PI * 16
              const rankState = node.rank?.state ?? 'idle'
              const isLoading = rankState === 'loading'
              const loadPct = isLoading && node.rank?.layer_total
                ? Math.round(((node.rank.layer_loaded || 0) / Math.max(1, node.rank.layer_total)) * 100)
                : 0

              return (
                <g
                  key={node.id}
                  transform={`translate(${p.x - 108}, ${p.y - 52})`}
                  className={`cs-node-svg-card ${included ? '' : 'excluded'}`}
                  onClick={() => onFocusNode(node.id)}
                >
                  {/* Selection / Master Halo */}
                  {(isFocused || node.role === 'master') && (
                    <rect
                      x="-3" y="-3" width="222" height="110" rx="17"
                      fill="none"
                      stroke={node.color}
                      strokeWidth={isFocused ? 2.2 : 1.2}
                      strokeOpacity={isFocused ? 0.85 : 0.35}
                    />
                  )}
                  <rect
                    x="0" y="0" width="216" height="104" rx="14"
                    fill="var(--card)"
                    stroke={included ? node.color : 'var(--line-strong)'}
                    strokeWidth={included ? 1.6 : 1}
                  />
                  {/* Left color accent bar */}
                  <rect x="0" y="14" width="4" height="76" rx="2" fill={node.online ? node.color : '#666'} />

                  {/* Header: Name + Rank/Role */}
                  <text x="15" y="23" fontSize="13" fontWeight="700" fill="var(--ink)">
                    {node.name}
                  </text>
                  <rect
                    x="130" y="10" width="76" height="18" rx="6"
                    fill={node.role === 'master' ? 'rgba(61,107,255,0.18)' : 'rgba(139,77,255,0.16)'}
                  />
                  <text
                    x="168" y="22.5" textAnchor="middle" fontSize="9" fontWeight="700" fontFamily="var(--mono)"
                    fill={node.role === 'master' ? '#5a78ff' : '#a78bfa'}
                  >
                    {rp ? `RANK ${rp.rank} · ${node.role === 'master' ? 'MST' : 'SLV'}` : node.role.toUpperCase()}
                  </text>

                  {/* Chip + RAM subtitle */}
                  <text x="15" y="39" fontSize="10" fontFamily="var(--mono)" fill="var(--ink3)">
                    {node.hardware.chip} · {fmtGiB(node.hardware.ram_total_bytes)}
                  </text>

                  {/* Dual Live Arc Rings: GPU Compute + Unified RAM */}
                  <g transform="translate(36, 72)">
                    <circle r="16" fill="none" stroke="var(--line-strong)" strokeWidth="3.5" />
                    <circle
                      r="16" fill="none" stroke={node.color} strokeWidth="3.5"
                      strokeDasharray={circ}
                      strokeDashoffset={circ * (1 - Math.min(100, gpuPct) / 100)}
                      strokeLinecap="round"
                      transform="rotate(-90)"
                    />
                    <text textAnchor="middle" y="3.5" fontSize="9" fontWeight="700" fontFamily="var(--mono)" fill="var(--ink)">
                      {Math.round(gpuPct)}%
                    </text>
                    <text textAnchor="middle" y="26" fontSize="7.5" fontFamily="var(--mono)" fill="var(--ink3)">GPU</text>
                  </g>

                  <g transform="translate(82, 72)">
                    <circle r="16" fill="none" stroke="var(--line-strong)" strokeWidth="3.5" />
                    <circle
                      r="16" fill="none" stroke="#8b4dff" strokeWidth="3.5"
                      strokeDasharray={circ}
                      strokeDashoffset={circ * (1 - Math.min(100, ramPct) / 100)}
                      strokeLinecap="round"
                      transform="rotate(-90)"
                    />
                    <text textAnchor="middle" y="3.5" fontSize="9" fontWeight="700" fontFamily="var(--mono)" fill="var(--ink)">
                      {Math.round(ramPct)}%
                    </text>
                    <text textAnchor="middle" y="26" fontSize="7.5" fontFamily="var(--mono)" fill="var(--ink3)">RAM</text>
                  </g>

                  {/* Right Status & Shard Readout */}
                  <text x="112" y="64" fontSize="9.5" fontWeight="700" fontFamily="var(--mono)"
                    fill={rankState === 'running' ? '#3ccf91' : isLoading ? '#f2b234' : rankState === 'failed' ? '#f0525c' : 'var(--ink2)'}
                  >
                    {rankState === 'running' ? '● SERVING' : isLoading ? `◔ LOAD ${loadPct}%` : rankState === 'failed' ? '✕ FAILED' : node.online ? '○ READY' : '✕ OFFLINE'}
                  </text>
                  <text x="112" y="79" fontSize="9" fontFamily="var(--mono)" fill="var(--ink3)">
                    {rp ? `L${rp.layer_range[0]}–${rp.layer_range[1] - 1} · ${rp.share_pct}%` : 'Excluded'}
                  </text>
                  <text x="112" y="92" fontSize="8.5" fontFamily="var(--mono)" fill="var(--ink3)">
                    {rp ? `${rp.weights_gib} GiB · ${rp.bandwidth_gbs} GB/s` : `${node.rdma.length} TB5 ports`}
                  </text>
                </g>
              )
            })}
          </svg>
        </div>

        {/* Interactive Node Controls & Heterogeneous Weight Sliders */}
        <div className="cs-node-list">
          {nodes.map((node) => {
            const included = selectedIds.includes(node.id)
            const rp = rankMap[node.id]
            const w = weights[node.id] ?? 1.0
            const usedGiB = (node.memory.used_bytes / 1024 ** 3).toFixed(1)
            const totGiB = (node.hardware.ram_total_bytes / 1024 ** 3).toFixed(0)

            return (
              <div
                key={node.id}
                className={`cs-node-card ${focusedNodeId === node.id ? 'selected' : ''} ${included ? '' : 'excluded'}`}
                style={{ '--node-accent': node.color } as React.CSSProperties}
                onClick={() => onFocusNode(node.id)}
              >
                <div className="cs-node-row">
                  <div className="cs-node-ident">
                    <span className="cs-node-dot" style={{ background: node.online ? node.color : '#666' }} />
                    <div>
                      <div className="cs-node-name">
                        {node.name}
                        <span className={`cs-badge ${node.role === 'master' ? 'master' : ''}`}>
                          {rp ? `R${rp.rank}` : node.role}
                        </span>
                        {node.rank?.state === 'running' && <span className="cs-badge good">LIVE</span>}
                        {node.rank?.state === 'loading' && <span className="cs-badge warn">L{node.rank.layer_loaded}/{node.rank.layer_total}</span>}
                        {!node.online && <span className="cs-badge bad">OFFLINE</span>}
                      </div>
                      <div className="cs-node-specs">
                        {node.hardware.chip} · {node.hardware.gpu_cores}c GPU · {node.hardware.memory_bandwidth_gbs} GB/s
                      </div>
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    {node.role === 'slave' && (
                      <>
                        <label style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: '0.73rem', cursor: 'pointer' }} onClick={(e) => e.stopPropagation()}>
                          <input type="checkbox" checked={included} disabled={!node.online} onChange={() => onToggleNode(node.id)} />
                          Use
                        </label>
                        <button
                          className="cs-btn sm"
                          title="Remove slave"
                          onClick={(e) => { e.stopPropagation(); onRemoveNode(node.id) }}
                        >
                          ✕
                        </button>
                      </>
                    )}
                  </div>
                </div>

                <div className="cs-mini-meters">
                  <div className="cs-meter">
                    <div className="cs-meter-lbl">
                      <span>GPU Compute</span>
                      <span>{Math.round(node.gpu.util_pct)}%</span>
                    </div>
                    <div className="cs-bar">
                      <div className="cs-bar-fill" style={{ width: `${Math.min(100, node.gpu.util_pct)}%`, background: node.color }} />
                    </div>
                  </div>
                  <div className="cs-meter">
                    <div className="cs-meter-lbl">
                      <span>Unified RAM</span>
                      <span>{usedGiB}/{totGiB}G</span>
                    </div>
                    <div className="cs-bar">
                      <div className="cs-bar-fill" style={{ width: `${Math.min(100, node.memory.pressure_pct)}%`, background: '#8b4dff' }} />
                    </div>
                  </div>
                </div>

                {included && (
                  <div className="cs-weight-row" onClick={(e) => e.stopPropagation()}>
                    <span>Shard Weight</span>
                    <input
                      type="range"
                      min="0.5"
                      max="2.0"
                      step="0.1"
                      value={w}
                      onChange={(e) => onWeightChange(node.id, parseFloat(e.target.value))}
                    />
                    <span style={{ fontFamily: 'var(--mono)', minWidth: 34, textAlign: 'right' }}>{w.toFixed(1)}×</span>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
