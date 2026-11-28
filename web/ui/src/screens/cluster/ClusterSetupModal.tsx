import { useEffect, useState } from 'react'
import { api } from '../../api'
import type { ClusterDiscoverCandidate } from '../../clusterTypes'

type Props = {
  open: boolean
  joinCommand: string
  clusterToken: string
  onClose: () => void
  onAdded: () => void
  toast: (text: string, kind?: 'ok' | 'bad') => void
}

export default function ClusterSetupModal({ open, joinCommand, clusterToken, onClose, onAdded, toast }: Props) {
  const [tab, setTab] = useState<'discover' | 'ssh' | 'manual'>('discover')
  const [discovering, setDiscovering] = useState(false)
  const [candidates, setCandidates] = useState<ClusterDiscoverCandidate[]>([])
  const [bootstrapping, setBootstrapping] = useState<Record<string, boolean>>({})

  // SSH form
  const [host, setHost] = useState('')
  const [user, setUser] = useState('')
  const [keyPath, setKeyPath] = useState('~/.ssh/id_ed25519')
  const [slavePort, setSlavePort] = useState(8091)
  const [nodeName, setNodeName] = useState('')
  const [testing, setTesting] = useState(false)
  const [sshResult, setSshResult] = useState<Record<string, any> | null>(null)

  // Manual slave join form
  const [manualUrl, setManualUrl] = useState('')
  const [manualName, setManualName] = useState('')
  const [manualToken, setManualToken] = useState(clusterToken)

  const runDiscover = () => {
    setDiscovering(true)
    api.clusterDiscover()
      .then((r) => setCandidates(r.candidates))
      .catch((e) => toast(`Discovery failed: ${e}`, 'bad'))
      .finally(() => setDiscovering(false))
  }

  useEffect(() => {
    if (open) {
      setManualToken(clusterToken)
      runDiscover()
    }
  }, [open, clusterToken])

  if (!open) return null

  const bootstrapCandidate = async (c: ClusterDiscoverCandidate) => {
    setBootstrapping((b) => ({ ...b, [c.ssh_host]: true }))
    try {
      await api.clusterSshBootstrap({
        host: c.ssh_host,
        user: c.ssh_user || undefined,
        key_path: c.ssh_key || undefined,
        name: c.hostname || c.name,
        slave_host: c.host,
        slave_port: 8091,
      })
      toast(`${c.hostname || c.name} bootstrapped in Slave mode!`)
      onAdded()
      runDiscover()
    } catch (e) {
      toast(`Bootstrap failed for ${c.name}: ${e}`, 'bad')
    } finally {
      setBootstrapping((b) => ({ ...b, [c.ssh_host]: false }))
    }
  }

  const bootstrapAll = async () => {
    const ready = candidates.filter((c) => c.ssh_reachable && !c.already_added)
    for (const c of ready) {
      await bootstrapCandidate(c)
    }
  }

  const testSsh = async () => {
    if (!host.trim()) return
    setTesting(true)
    setSshResult(null)
    try {
      const r = await api.clusterSshTest({
        host: host.trim(),
        user: user.trim() || undefined,
        key_path: keyPath.trim() || undefined,
      })
      setSshResult(r)
      if (r.ok && r.hostname && !nodeName) setNodeName(r.hostname)
    } catch (e) {
      setSshResult({ ok: false, error: String(e) })
    } finally {
      setTesting(false)
    }
  }

  const submitSshBootstrap = async () => {
    if (!host.trim()) return
    setBootstrapping((b) => ({ ...b, manual_ssh: true }))
    try {
      await api.clusterSshBootstrap({
        host: host.trim(),
        user: user.trim() || undefined,
        key_path: keyPath.trim() || undefined,
        name: nodeName.trim() || undefined,
        slave_port: slavePort,
      })
      toast(`Node ${nodeName || host} joined as Slave!`)
      onAdded()
      onClose()
    } catch (e) {
      toast(`SSH bootstrap failed: ${e}`, 'bad')
    } finally {
      setBootstrapping((b) => ({ ...b, manual_ssh: false }))
    }
  }

  const submitManualSlave = async () => {
    if (!manualUrl.trim()) return
    try {
      await api.clusterRegisterNode({
        slave_url: manualUrl.trim(),
        name: manualName.trim() || undefined,
        token: manualToken.trim() || undefined,
      })
      toast('Slave node connected!')
      onAdded()
      onClose()
    } catch (e) {
      toast(`Could not connect slave: ${e}`, 'bad')
    }
  }

  const readyToAdd = candidates.filter((c) => c.ssh_reachable && !c.already_added)

  return (
    <div className="cs-modal-backdrop" onClick={onClose}>
      <div className="cs-modal" role="dialog" aria-modal="true" aria-label="Add Slave Nodes" onClick={(e) => e.stopPropagation()}>
        <div className="cs-panel-head">
          <div>
            <h2 className="cs-panel-title">Add Slave Nodes · Easy 1-2-3 Setup</h2>
            <p className="cs-panel-sub">Auto-discover TB5 peers, bootstrap over SSH in one click, or attach a remote <code>--slave</code> daemon</p>
          </div>
          <button className="cs-btn sm" onClick={onClose}>Close</button>
        </div>

        <div className="cs-seg">
          <button className={tab === 'discover' ? 'active' : ''} onClick={() => setTab('discover')}>
            1. Auto-Discover TB5 / SSH
          </button>
          <button className={tab === 'ssh' ? 'active' : ''} onClick={() => setTab('ssh')}>
            2. Bootstrap via SSH
          </button>
          <button className={tab === 'manual' ? 'active' : ''} onClick={() => setTab('manual')}>
            3. Run Slave Command
          </button>
        </div>

        {tab === 'discover' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--ink2)' }}>
                Scanning direct Thunderbolt 5 <code>/30</code> subnets &amp; <code>~/.ssh/config</code>…
              </span>
              <div style={{ display: 'flex', gap: 8 }}>
                <button className="cs-btn sm" disabled={discovering} onClick={runDiscover}>
                  {discovering ? 'Scanning…' : 'Rescan'}
                </button>
                {readyToAdd.length > 0 && (
                  <button className="cs-btn sm primary" onClick={bootstrapAll}>
                    Bootstrap All ({readyToAdd.length})
                  </button>
                )}
              </div>
            </div>

            {candidates.length === 0 && !discovering && (
              <div style={{ padding: 18, textAlign: 'center', color: 'var(--ink3)', fontSize: '0.83rem' }}>
                No peers found on local TB5 links or `~/.ssh/config`. Switch to <strong>Bootstrap via SSH</strong> to add any Mac or GPU server by IP.
              </div>
            )}

            {candidates.map((c) => (
              <div key={c.ssh_host} className="cs-cand-row">
                <div>
                  <div style={{ fontWeight: 700, fontSize: '0.88rem', display: 'flex', alignItems: 'center', gap: 7 }}>
                    <span>{c.hostname || c.name}</span>
                    <span style={{ fontFamily: 'var(--mono)', fontSize: '0.75rem', color: 'var(--ink3)' }}>{c.host}</span>
                    {c.ssh_reachable ? <span className="cs-badge good">SSH OK</span> : <span className="cs-badge bad">No SSH</span>}
                    {c.slave_running && <span className="cs-badge master">Slave Active</span>}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--ink3)', marginTop: 2 }}>
                    {c.source}{c.chip ? ` · ${c.chip}` : ''}{c.ram_gb ? ` · ${c.ram_gb} GB` : ''}
                  </div>
                </div>
                {c.already_added ? (
                  <span className="cs-badge good">Linked ✓</span>
                ) : (
                  <button
                    className="cs-btn sm primary"
                    disabled={!c.ssh_reachable || Boolean(bootstrapping[c.ssh_host])}
                    onClick={() => bootstrapCandidate(c)}
                  >
                    {bootstrapping[c.ssh_host] ? 'Bootstrapping…' : 'Bootstrap Slave'}
                  </button>
                )}
              </div>
            ))}
          </div>
        )}

        {tab === 'ssh' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 1fr', gap: 10 }}>
              <div className="cs-field">
                <label>SSH Host / IP / Alias</label>
                <input className="cs-input" placeholder="node-2 or 10.10.12.2" value={host} onChange={(e) => setHost(e.target.value)} />
              </div>
              <div className="cs-field">
                <label>Username (optional if in SSH config)</label>
                <input className="cs-input" placeholder="username" value={user} onChange={(e) => setUser(e.target.value)} />
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 0.9fr 0.7fr', gap: 10 }}>
              <div className="cs-field">
                <label>SSH Private Key Path</label>
                <input className="cs-input" value={keyPath} onChange={(e) => setKeyPath(e.target.value)} />
              </div>
              <div className="cs-field">
                <label>Display Name</label>
                <input className="cs-input" placeholder="node-2" value={nodeName} onChange={(e) => setNodeName(e.target.value)} />
              </div>
              <div className="cs-field">
                <label>Slave Port</label>
                <input className="cs-input" type="number" value={slavePort} onChange={(e) => setSlavePort(Number(e.target.value))} />
              </div>
            </div>

            {sshResult && (
              <div className="cs-cand-row" style={{ borderColor: sshResult.ok ? 'var(--positive)' : 'var(--danger)' }}>
                {sshResult.ok ? (
                  <div style={{ fontSize: '0.8rem' }}>
                    <strong style={{ color: 'var(--positive)' }}>✓ Connected to {sshResult.hostname} ({sshResult.latency_ms} ms)</strong>
                    <div style={{ color: 'var(--ink2)', marginTop: 2, fontFamily: 'var(--mono)', fontSize: '0.75rem' }}>
                      {sshResult.chip} · {sshResult.ram_gb} GB · {sshResult.gpu_cores ?? '?'} GPU cores · RDMA: {(sshResult.rdma_devices || []).join(', ') || 'none'}
                    </div>
                  </div>
                ) : (
                  <div style={{ fontSize: '0.8rem', color: 'var(--danger)' }}>✕ {sshResult.error}</div>
                )}
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <button className="cs-btn" disabled={testing || !host.trim()} onClick={testSsh}>
                {testing ? 'Testing SSH…' : 'Test SSH Connection'}
              </button>
              <button className="cs-btn primary" disabled={Boolean(bootstrapping.manual_ssh) || !host.trim()} onClick={submitSshBootstrap}>
                {bootstrapping.manual_ssh ? 'Deploying & Starting Slave…' : 'Deploy & Start Slave Mode'}
              </button>
            </div>
          </div>
        )}

        {tab === 'manual' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ fontSize: '0.82rem', color: 'var(--ink2)' }}>
              Run this command on a worker (bind <code>127.0.0.1</code> over an SSH tunnel, or pass <code>--tls-cert</code> and <code>--tls-key</code> for remote HTTPS), then register its URL and token below:
            </div>
            <div className="cs-code">{joinCommand}</div>
            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button
                className="cs-btn sm"
                onClick={() => { navigator.clipboard?.writeText(joinCommand); toast('Copied slave command!') }}
              >
                Copy Command
              </button>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 1fr 1fr', gap: 10 }}>
              <div className="cs-field">
                <label>Slave URL (HTTPS or loopback)</label>
                <input className="cs-input" placeholder="https://10.10.12.2:8091" value={manualUrl} onChange={(e) => setManualUrl(e.target.value)} />
              </div>
              <div className="cs-field">
                <label>Worker Token</label>
                <input className="cs-input" type="password" placeholder="Per-node secret token" value={manualToken} onChange={(e) => setManualToken(e.target.value)} />
              </div>
              <div className="cs-field">
                <label>Node Name</label>
                <input className="cs-input" placeholder="node-2" value={manualName} onChange={(e) => setManualName(e.target.value)} />
              </div>
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <button className="cs-btn primary" disabled={!manualUrl.trim()} onClick={submitManualSlave}>
                Connect Slave Node
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
