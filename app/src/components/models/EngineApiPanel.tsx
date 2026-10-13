import { useEffect, useState } from 'react'
import { api } from '../../lib/ipc'
import { useEngine } from '../../lib/engine'
import { engineBaseUrl, engineCurl } from '../../lib/engine-workspace'
import { Button } from '../ui'

export function EngineApiPanel({ onModels }: { onModels: () => void }) {
  const { snapshot } = useEngine()
  const endpoint = engineBaseUrl(snapshot?.status)
  const [copied, setCopied] = useState<'url' | 'request' | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => { setCopied(null); setError('') }, [endpoint, snapshot?.status.model])
  const copy = async (kind: 'url' | 'request') => {
    setBusy(true); setError(''); setCopied(null)
    try {
      let text = endpoint
      if (kind === 'request') {
        // Connection credentials exist only in this explicit clipboard action.
        const connection = await api.engineConnection()
        if (!connection) throw new Error('No live connection')
        text = engineCurl(connection)
      }
      if (!text || !(await api.writeClipboard(text)).ok) throw new Error('Clipboard unavailable')
      setCopied(kind)
    } catch {
      setError(kind === 'request' ? 'A live desktop connection is required; load a model and try copying again.' : 'The URL could not be copied; try again after loading a model.')
    } finally { setBusy(false) }
  }

  return (
    <section data-testid="local-api-panel" className="mt-6">
      <h2 className="text-[16px] font-medium text-ink">Use your model in other tools</h2>
      <p className="mt-1 max-w-2xl text-[13px] leading-relaxed text-ink-2">The same loaded model serves TensorFold and OpenAI-compatible clients on this Mac.</p>
      {!endpoint ? (
        <div className="mt-4 rounded-xl border border-line bg-card p-5"><p className="text-[14px] font-medium text-ink">Load a model to start the API</p><p className="mt-1 text-[13px] text-ink-2">The endpoint and its connection key become available when the engine is ready.</p><Button className="mt-4" size="sm" icon="cpu" onClick={onModels}>Choose a model</Button></div>
      ) : (
        <div className="mt-4 overflow-hidden rounded-xl border border-line bg-card">
          <div className="flex flex-wrap items-center justify-between gap-3 p-4"><div className="min-w-0"><p className="text-[12px] text-ink-2">Base URL</p><code data-testid="local-api-url" className="mt-1 block break-all text-[14px] text-ink">{endpoint}</code></div><Button size="sm" icon={copied === 'url' ? 'check' : 'copy'} disabled={busy} onClick={() => void copy('url')}>{copied === 'url' ? 'Copied' : 'Copy URL'}</Button></div>
          <div className="border-t border-line px-4 py-4"><h3 className="text-[14px] font-medium text-ink">Try a streaming request</h3><p className="mt-1 text-[13px] text-ink-2">Copy a ready-to-run curl command with this model and its current connection key.</p><Button className="mt-3" size="sm" icon={copied === 'request' ? 'check' : 'terminal'} disabled={busy} onClick={() => void copy('request')}>{copied === 'request' ? 'Request copied' : 'Copy authenticated request'}</Button></div>
          <dl className="grid gap-3 border-t border-line bg-card-2 px-4 py-4 text-[13px]"><div><dt className="font-medium text-ink">Model ID</dt><dd className="mt-1 break-all font-mono text-ink-2">{snapshot?.status.model}</dd></div><div><dt className="font-medium text-ink">Connection lifetime</dt><dd className="mt-1 text-ink-2">The port and key change when the engine restarts; copy fresh details after reloading a model.</dd></div><div><dt className="font-medium text-ink">Access</dt><dd className="mt-1 text-ink-2">Loopback only, with a key required for requests.</dd></div></dl>
        </div>
      )}
      {error && <p role="alert" className="mt-3 text-[13px] text-danger">{error}</p>}
      <div role="status" className="sr-only">{copied === 'request' ? 'Authenticated request copied to clipboard.' : copied === 'url' ? 'Base URL copied to clipboard.' : ''}</div>
    </section>
  )
}
