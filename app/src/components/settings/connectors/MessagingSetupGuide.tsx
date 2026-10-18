import { OriginalDiagnostic } from '../../OriginalDiagnostic'
import { useAuthoredText } from '../../../lib/preferences/useAuthoredText'
import { useState } from 'react'
import { api } from '../../../lib/ipc'
import { Button } from '../../ui'

export function MessagingSetupGuide({ steps, links }: { steps: string[]; links: Array<{ label: string; url: string }> }) {
  const tx = useAuthoredText()
  const [error, setError] = useState('')
  return <div className="rounded-xl border border-line bg-card-2 p-3.5 text-[13px] leading-relaxed text-ink-2">
    <ol className="list-decimal space-y-1 pl-4">{steps.map((step) => <li key={step}>{step}</li>)}</ol>
    <div className="mt-3 flex flex-wrap gap-2">{links.map((link) => <Button key={link.url} size="sm" onClick={async () => { setError(''); try { await api.openExternal(link.url) } catch { setError(tx("Could not open the setup page; check your browser and try again.")) } }}>{link.label}</Button>)}</div>
    {error && <p role="alert" className="mt-2 text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
  </div>
}
