import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useState } from 'react'
import { api } from '../../lib/ipc'
import { useEngine } from '../../lib/engine'
import { Button, Input, Toggle } from '../ui'
import { Section } from './shared'
import { CONTROL_VECTOR_ALPHA_MAX } from '../../../shared/control-vector'
import type { EngineControlVectorSnapshot } from '../../lib/types'

// The control vector the local engine applies: a file on this Mac, a strength, and
// whether it is in force. It is opt-in: nothing changes until the operator turns it
// on, and the engine says at startup what it is holding.

const fileName = (file: string | null): string => {
  if (!file) return ''
  const parts = file.split('/')
  return parts[parts.length - 1] || file
}

export function ControlVectorSettings() {
  const tx = useAuthoredText()
  const { snapshot: engine } = useEngine()
  const [vector, setVector] = useState<EngineControlVectorSnapshot | null>(null)
  const [alpha, setAlpha] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    let mounted = true
    api.engineControlVector().then((next) => {
      if (!mounted) return
      setVector(next); setAlpha(String(next.alpha))
    }).catch((e) => { if (mounted) setError(String((e as Error)?.message ?? e)) })
    return () => { mounted = false }
  }, [])

  const run = async (action: () => Promise<EngineControlVectorSnapshot>) => {
    setError(''); setBusy(true)
    try {
      const next = await action()
      setVector(next); setAlpha(String(next.alpha))
    } catch (e) { setError(String((e as Error)?.message ?? e)) } finally { setBusy(false) }
  }

  const commitAlpha = (value: string) => {
    const parsed = Number(value)
    if (!Number.isFinite(parsed) || parsed <= 0 || parsed > CONTROL_VECTOR_ALPHA_MAX) {
      setError(tx('The strength must be above 0 and at most {value0}.', { value0: String(CONTROL_VECTOR_ALPHA_MAX) }))
      return
    }
    if (vector && parsed === vector.alpha) return
    void run(() => api.engineSetControlVector({ alpha: parsed }))
  }

  const live = engine?.status.controlVector?.active === true
  const supported = vector?.supported === true
  const name = fileName(vector?.file ?? null)

  return (
    <Section title={tx('Control vector')} desc={tx('A control vector steers the local model: it changes what the model says. It is off until you turn it on, and the engine names what is in force when it starts.')}>
      {!supported && (
        <p className="mb-3 rounded-lg border border-line bg-card px-3 py-2.5 text-[13px] text-ink-3">
          {tx("This build's TensorFold engine does not offer control vectors yet, so the engine in use cannot take one.")}
        </p>
      )}
      <div className="rounded-xl border border-line bg-card shadow-card">
        <div className="flex items-start justify-between gap-4 px-3.5 py-3">
          <div className="min-w-0">
            <div className="text-[13px] font-medium text-ink">{tx('Apply a control vector')}</div>
            <div className="mt-0.5 text-[12px] leading-snug text-ink-3">
              {name ? tx('File: {value0}', { value0: name }) : tx('No file selected')}
              {live && <> · {tx('In force in the loaded model.')}</>}
            </div>
          </div>
          <Toggle
            label={tx('Apply a control vector')}
            checked={vector?.enabled === true}
            disabled={!supported || busy || !vector?.file}
            onChange={(enabled) => void run(() => api.engineSetControlVector({ enabled }))}
          />
        </div>
        <div className="space-y-3 border-t border-line px-3.5 py-3">
          <div className="flex flex-wrap items-end gap-2">
            <div className="min-w-0 flex-1">
              <label className="mb-1 block text-[12px] text-ink-3" htmlFor="control-vector-file">{tx('File')}</label>
              <div className="truncate rounded-lg border border-line bg-panel px-2.5 py-1.5 font-mono text-[12px] text-ink-2">
                {vector?.file ?? tx('No file selected')}
              </div>
            </div>
            <Button variant="default" size="sm" icon="folder" disabled={busy} onClick={() => void run(() => api.engineChooseControlVector())}>{tx('Choose…')}</Button>
            {vector?.file && <Button variant="ghost" size="sm" icon="trash" disabled={busy} onClick={() => void run(() => api.engineClearControlVector())}>{tx('Remove')}</Button>}
          </div>
          <div className="flex flex-wrap items-end gap-2">
            <div className="w-32">
              <label className="mb-1 block text-[12px] text-ink-3" htmlFor="control-vector-alpha">{tx('Strength')}</label>
              <Input
                id="control-vector-alpha"
                type="number"
                min={0}
                max={CONTROL_VECTOR_ALPHA_MAX}
                step={0.1}
                value={alpha}
                disabled={busy}
                onChange={(e) => setAlpha(e.target.value)}
                onBlur={(e) => commitAlpha(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') commitAlpha((e.target as HTMLInputElement).value) }}
              />
            </div>
            <p className="flex-1 text-[12px] leading-snug text-ink-3">
              {tx('One is the strength the file was derived at. A strength past a few can leave the model without an answer.')}
            </p>
          </div>
        </div>
      </div>
      {vector?.enabled && (
        <p className="mt-2 text-[12px] leading-relaxed text-ink-3">
          {tx('With a control vector in force the model replies differently. It is a steering direction, not a safety setting: it changes what the model is willing to say.')}
        </p>
      )}
      {error && <p className="mt-2 text-[13px] text-red-500">{error}</p>}
    </Section>
  )
}
