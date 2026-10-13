import clsx from 'clsx'
import { Icon } from '../../lib/icons'
import { engineStateLabel, formatBytes, useEngine } from '../../lib/engine'
import { Button } from '../ui'
import { engineDotTone } from './EngineStatus'
import { BrandMark } from '../BrandMark'

// Onboarding's first choice: the engine that ships inside the app.

export function EngineChoice({ onOpenModels, onUse }: { onOpenModels: () => void; onUse: (repo: string) => void }) {
  const { snapshot, error } = useEngine()
  const status = snapshot?.status
  const installed = snapshot?.installed ?? []
  const usable = installed.filter((m) => m.fitsMemory !== false)
  return (
    <div data-testid="onboarding-provider-engine" className="rounded-xl border border-line-strong bg-card p-5 shadow-card">
      <div className="flex items-start gap-3">
        <BrandMark size={40} tile />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[15px] font-medium text-ink">TensorFold engine (built in)</span>
            <span className="rounded-md bg-surface px-2 py-0.5 text-[12px] text-ink-3">Recommended</span>
          </div>
          <p className="mt-1 text-[13px] leading-snug text-ink-3">
            Runs models on this Mac with the engine inside the app. No account or server to set up: download a model, then chat or code.
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[13px] text-ink-3">
            <span className="flex items-center gap-1.5"><span className={clsx('h-1.5 w-1.5 rounded-full', engineDotTone(status))} />{engineStateLabel(status)}</span>
            {status && <span>models may use up to {formatBytes(status.memoryLimitBytes)} of memory</span>}
            <span>{installed.length ? `${installed.length} model${installed.length === 1 ? '' : 's'} on this Mac` : 'no models on this Mac yet'}</span>
          </div>
          {error && <div className="mt-2 text-[13px] text-danger">{error}</div>}
        </div>
      </div>
      {usable.length > 0 && (
        <div className="mt-3 grid gap-1.5">
          {usable.slice(0, 3).map((model) => (
            <div key={model.repo} className="flex items-center gap-2 rounded-lg border border-line bg-card-2 px-3 py-2">
              <span className="min-w-0 flex-1 truncate text-[13px] text-ink-2">{model.repo}</span>
              <Button size="xs" variant="primary" icon="check" onClick={() => onUse(model.repo)}>Use</Button>
            </div>
          ))}
        </div>
      )}
      <div className="mt-3 flex flex-wrap gap-2">
        {usable.length ? (
          <Button size="sm" variant="default" icon="arrowDown" onClick={onOpenModels} disabled={status?.available === false}>Get another model</Button>
        ) : (
          <button type="button" data-testid="onboarding-choose-model" onClick={onOpenModels} disabled={status?.available === false} className="no-drag inline-flex h-8 select-none items-center gap-1.5 rounded-lg bg-brand px-3 text-[13px] font-medium text-white shadow-[0_1px_2px_rgb(0_0_0/0.12)] transition-[opacity,transform] duration-100 hover:opacity-90 active:scale-[0.985] disabled:pointer-events-none disabled:opacity-40">
            <Icon name="arrowDown" size={15} />Choose a model
          </button>
        )}
      </div>
    </div>
  )
}
