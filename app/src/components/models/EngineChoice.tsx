import { useFlowText } from '../../lib/preferences/useFlowText'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import clsx from 'clsx'
import { Icon } from '../../lib/icons'
import { formatBytes, useEngine } from '../../lib/engine'
import { Button } from '../ui'
import { engineDotTone } from './EngineStatus'
import { BrandMark } from '../BrandMark'
import { repoDisplayName } from '../../lib/turn-stats-line'
import { engineReadiness } from '../../lib/engine-readiness'

// Onboarding's first choice: the engine that ships inside the app.

export function EngineChoice({ onOpenModels, onUse, minimal = false }: { onOpenModels: () => void; onUse: (repo: string) => void; minimal?: boolean }) {
  const { t } = useFlowText()
  const tx = useAuthoredText()
  const { snapshot, error } = useEngine()
  const status = snapshot?.status
  const installed = snapshot?.installed ?? []
  const usable = installed.filter((m) => m.fitsMemory !== false)
  const ready = !error && engineReadiness(snapshot).ready
  const loading = !!status?.available && status.state === 'starting'
  const budget = status?.memoryLimitBytes
  return (
    <div data-testid="onboarding-provider-engine" className="rounded-xl border border-line-strong bg-card p-5 shadow-card">
      <div className="flex items-start gap-3">
        <BrandMark size={40} tile />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[15px] font-medium text-ink">{t("engine.title")}</span>
            <span className="rounded-md bg-surface px-2 py-0.5 text-[12px] text-ink-3">{t("engine.recommended")}</span>
          </div>
          <p className="mt-1 text-[13px] leading-snug text-ink-3">
            {minimal ? t("engine.minimalHelp") : t("engine.fullHelp")}
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[13px] text-ink-3">
            <span role="status" className="flex items-center gap-1.5"><span className={clsx('h-1.5 w-1.5 rounded-full', error ? 'bg-danger' : engineDotTone(status))} />{error ? tx('Engine status unavailable') : !status ? t('engine.checking') : !status.available ? t('engine.missing') : ready ? tx('Model running locally') : loading ? tx('Loading model…') : status.state === 'failed' ? tx('Model needs attention') : status.version ? t('engine.versionIdle', { version: status.version }) : t('engine.idle')}</span>
            {typeof budget === 'number' && Number.isFinite(budget) && budget > 0 && <span>{t("engine.memory", { size: formatBytes(budget) })}</span>}
            <span>{installed.length ? t(installed.length === 1 ? 'engine.model' : 'engine.models', { count: installed.length }) : t("engine.none")}</span>
          </div>
          {error && <div className="mt-2 text-[13px] text-danger">{t("common.diagnostic")}: {error}</div>}
        </div>
      </div>
      {usable.length > 0 && (
        <div className="mt-4 grid gap-2">
          {usable.slice(0, 3).map((model) => {
            const name = model.title || snapshot?.suggestions.find((item) => item.repo === model.repo)?.title || repoDisplayName(model.repo)
            const running = ready && status?.model === model.repo
            const floor = model.memoryFloorBytes
            return <div key={model.repo} className="flex flex-wrap items-center gap-3 rounded-xl border border-line bg-card-2 px-4 py-3">
              <div className="min-w-0 flex-1 basis-44">
                <p title={model.repo} className="break-words text-[14px] font-medium text-ink">{name}</p>
                <p className="mt-1 text-[12px] text-ink-2">{running ? tx('Running on this Mac') : tx('Downloaded')}{typeof floor === 'number' && Number.isFinite(floor) && floor > 0 ? ` · ${formatBytes(floor)} ${tx('memory floor')}` : ''}</p>
              </div>
              <Button size="sm" variant="primary" iconRight="arrowRight" disabled={!status?.available || loading || !!error} aria-label={`${tx('Choose this model')}: ${name}`} onClick={() => onUse(model.repo)}>{tx('Choose this model')}</Button>
            </div>
          })}
          <p className="px-1 text-[12px] leading-relaxed text-ink-2">{minimal ? tx('Choose a model now, then start your local server from Models.') : tx('Choose a model, then prepare your first message. The model loads when you send it.')}</p>
        </div>
      )}
      <div className="mt-3 flex flex-wrap gap-2">
        {usable.length ? (
          <Button size="sm" variant="ghost" icon="arrowDown" onClick={onOpenModels}>{t("engine.another")}</Button>
        ) : (
          <button type="button" data-testid="onboarding-choose-model" onClick={onOpenModels} className="no-drag inline-flex h-9 select-none items-center gap-1.5 rounded-lg bg-brand px-3 text-[13px] font-medium text-white shadow-card transition-[opacity,transform] duration-100 hover:opacity-90 active:scale-[0.985] focus-visible:outline-2 focus-visible:outline-accent">
            <Icon name="arrowDown" size={15} />{t("first.chooseAction")}</button>
        )}
      </div>
    </div>
  )
}
