import clsx from 'clsx'
import { useEngine, ENGINE_PROVIDER_ID } from '../../lib/engine'
import { useLearning, learningBusy } from '../../lib/learning'
import { useStore } from '../../lib/store'
import { Icon } from '../../lib/icons'
import { Menu } from '../ui'
import { CortheonToggle } from '../models/CortheonToggle'
import { LearningControls } from '../models/LearningControls'

export function LocalEngineControls({ model }: { model: string | null }) {
  const { snapshot } = useEngine()
  const { snapshot: learning } = useLearning()
  const threads = useStore((s) => s.threads)
  const statuses = useStore((s) => s.status)
  const activeRuns = threads.filter((thread) => thread.providerId === ENGINE_PROVIDER_ID && !thread.orchestratorHostId && thread.workspace?.kind !== 'ssh' && ['thinking', 'streaming', 'tool', 'awaiting'].includes(statuses[thread.id] ?? (thread.status === 'running' ? 'thinking' : 'idle'))).length
  const active = snapshot?.status.cortheon?.active || learning?.active && snapshot?.status.state === 'ready' && snapshot.status.model === learning.model
  return <Menu width={400} placement="top" align="end" trigger={(open) => <button type="button" aria-label="AI settings" aria-expanded={open} title="Cortheon and Living Weights settings" className={clsx('inline-flex h-8 shrink-0 items-center gap-1.5 rounded-full px-2 text-[13px] transition-colors focus-visible:outline-2 focus-visible:outline-accent', open ? 'bg-ink-5 text-ink' : active ? 'text-accent hover:bg-ink-5' : 'text-ink-2 hover:bg-ink-5')}><Icon name="sliders" size={15} /><span className="hidden @sm:inline">AI settings</span>{active && <span className="h-1.5 w-1.5 rounded-full bg-accent" />}</button>}>
    {(close) => <div><div className="px-3 py-3"><h2 className="text-[14px] font-semibold text-ink">AI settings</h2><p className="mt-1 text-[12px] text-ink-2">Engine-wide settings apply to every local chat and API client.</p></div><CortheonToggle status={snapshot?.status} activeRuns={activeRuns} blocked={learningBusy(learning) || snapshot?.status.state === 'starting'} /><div className="border-t border-line"><LearningControls compact model={model} activeRuns={activeRuns} onGraph={close} /></div></div>}
  </Menu>
}
