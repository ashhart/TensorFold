import { useStore } from '../../lib/store'
import { Icon } from '../../lib/icons'

export function FullLayoutSwitch() {
  const layout = useStore((state) => state.fullWorkspaceLayout)
  const choose = useStore((state) => state.setFullWorkspaceLayout)
  const simple = layout === 'chat'
  return <button type="button" data-full-layout-switch aria-label={simple ? 'Complete workspace' : 'Use Chat layout'} title={simple ? 'Return to the complete workspace; your panel positions are kept.' : 'Use a simpler chat layout inside Full mode.'} onClick={() => {
    choose(simple ? 'workspace' : 'chat')
    requestAnimationFrame(() => document.querySelector<HTMLButtonElement>('[data-full-layout-switch]')?.focus())
  }} className="no-drag inline-flex h-8 shrink-0 items-center gap-1.5 rounded-lg border border-line bg-card px-2.5 text-[12px] text-ink transition-colors hover:border-line-strong hover:bg-card-2 focus-visible:outline-2 focus-visible:outline-accent">
    <Icon name={simple ? 'grid' : 'message'} size={14} />
    {simple ? <><span className="hidden sm:inline">Complete workspace</span><span className="sm:hidden">Workspace</span></> : <span>Chat layout</span>}
  </button>
}
