import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import clsx from 'clsx'
import type { Provider, Settings } from '../../lib/types'
import { formatContextTokens, providerEffortOptions, type ContextUsageSummary } from '../../lib/provider-model-metadata'

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex min-h-8 items-center gap-3 px-2.5">
      <span className="w-[72px] shrink-0 text-[13px] text-ink-3">{label}</span>
      <div className="flex min-w-0 flex-1 flex-wrap items-center justify-end gap-1">{children}</div>
    </div>
  )
}

function Option({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button type="button" onClick={onClick} aria-pressed={active} className={clsx('rounded-md px-2 py-1 text-[13px] leading-4 transition-colors', active ? 'bg-ink-5 font-medium text-ink' : 'text-ink-3 hover:bg-ink-5 hover:text-ink')}>
      {children}
    </button>
  )
}

// Run settings that used to sit beside the composer: reasoning effort, Codex speed, context use.
export function ModelMenuSettings({ provider, model, reasoningEffort, onChangeEffort, serviceTier, onChangeServiceTier, context }: {
  provider: Provider | null | undefined
  model: string | null
  reasoningEffort: Settings['reasoningEffort']
  onChangeEffort: (effort: Settings['reasoningEffort']) => void
  /** Present only for Codex models with a speed tier. */
  serviceTier?: Settings['serviceTier'] | null
  onChangeServiceTier?: (tier: NonNullable<Settings['serviceTier']>) => void
  context: ContextUsageSummary
}) {
  const tx = useAuthoredText()
  const effortOptions = providerEffortOptions(provider, model)
  const pct = Math.max(0, Math.min(100, Math.round(context.percent * 100)))
  const ring = `conic-gradient(rgb(var(--ink) / 0.62) ${pct * 3.6}deg, rgb(var(--ink) / 0.12) 0deg)`
  return (
    <div className="mb-1 space-y-0.5 border-b border-line pb-1.5">
      {effortOptions.length > 0 && (
        <Row label={tx("Reasoning")}>
          {effortOptions.map((option) => (
            <Option key={option.value} active={reasoningEffort === option.value} onClick={() => onChangeEffort(option.value)}>{option.label}</Option>
          ))}
        </Row>
      )}
      {onChangeServiceTier && (
        <Row label={tx("Speed")}>
          {(['fast', 'flex'] as const).map((tier) => (
            <Option key={tier} active={(serviceTier ?? 'fast') === tier} onClick={() => onChangeServiceTier(tier)}>{tier === 'fast' ? tx("Fast") : tx("Flex")}</Option>
          ))}
        </Row>
      )}
      <Row label={tx("Context")}>
        <span title={context.workingLimitTokens && context.workingLimitTokens !== context.windowTokens ? tx("Working set {value0} tokens", { value0: formatContextTokens(context.workingLimitTokens) }) : undefined} className="flex items-center gap-2 text-[13px] tabular-nums text-ink-2">
          {pct}{tx("% used · ")}{formatContextTokens(context.usedTokens)} {tx("of ")}{formatContextTokens(context.windowTokens)}
          <span className="flex h-3.5 w-3.5 items-center justify-center rounded-full" style={{ background: ring }}>
            <span className="h-2 w-2 rounded-full bg-card" />
          </span>
        </span>
      </Row>
    </div>
  )
}
