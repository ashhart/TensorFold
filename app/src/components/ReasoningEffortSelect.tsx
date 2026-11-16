import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import { effortDisplayLabel, providerEffortOptions } from '../lib/provider-model-metadata'
import type { Provider, ReasoningEffortSetting } from '../lib/types'

export function effortForModel(value: ReasoningEffortSetting | undefined, provider?: Provider, model?: string): ReasoningEffortSetting | undefined {
  if (value === undefined) return undefined
  const options = providerEffortOptions(model?.trim() ? provider : undefined, model)
  return options.some((option) => option.value === value) ? value : options.find((option) => option.isDefault)?.value ?? options[0]?.value
}

export function ReasoningEffortSelect({ provider, model, value, defaultEffort, disabled = false, onChange }: {
  provider?: Provider
  model?: string
  value?: ReasoningEffortSetting
  disabled?: boolean
  defaultEffort?: ReasoningEffortSetting
  onChange(value: ReasoningEffortSetting | undefined): void
}) {
  const tx = useAuthoredText()
  const options = providerEffortOptions(model?.trim() ? provider : undefined, model)
  const unavailable = value !== undefined && !options.some((option) => option.value === value)
  return <label className="block space-y-1.5 text-[13px] text-ink-2">
    <span className="block font-medium">{tx("Effort")}</span>
    <select disabled={disabled} value={value ?? ''} onChange={(event) => onChange(event.target.value ? event.target.value as ReasoningEffortSetting : undefined)} className="h-10 w-full min-w-0 rounded-lg border border-line bg-card-2 px-2 text-[13px] text-ink">
      <option value="">{tx("Default")}{defaultEffort ? ` (${tx(effortDisplayLabel(provider, model, defaultEffort))})` : ''}</option>
      {unavailable && <option value={value} disabled>{tx("Unavailable: {value0}", { value0: value })}</option>}
      {options.map((option) => <option key={option.value} value={option.value}>{tx(option.label)}</option>)}
    </select>
  </label>
}
