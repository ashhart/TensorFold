import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { providerEffortOptions } from '../../lib/provider-model-metadata'
import type { Provider, ReasoningEffortSetting } from '../../lib/types'

export function effortForBotModel(value: ReasoningEffortSetting | undefined, provider?: Provider, model?: string): ReasoningEffortSetting | undefined {
  if (value === undefined) return undefined
  const options = providerEffortOptions(provider, model)
  return options.some((option) => option.value === value) ? value : options.find((option) => option.isDefault)?.value ?? options[0]?.value
}

export function BotReasoningEffort({ provider, model, value, onChange }: {
  provider?: Provider
  model?: string
  value?: ReasoningEffortSetting
  onChange(value: ReasoningEffortSetting | undefined): void
}) {
  const tx = useAuthoredText()
  const options = providerEffortOptions(provider, model)
  const unavailable = value !== undefined && !options.some((option) => option.value === value)
  return <label className="block space-y-1.5 text-[13px] text-ink-2">
    <span className="block font-medium">{tx("Reasoning effort")}</span>
    <select value={value ?? ''} onChange={(event) => onChange(event.target.value ? event.target.value as ReasoningEffortSetting : undefined)} className="h-10 w-full min-w-0 rounded-lg border border-line bg-card-2 px-2 text-[13px] text-ink">
      <option value="">{tx("Use app setting")}</option>
      {unavailable && <option value={value} disabled>{tx("Unavailable: {value0}", { value0: value })}</option>}
      {options.map((option) => <option key={option.value} value={option.value}>{tx(option.label)}</option>)}
    </select>
    <span className="block text-[12px] text-ink-2">{tx("Saved for this bot. Changes apply to new chats; existing chats keep their captured setting.")}</span>
  </label>
}
