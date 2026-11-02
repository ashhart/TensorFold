import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { Icon } from '../../lib/icons'
import { Button, Input } from '../ui'
import type { Provider } from '../../lib/types'
import { useState } from 'react'

export function ModelChecklistModal({ provider, onClose, onSave }: { provider: Provider; onClose: () => void; onSave: (models: string[]) => void }) {
  const tx = useAuthoredText()
  const discovered = provider.discoveredModels?.length ? provider.discoveredModels : provider.models

  const initialCatalogue = Array.from(new Set([...(discovered ?? []), ...provider.models]))
  const [catalogue, setCatalogue] = useState<string[]>(initialCatalogue)
  const [selected, setSelected] = useState<Set<string>>(new Set(provider.models))
  const [customModel, setCustomModel] = useState('')

  const toggle = (model: string) => {
    setSelected((current) => {
      const next = new Set(current)
      if (next.has(model)) next.delete(model)
      else next.add(model)
      return next
    })
  }
  const addCustom = () => {
    const model = customModel.trim()
    if (!model) return
    setCatalogue((current) => (current.includes(model) ? current : [...current, model]))
    setSelected((current) => new Set(current).add(model))
    setCustomModel('')
  }
  // Preserve provider model ordering for already-selected ids; append the rest.
  const save = () => {
    const order = [...provider.models, ...catalogue]
    const ordered = order.filter((model, index) => selected.has(model) && order.indexOf(model) === index)
    onSave(ordered)
  }

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 p-5 backdrop-blur-sm" onClick={onClose}>
      <div className="flex max-h-[min(560px,calc(100vh-80px))] w-[min(480px,calc(100vw-40px))] flex-col overflow-hidden rounded-2xl border border-line bg-bg shadow-[0_24px_90px_rgb(0_0_0/0.45)]" onClick={(e) => e.stopPropagation()}>
        <div className="flex shrink-0 items-center justify-between gap-3 border-b border-line px-4 py-3">
          <div className="min-w-0">
            <div className="truncate text-[14px] font-semibold text-ink">{tx("Edit models")}</div>
            <div className="mt-0.5 truncate text-[13px] text-ink-3">{provider.name} · {selected.size} {tx("selected")}</div>
          </div>
          <button onClick={onClose} aria-label={tx("Close")} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink">
            <Icon name="x" size={15} />
          </button>
        </div>
        <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-4 py-3">
          {catalogue.length === 0 ? (
            <div className="rounded-lg border border-dashed border-line px-3 py-6 text-center text-[13px] text-ink-3">
              {tx("No models discovered yet. Use Connect to refresh, or add one below.")}</div>
          ) : (
            <div className="space-y-1.5">
              {catalogue.map((model) => (
                <label key={model} className="flex cursor-pointer items-center gap-2.5 rounded-lg border border-line bg-card px-3 py-2 text-[13px] text-ink-2 transition-colors hover:border-line-strong">
                  <input type="checkbox" checked={selected.has(model)} onChange={() => toggle(model)} className="h-4 w-4 accent-[var(--accent)]" />
                  <span className="min-w-0 flex-1 truncate font-mono text-[13px]">{model}</span>
                </label>
              ))}
            </div>
          )}
        </div>
        <div className="shrink-0 border-t border-line px-4 py-3">
          <div className="mb-2.5 flex items-center gap-2">
            <Input value={customModel} onChange={(e) => setCustomModel(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addCustom() } }} placeholder={tx("Add a model id")} spellCheck={false} />
            <Button variant="default" size="sm" icon="plus" onClick={addCustom}>{tx("Add")}</Button>
          </div>
          <div className="flex items-center justify-end gap-2">
            <Button variant="ghost" size="sm" onClick={onClose}>{tx("Cancel")}</Button>
            <Button variant="primary" size="sm" icon="check" onClick={save}>{tx("Save")}</Button>
          </div>
        </div>
      </div>
    </div>
  )
}
