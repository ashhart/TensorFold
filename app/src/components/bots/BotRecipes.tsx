import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useMemo, useRef, useState } from 'react'
import { BOT_RECIPE_MAX_BYTES, type BotRecipePreview, type BotRecipeTarget } from '../../../shared/bot-recipe'
import { botProviderSupported } from '../../../shared/bot-policy'
import { isExternalProvider } from '../../../shared/provider-trust'
import { api } from '../../lib/ipc'
import { useStore } from '../../lib/store'
import { modelDisplayName, providerModelGroups } from '../../lib/provider-accounts'
import { Button } from '../ui'

const selectClass = 'h-10 w-full min-w-0 rounded-lg border border-line bg-card-2 px-2 text-[13px] text-ink'
export function BotRecipes({ disabled = false }: { disabled?: boolean }) {
  const tx = useAuthoredText()
  const settings = useStore((state) => state.settings)
  const [mode, setMode] = useState<'import' | 'export' | null>(null)
  const [text, setText] = useState('')
  const [presetId, setPresetId] = useState('')
  const [preview, setPreview] = useState<BotRecipePreview | null>(null)
  const [reviewed, setReviewed] = useState(false)
  const [model, setModel] = useState('')
  const [workspace, setWorkspace] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const inFlight = useRef(false)
  const blocked = disabled || busy || api.platform === 'web-remote'
  const models = useMemo(() => (settings?.providers ?? []).filter((provider) => botProviderSupported(provider) && !isExternalProvider(provider)).flatMap((provider) => providerModelGroups(provider).filter((group) => group.models.length).map((group) => ({ provider, group }))), [settings?.providers])
  const reset = (next: typeof mode) => { setMode(next); setPreview(null); setText(''); setReviewed(false); setPresetId(''); setModel(''); setWorkspace(''); setError(null); setNotice(null) }
  const act = async (operation: () => Promise<void>) => {
    if (inFlight.current || disabled || api.platform === 'web-remote') return
    inFlight.current = true; setBusy(true); setError(null); setNotice(null)
    try { await operation() }
    catch (cause) { setError(cause instanceof Error ? cause.message : tx("Could not finish this recipe. Your review is still here.")) }
    finally { inFlight.current = false; setBusy(false) }
  }
  const review = () => act(async () => {
    const result = await api.botRecipePreview({ kind: mode!, value: mode === 'import' ? text : presetId })
    setPreview(result); setReviewed(false)
  })
  const finish = () => act(async () => {
    if (!preview || !reviewed) throw new Error(tx("Review the instructions first."))
    if (mode === 'import') {
      if (!model || workspace !== 'unassigned') throw new Error(tx("Choose the local model and unassigned workspace."))
      const [providerId, providerAccountId, selectedModel] = JSON.parse(model) as [string, string | null, string]
      const target: BotRecipeTarget = { providerId, providerAccountId, model: selectedModel, workspace: 'unassigned' }
      const saved = await api.botRecipeImport({ text: preview.text, target, reviewed: true })
      useStore.setState({ settings: saved }); reset(null)
      setNotice(tx("Imported as a new paused, read-only bot. Edit its setup and resume it when you are ready."))
    } else {
      const result = await api.botRecipeExport({ presetId, reviewedText: preview.text, reviewed: true })
      if (result.saved) { reset(null); setNotice(tx("Recipe saved. Keep the file private until you are ready to share its instructions.")) }
    }
  })
  return <div className="mb-5">
    <div className="flex flex-wrap gap-2"><Button size="sm" disabled={blocked} onClick={() => reset('import')}>{tx("Import recipe")}</Button><Button size="sm" disabled={blocked || !settings?.agentPresets.length} onClick={() => reset('export')}>{tx("Export recipe")}</Button><span className="self-center text-[12px] text-ink-2">{tx("Share instructions, then choose local setup.")}</span></div>
    {notice && <p role="status" className="mt-3 text-[13px] text-ink-2">{notice}</p>}
    {mode && <section aria-label={tx("Portable bot recipe")} className="mt-3 rounded-xl border border-line-strong bg-card p-4 shadow-card">
      <div className="flex items-start justify-between gap-3"><h2 className="text-[15px] font-semibold">{mode === 'import' ? tx("Review an imported recipe") : tx("Review a recipe before sharing")}</h2><Button size="sm" variant="ghost" disabled={busy} onClick={() => reset(null)}>{tx("Close")}</Button></div>
      <p className="mt-2 max-w-[760px] text-[13px] leading-relaxed text-ink-2">{tx("A recipe contains the name, responsibility and instructions shown below. Instructions can include private text. Common credential and home-folder patterns are checked, but TensorFold cannot find every secret or personal detail. Review every line before sharing or importing.")}</p>
      <p className="mt-2 text-[12px] leading-relaxed text-ink-2">{tx("The file excludes model and account settings, saved credentials, folder grants, VM identities, chats and tool permissions. Your instruction text is included. Importing never starts a chat or grants folder access.")}</p>
      <div className="mt-4 space-y-4">
        {mode === 'import' ? <>
          <label className="block space-y-1.5 text-[13px] text-ink-2"><span className="block font-medium">{tx("Choose a JSON recipe, up to 64 KiB")}</span><input type="file" accept=".json,application/json" disabled={blocked} className="block w-full min-w-0 text-[13px]" onChange={(event) => { const file = event.currentTarget.files?.[0]; event.currentTarget.value = ''; if (!file) return; setPreview(null); setReviewed(false); void act(async () => { if (file.size > BOT_RECIPE_MAX_BYTES) throw new Error(tx("Recipes must be at most 64 KiB.")); setText(await file.text()) }) }} /></label>
          <label className="block space-y-1.5 text-[13px] text-ink-2"><span className="block font-medium">{tx("Or paste recipe JSON")}</span><textarea aria-label={tx("Recipe JSON")} rows={6} maxLength={BOT_RECIPE_MAX_BYTES} value={text} disabled={blocked} onChange={(event) => { setText(event.target.value); setPreview(null); setReviewed(false) }} className="w-full resize-y rounded-lg border border-line bg-card-2 px-3 py-2 font-mono text-[12px] leading-relaxed text-ink outline-none focus:border-accent" /></label>
        </> : <label className="block space-y-1.5 text-[13px] text-ink-2"><span className="block font-medium">{tx("Saved bot")}</span><select aria-label={tx("Bot to export")} value={presetId} disabled={blocked} onChange={(event) => { setPresetId(event.target.value); setPreview(null); setReviewed(false) }} className={selectClass}><option value="">{tx("Choose a bot")}</option>{settings?.agentPresets.map((bot) => <option key={bot.id} value={bot.id}>{bot.name}{bot.archivedAt ? tx(" · archived") : ''}</option>)}</select></label>}
        <Button disabled={blocked || (mode === 'import' ? !text.trim() : !presetId)} onClick={() => void review()}>{busy ? tx("Working…") : tx("Review recipe")}</Button>
        {preview && <div className="space-y-4">
          <div><h3 className="text-[13px] font-medium">{tx("Exactly what the recipe contains")}</h3><pre aria-label={tx("Reviewed recipe contents")} className="mt-2 max-h-[320px] overflow-auto whitespace-pre-wrap break-words rounded-lg border border-line bg-card-2 p-3 font-mono text-[12px] leading-relaxed text-ink">{preview.text}</pre></div>
          {mode === 'import' && <div className="grid gap-4 sm:grid-cols-2">
            <label className="space-y-1.5 text-[13px] text-ink-2"><span className="block font-medium">{tx("Model and account on this Mac")}</span><select aria-label={tx("Recipe local model")} value={model} disabled={blocked} onChange={(event) => setModel(event.target.value)} className={selectClass}><option value="">{tx("Choose a local model")}</option>{models.map(({ provider, group }) => <optgroup key={`${provider.id}:${group.id}`} label={group.label}>{group.models.map((item) => <option key={item} value={JSON.stringify([provider.id, group.accountId, item])}>{modelDisplayName(provider, item)}</option>)}</optgroup>)}</select>{!models.length && <span className="block text-[12px]">{tx("Set up a local model in Settings first. The recipe stays here.")}</span>}</label>
            <label className="space-y-1.5 text-[13px] text-ink-2"><span className="block font-medium">{tx("Workspace setup")}</span><select aria-label={tx("Recipe workspace")} value={workspace} disabled={blocked} onChange={(event) => setWorkspace(event.target.value)} className={selectClass}><option value="">{tx("Choose local setup")}</option><option value="unassigned">{tx("This Mac · no project or folders granted")}</option></select><span className="block text-[12px]">{tx("Imported bots start paused and read only. VM and tool setup stay separate.")}</span></label>
          </div>}
          <label className="flex items-start gap-2 text-[13px] leading-relaxed text-ink"><input type="checkbox" className="mt-1 size-4 shrink-0 accent-accent" checked={reviewed} disabled={blocked} onChange={(event) => setReviewed(event.target.checked)} /><span>{tx("I have reviewed every instruction and understand this text may contain private information.")}</span></label>
          <div className="flex flex-wrap justify-end gap-2"><Button variant="primary" disabled={blocked || !reviewed || (mode === 'import' && (!model || workspace !== 'unassigned'))} onClick={() => void finish()}>{busy ? tx("Working…") : mode === 'import' ? tx("Import paused bot") : tx("Save recipe file…")}</Button></div>
        </div>}
        {error && <p role="alert" className="break-words rounded-lg border border-danger/30 bg-danger/5 p-3 text-[13px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
      </div>
    </section>}
  </div>
}
