import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useState } from 'react'
import { botToolScopeOriginalValue, botToolScopeReviewText, inspectBotToolScope, normalizeBotToolScope } from '../../../shared/bot-tool-policy'
import { api } from '../../lib/ipc'
import type { AgentPreset, McpToolDefinition } from '../../lib/types'
import { Button, Input } from '../ui'

const commonTools = [
  ['ask_user', 'Ask a structured question'], ['read_file', 'Read files'], ['list_dir', 'List folders'],
  ['search', 'Search file contents'], ['glob', 'Find files'], ['write_file', 'Write files'],
  ['edit_file', 'Edit files'], ['run_command', 'Run a command'], ['web_search', 'Search the web'],
  ['web_fetch', 'Read a web page'], ['skills_list', 'List skills'], ['skill_view', 'Read a skill'],
  ['session_search', 'Search saved chats'], ['session_read', 'Read a saved chat'],
  ['phone_observe', 'Observe a granted phone'], ['phone_control', 'Control a granted phone'],
  ['visualize', 'Create a visualization'],
] as const
const guestTools = new Set(['ask_user', 'read_file', 'list_dir', 'search', 'glob', 'write_file', 'edit_file', 'run_command'])

export function BotToolScopeEditor({ bot, onChange }: { bot: AgentPreset; onChange(bot: AgentPreset): void }) {
  const tx = useAuthoredText()
  const inspected = inspectBotToolScope(bot.toolScope)
  const problem = bot.toolScopeProblem !== undefined || !!inspected.error
  const selected = inspected.names
  const restricted = problem || bot.toolScope !== undefined
  const [query, setQuery] = useState('')
  const [otherName, setOtherName] = useState('')
  const [tools, setTools] = useState<McpToolDefinition[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [refresh, setRefresh] = useState(0)
  useEffect(() => {
    setError(null)
    if (problem || !restricted || bot.executionTarget === 'vm') { setLoading(false); return }
    let live = true
    setLoading(true); setError(null)
    api.getMcpSnapshot().then((snapshot) => { if (live) setTools(snapshot.tools) }).catch(() => { if (live) setError(tx("Could not read connected MCP tools. Retry, or keep the current selection.")) }).finally(() => { if (live) setLoading(false) })
    return () => { live = false }
  }, [problem, restricted, bot.executionTarget, refresh])
  const change = (name: string, checked: boolean) => {
    try {
      const toolScope = normalizeBotToolScope({ mode: 'selected', names: checked ? [...selected, name] : selected.filter((value) => value !== name) })!
      onChange({ ...bot, toolScope }); setError(null)
      return true
    } catch (cause) { setError(cause instanceof Error ? cause.message : tx("Check the selected tool names.")); return false }
  }
  const matches = (name: string, title: string) => `${name} ${title}`.toLowerCase().includes(query.trim().toLowerCase())
  const visibleApp = commonTools.filter(([name, title]) => (bot.executionTarget !== 'vm' || guestTools.has(name)) && matches(name, title))
  const visibleMcp = bot.executionTarget === 'vm' ? [] : tools.filter((tool) => matches(tool.callName, `${tool.serverName} ${tool.title ?? tool.name}`)).slice(0, 80)
  const otherSelected = selected.filter((name) => !commonTools.some(([known]) => known === name && (bot.executionTarget !== 'vm' || guestTools.has(name))) && (bot.executionTarget === 'vm' || !tools.some((tool) => tool.callName === name)))
  const row = (name: string, title: string, detail?: string) => <label key={name} className="flex min-w-0 cursor-pointer items-start gap-2.5 rounded-lg border border-line bg-card-2 p-2.5 text-[12px] text-ink">
    <input type="checkbox" className="mt-0.5 size-4 shrink-0 accent-accent" checked={selected.includes(name)} onChange={(event) => change(name, event.target.checked)} />
    <span className="min-w-0"><span className="block break-words font-medium">{title}</span><span className="block break-all leading-relaxed text-ink-2">{detail ?? name}</span></span>
  </label>
  if (problem) return <details open className="rounded-lg border border-line bg-card">
    <summary className="cursor-pointer px-3 py-2.5 text-[13px] font-medium text-ink">{tx("Tool access needs review")}</summary>
    <div className="space-y-3 border-t border-line p-3 text-[12px] leading-relaxed text-ink-2">
      <p role="alert">{tx("The saved tool selection is invalid. This bot remains blocked until you repair it.")}</p>
      <details><summary className="cursor-pointer">{tx("Original saved selection")}</summary><pre className="mt-2 max-h-60 overflow-auto whitespace-pre-wrap break-all rounded-lg bg-card-2 p-3">{botToolScopeReviewText(botToolScopeOriginalValue(bot))}</pre></details>
      <p>{tx("Reset to no tools, then choose the tools you need. Your bot name, instructions and chat history are retained; Discard keeps the original selection until you save.")}</p>
      <Button size="sm" onClick={() => onChange({ ...bot, toolScope: { mode: 'selected', names: [] }, toolScopeProblem: undefined })}>{tx("Reset selection to no tools")}</Button>
    </div>
  </details>
  return <details className="rounded-lg border border-line bg-card" open={restricted || undefined}>
    <summary className="cursor-pointer px-3 py-2.5 text-[13px] font-medium text-ink">{tx("Tool access ")}<span className="ml-1 font-normal text-ink-2">{restricted ? tx("{value0} selected", { value0: selected.length }) : tx("Optional")}</span></summary>
    <div className="space-y-3 border-t border-line p-3">
      <label className="flex cursor-pointer items-center gap-2 text-[13px] text-ink"><input type="checkbox" className="size-4 accent-accent" checked={restricted} onChange={(event) => onChange({ ...bot, toolScope: event.target.checked ? { mode: 'selected', names: [] } : undefined })} />{tx("Use only selected tools")}</label>
      {restricted ? <>
        <p className="text-[12px] leading-relaxed text-ink-2">{selected.length ? tx("Only these named TensorFold tools can run. Your approvals, disabled capabilities and phone grants still apply.") : tx("No tools selected. The bot can chat, but cannot call TensorFold tools.")} {bot.executionTarget === 'vm' ? tx("VM limits also apply; MCP tools are unavailable in the guest.") : tx("Commands and MCP servers can perform broader operations. This is an app tool limit, not an OS sandbox.")}</p>
        <Input aria-label={tx("Find a tool")} value={query} onChange={(event) => setQuery(event.target.value)} placeholder={tx("Find a tool by name")} />
        <div className="grid gap-2 sm:grid-cols-2">{visibleApp.map(([name, title]) => row(name, title))}</div>
        {!visibleApp.length && <p className="text-[12px] text-ink-2">{bot.executionTarget === 'vm' ? tx("No guest tools match this search.") : tx("No app tools match this search.")}</p>}
        {bot.executionTarget !== 'vm' && <div className="space-y-2">
          <div className="flex items-center justify-between gap-2"><p className="text-[12px] font-medium text-ink">{tx("Connected MCP tools")}</p><Button size="sm" variant="ghost" disabled={loading} onClick={() => setRefresh((value) => value + 1)}>{tx("Refresh")}</Button></div>
          {loading ? <p role="status" className="text-[12px] text-ink-2">{tx("Reading tool names…")}</p> : !visibleMcp.length ? <p className="text-[12px] text-ink-2">{tools.length ? tx("No MCP tools match this search.") : tx("No connected MCP tools. Connect a server in Apps, then refresh.")}</p> : <div className="grid gap-2 sm:grid-cols-2">{visibleMcp.map((tool) => row(tool.callName, `${tool.serverName} · ${tool.title ?? tool.name}`, tool.callName))}</div>}
          {tools.length > 80 && <p className="text-[12px] text-ink-2">{tx("Showing up to 80 matches. Search for a specific tool to narrow the list.")}</p>}
          <p className="text-[12px] leading-relaxed text-ink-2">{tx("Every MCP call still follows approval rules. Read-only bots cannot run MCP actions.")}</p>
        </div>}
        {!!otherSelected.length && <div className="grid gap-2 sm:grid-cols-2">{otherSelected.map((name) => row(name, bot.executionTarget === 'vm' ? tx("Unavailable in this VM") : tx("Other saved tool"), name))}</div>}
        {bot.executionTarget !== 'vm' && <details className="text-[12px] text-ink-2"><summary className="cursor-pointer">{tx("Another exact tool name")}</summary><div className="mt-2 flex min-w-0 gap-2"><Input aria-label={tx("Additional exact tool name")} value={otherName} maxLength={200} onChange={(event) => setOtherName(event.target.value)} placeholder={tx("Registered tool name")} className="min-w-0 flex-1" /><Button size="sm" disabled={!otherName.trim()} onClick={() => { if (change(otherName.trim(), true)) setOtherName('') }}>{tx("Add")}</Button></div><p className="mt-2 leading-relaxed">{tx("An unknown or disabled tool stays unavailable. Plugin tools, delegation and global orchestration cannot be added.")}</p></details>}
        <p className="text-[12px] leading-relaxed text-ink-2">{tx("Plugins, delegation and automatic skill review are off for restricted bots. Compaction uses a local summary without extra model requests. TensorFold's Cortheon mode must be off throughout restricted bot work.")}</p>
        {!!selected.length && <Button size="sm" variant="ghost" onClick={() => onChange({ ...bot, toolScope: { mode: 'selected', names: [] } })}>{tx("Clear selections")}</Button>}
      </> : <p className="text-[12px] leading-relaxed text-ink-2">{tx("Uses the existing chat tools and connected sources under your permission rules. Turn this on when a bot needs a narrower list.")}</p>}
      {error && <p role="alert" className="break-words text-[12px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
      <p className="text-[12px] leading-relaxed text-ink-2">{tx("Tool changes apply to new chats. Existing conversations keep the selection captured at creation.")}</p>
    </div>
  </details>
}
