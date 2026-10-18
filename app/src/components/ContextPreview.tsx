import { currentIntlLocale } from '../lib/preferences/current-locale'
import { authoredText } from '../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../lib/store'
import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import { inspectThreadMemoryPolicy } from '../../shared/thread-memory-policy'
import { useEffect, useMemo, useState } from 'react'
import clsx from 'clsx'
import { useStore } from '../lib/store'
import { api } from '../lib/ipc'
import { Icon, type IconName } from '../lib/icons'
import { resolveProviderModel } from '../lib/provider-accounts'
import { resolveThreadInstructionPreview } from '../lib/context-preview'
import type { Thread, ResolvedInstructionGraph, ProjectMemoryCandidate, ProjectMemoryEntry, Skill, Command } from '../lib/types'
import { effectiveThreadCwd } from '../lib/thread-workspace'

const KIND_LABEL: Record<string, string> = { get shared() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "shared") }, get provider() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "overlay") }, get local() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "local") }, get imported() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "import") } }
const CONNECTOR_TOOL_RE = /^(email_|controller|mcp_|bot_|webhook|connector)/i

function roughTokens(text: string): number {
  return Math.max(0, Math.ceil(text.length / 4))
}

function itemPayload(item: Thread['timeline'][number]): string {
  return [item.text, item.output, item.command, item.path, item.tool, item.title, item.meta, item.args ? JSON.stringify(item.args) : ''].filter(Boolean).join('\n')
}

type ContextBucket = {
  id: string
  label: string
  icon: IconName
  count: number
  tokens: number
  detail: string
  active: boolean
}

// "Resolved context" rail section (build.md P0): shows exactly what instruction
// graph the model receives — shared AGENTS.md + overlays + imports, cycles, token
// cost, and the fully-resolved text. Makes hidden context auditable.
export function ContextPreviewSection({ thread }: { thread: Thread }) {
  const tx = useAuthoredText()
  const [graph, setGraph] = useState<ResolvedInstructionGraph | null>(null)
  const [instructionUnavailable, setInstructionUnavailable] = useState<string | null>(null)
  const [memoryEntries, setMemoryEntries] = useState<ProjectMemoryEntry[]>([])
  const [candidates, setCandidates] = useState<ProjectMemoryCandidate[]>([])
  const [skills, setSkills] = useState<Skill[]>([])
  const [commands, setCommands] = useState<Command[]>([])
  const [open, setOpen] = useState(false)
  const [copied, setCopied] = useState(false)
  const provider = useStore((s) => s.settings?.providers.find((p) => p.id === thread.providerId))
  const project = useStore((s) => thread.projectId ? s.projects.find((p) => p.id === thread.projectId) ?? null : null)
  const memoryState = inspectThreadMemoryPolicy(thread)
  const memoryAllowed = memoryState.policy?.useExisting === true
  const memoryInjected = memoryAllowed && thread.kind !== 'chat' && thread.bot?.executionTarget !== 'vm'
  const previewCwd = effectiveThreadCwd(thread, project)

  useEffect(() => {
    let live = true
    setGraph(null)
    setInstructionUnavailable(null)
    resolveThreadInstructionPreview(thread, api.resolveInstructions, project)
      .then((result) => { if (live) { setGraph(result.graph); setInstructionUnavailable(result.unavailableReason) } })
      .catch(() => {})
    return () => { live = false }
    // Graph is a pure function of cwd + providerId — don't re-resolve on every timeline tick (thread.updatedAt).
  }, [thread.id, thread.cwd, thread.providerId, thread.workspace?.kind, thread.workspace?.hostId, thread.workspace?.path, project?.cwd, project?.workspace?.kind, project?.workspace?.hostId, project?.workspace?.path])

  useEffect(() => {
    let live = true
    if (!thread.projectId) { setMemoryEntries([]); setCandidates([]); return () => { live = false } }
    api.getProjectMemory(thread.projectId).then((entries) => { if (live) setMemoryEntries(entries) }).catch(() => {})
    api.getProjectMemoryCandidates(thread.projectId).then((next) => { if (live) setCandidates(next) }).catch(() => {})
    return () => { live = false }
    // Re-run on compaction count too: harvest mints candidates when the middle is folded.
  }, [thread.projectId, thread.compactions?.length])

  const reloadCandidates = () => {
    if (!thread.projectId) return
    api.getProjectMemoryCandidates(thread.projectId).then(setCandidates).catch(() => {})
  }

  useEffect(() => {
    let live = true
    Promise.all([api.getSkills(), api.getCommands()])
      .then(([nextSkills, nextCommands]) => { if (live) { setSkills(nextSkills); setCommands(nextCommands) } })
      .catch(() => {})
    return () => { live = false }
  }, [])

  const nodes = graph?.nodes ?? []
  const hasContent = !!graph?.resolvedContent.trim()
  const buckets = useMemo<ContextBucket[]>(() => {
    const userItems = thread.timeline.filter((item) => item.kind === 'user' || item.kind === 'cue')
    const toolItems = thread.timeline.filter((item) => item.kind === 'tool')
    const connectorItems = thread.timeline.filter((item) => CONNECTOR_TOOL_RE.test(item.tool || '') || /<vontra_capability|connector|webhook/i.test(itemPayload(item)))
    const filePaths = new Set<string>()
    for (const node of nodes) if (node.path) filePaths.add(node.path)
    for (const item of thread.timeline) if (item.path) filePaths.add(item.path)
    const enabledSkills = skills.filter((skill) => skill.enabled)
    const enabledCommands = commands
    const userText = userItems.map(itemPayload).join('\n')
    const toolText = toolItems.map(itemPayload).join('\n')
    const connectorText = connectorItems.map(itemPayload).join('\n')
    const memoryText = memoryInjected ? [project?.memory, ...memoryEntries.map((entry) => entry.text)].filter(Boolean).join('\n') : ''
    const skillHits = enabledSkills.filter((skill) => new RegExp(`\\$${skill.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i').test(userText))
    const commandHits = enabledCommands.filter((command) => new RegExp(`/${command.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i').test(userText))
    return [
      { id: 'system', label: tx("System"), icon: 'shield', count: nodes.length, tokens: graph?.tokenEstimate ?? 0, detail: instructionUnavailable ? 'Remote preview unavailable' : nodes.length ? 'Resolved instructions' : 'No instruction files', active: !!graph?.tokenEstimate },
      { id: 'user', label: tx("User"), icon: 'message', count: userItems.length, tokens: roughTokens(userText), detail: tx("User turns and Guide notes"), active: userItems.length > 0 },
      { id: 'tools', label: tx("Tools"), icon: 'terminal', count: toolItems.length, tokens: roughTokens(toolText) + (toolItems.length ? 220 : 0), detail: toolItems.length ? 'Tool calls and outputs' : 'Schemas only', active: toolItems.length > 0 },
      { id: 'memory', label: tx("Memory"), icon: 'layers', count: (project?.memory?.trim() ? 1 : 0) + memoryEntries.length, tokens: roughTokens(memoryText), detail: memoryState.problem ? 'Saved-memory choices need review' : !memoryAllowed ? 'Saved-memory recall is off' : !memoryInjected ? 'Available through tools; not injected in chat' : memoryEntries.length ? 'Project memory entries' : 'No typed memory', active: !!memoryText.trim() },
      { id: 'files', label: tx("Files"), icon: 'fileText', count: filePaths.size, tokens: nodes.reduce((sum, node) => sum + node.tokenEstimate, 0), detail: filePaths.size ? 'Instruction and touched files' : 'No file context', active: filePaths.size > 0 },
      { id: 'skills', label: tx("Skills"), icon: 'spark', count: skillHits.length + commandHits.length, tokens: roughTokens([...skillHits.map((skill) => skill.body), ...commandHits.map((command) => command.prompt)].join('\n')), detail: `${enabledSkills.length} skills, ${enabledCommands.length} commands available`, active: skillHits.length + commandHits.length > 0 },
      { id: 'connector', label: tx("Connector"), icon: 'plug', count: connectorItems.length, tokens: roughTokens(connectorText), detail: connectorItems.length ? 'Connector and capability context' : 'No connector context', active: connectorItems.length > 0 },
    ]
  }, [memoryState.problem, memoryAllowed, memoryInjected, thread.timeline, nodes, graph?.tokenEstimate, instructionUnavailable, project?.memory, memoryEntries, skills, commands, tx])
  const copy = async () => {
    if (!graph) return
    await api.writeClipboard(graph.resolvedContent)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1200)
  }
  const providerName = provider?.name ?? thread.providerId ?? tx("No provider")
  const displayModel = provider ? (resolveProviderModel(provider, thread.providerAccountId ?? null, thread.model) ?? thread.model) : thread.model
  const staleModel = thread.model && displayModel && thread.model !== displayModel ? thread.model : null
  const modelTitle = staleModel ? tx("Saved model unavailable: {value0}. Using {value1}.", { value0: String(staleModel), value1: String(displayModel) }) : displayModel ?? undefined

  return (
    <div className="px-4 py-3.5">
      <div className="mb-2.5 flex items-center justify-between">
        <span className="text-[13px] font-normal text-ink-3">{tx("Context")}</span>
        {nodes.length > 0 && <span className="rounded bg-ink-5 px-1.5 text-[12px] tabular-nums text-ink-3">{nodes.length}</span>}
      </div>

      <div className="rounded-xl border border-line bg-card p-3 shadow-card">
        <div className="flex min-w-0 items-center gap-2 text-[13px] text-ink-2">
          <Icon name={provider?.kind === 'local' ? 'cpu' : 'globe'} size={14} className="shrink-0 text-ink-3" />
          <span className="min-w-0 flex-1 truncate" title={providerName}>{providerName}</span>
        </div>
        {displayModel && (
          <div className="mt-1 flex min-w-0 items-center gap-2 pl-[22px]">
            <span className="min-w-0 max-w-full truncate font-mono text-[12px] text-ink-3" title={modelTitle}>{displayModel}</span>
          </div>
        )}
        <div className="mt-1.5 flex items-center gap-2 text-[13px] text-ink-3">
          <Icon name="folder" size={13} className="text-ink-4" />
          <span className="truncate">{previewCwd || tx("No working directory")}</span>
        </div>
        {graph && (
          <div className="mt-2 flex items-center gap-1.5 text-[12px] text-ink-3">
            <Icon name="layers" size={12} className="text-ink-4" />
            <span>~{graph.tokenEstimate.toLocaleString(currentIntlLocale())} {tx("tok")}</span>
            <span className="text-ink-4">· {hasContent ? tx("Instruction files: {count}", { count: nodes.length }) : tx("no AGENTS.md found")}</span>
          </div>
        )}
        {instructionUnavailable && (
          <div className="mt-2 flex items-start gap-1.5 rounded-md border border-line bg-card-2 px-2 py-1.5 text-[12px] text-ink-3">
            <Icon name="info" size={12} className="mt-0.5 shrink-0 text-ink-4" />
            <span>{instructionUnavailable}</span>
          </div>
        )}
      </div>

      <div className="mb-2.5 mt-3 flex items-center justify-between">
        <span className="text-[13px] font-normal text-ink-3">{tx("Buckets")}</span>
        <span className="rounded bg-ink-5 px-1.5 text-[12px] tabular-nums text-ink-3">7</span>
      </div>
      <div className="grid grid-cols-2 gap-1.5">
        {buckets.map((bucket) => (
          <div key={bucket.id} className={clsx('rounded-lg border px-2 py-1.5', bucket.active ? 'border-line bg-card shadow-card' : 'border-line bg-card-2/50 opacity-70')}>
            <div className="flex items-center gap-1.5">
              <Icon name={bucket.icon} size={12} className={bucket.active ? 'text-ink-3' : 'text-ink-4'} />
              <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-ink-2">{bucket.label}</span>
              <span className="font-mono text-[11px] tabular-nums text-ink-4">{bucket.count}</span>
            </div>
            <div className="mt-0.5 truncate text-[12px] text-ink-4">{bucket.detail}</div>
            <div className="mt-1 font-mono text-[11px] tabular-nums text-ink-3">~{bucket.tokens.toLocaleString(currentIntlLocale())} {tx("tok")}</div>
          </div>
        ))}
      </div>

      {/* per-node provenance list */}
      {nodes.length > 0 && (
        <div className="mt-2 space-y-0.5">
          {nodes.map((n) => (
            <div key={n.id} className="flex items-center gap-2 rounded-md px-1.5 py-1 text-[13px]">
              <Icon name="file" size={12} className="shrink-0 text-ink-4" />
              <span className={clsx('truncate', n.ignored ? 'text-ink-4 line-through' : 'text-ink-2')}>{n.path}</span>
              <span className="ml-auto shrink-0 font-mono text-[11px] uppercase tracking-wide text-ink-4">{n.ignored ? tx("ignored") : KIND_LABEL[n.kind] ?? n.kind}</span>
            </div>
          ))}
        </div>
      )}

      {/* cycle warnings */}
      {!!graph?.cycles.length && (
        <div className="mt-2 flex items-start gap-1.5 rounded-md bg-warn/10 px-2 py-1.5 text-[12px] text-warn">
          <Icon name="warning" size={12} className="mt-0.5 shrink-0" />
          <span>{tx("Import cycle detected: ")}{graph.cycles.join(', ')}</span>
        </div>
      )}

      {/* resolved-content disclosure — exactly what the model receives */}
      {hasContent && (
        <>
          <div className="mt-2 flex items-center justify-between">
            <button onClick={() => setOpen((o) => !o)} className="flex items-center gap-1 text-[13px] text-ink-3 transition-colors hover:text-ink">
              <Icon name={open ? 'chevronDown' : 'chevronRight'} size={13} />
              {open ? tx("Hide") : tx("Show")} {tx("resolved instructions")}</button>
          </div>
          {open && (
            <div className="mt-1.5 overflow-hidden rounded-lg border border-line bg-card-2">
              <pre className="scroll-thin max-h-64 overflow-auto whitespace-pre-wrap p-2.5 font-mono text-[13px] leading-relaxed text-ink-2">
                {graph!.resolvedContent}
              </pre>
              <div className="flex justify-end border-t border-line bg-card px-2 py-1">
                <button onClick={copy} className="inline-flex h-7 items-center gap-1.5 rounded-md px-2 text-[12px] font-medium text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink" title={tx("Copy resolved instructions")}>
                  <Icon name={copied ? 'check' : 'copy'} size={12} />{copied ? tx("Copied") : tx("Copy")}
                </button>
              </div>
            </div>
          )}
        </>
      )}

      {/* Compaction ledger — what older context was folded into a summary (build.md P1) */}
      {!!thread.compactions?.length && (
        <div className="mt-3">
          <div className="mb-1.5 flex items-center justify-between">
            <span className="text-[13px] font-normal text-ink-3">{tx("Compaction")}</span>
            <span className="rounded bg-ink-5 px-1.5 text-[12px] tabular-nums text-ink-3">{thread.compactions.length}</span>
          </div>
          <div className="space-y-1.5">
            {thread.compactions.slice().reverse().map((c) => (
              <div key={c.id} className="rounded-lg border border-line bg-card p-2 text-[13px]">
                <div className="flex items-center gap-2 text-ink-3">
                  <Icon name="layers" size={12} className="text-ink-4" />
                  <span className="tabular-nums">{c.messagesFolded} {tx("folded")}</span>
                  <span className="text-ink-4">· ~{c.tokensBefore.toLocaleString(currentIntlLocale())} → ~{c.tokensAfter.toLocaleString(currentIntlLocale())} {tx("tok")}</span>
                  <span className="ml-auto font-mono text-[11px] uppercase text-ink-4">{c.trigger}</span>
                </div>
                {c.summaryExcerpt && <p className="mt-1 line-clamp-3 text-ink-2">{c.summaryExcerpt}</p>}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Harvested memory candidates — durable facts salvaged from the folded middle at
          compaction. Approve before anything is written durably. Mirrors the project-room
          review, surfaced here so the harvest → approve loop closes in the same panel. */}
      {candidates.length > 0 && (
        <div className="mt-3 border-t border-line pt-3">
          <div className="mb-1.5 flex items-center gap-2">
            <Icon name="layers" size={12} className="text-ink-4" />
            <span className="text-[13px] font-normal text-ink-3">{tx("Memory candidates")}</span>
            <span className="rounded bg-ink-5 px-1.5 text-[12px] tabular-nums text-ink-3">{candidates.length}</span>
            <span className="ml-auto text-[12px] text-ink-4">{tx("approve before durable write")}</span>
          </div>
          <div className="space-y-1.5">
            {candidates.map((candidate) => (
              <div key={candidate.id} className="flex items-start gap-2 rounded-lg border border-line bg-card p-2 text-[13px]">
                <span className="mt-0.5 shrink-0 font-mono text-[11px] uppercase tracking-wide text-ink-4">{candidate.kind}</span>
                <div className="min-w-0 flex-1">
                  <span className="block text-ink-2">{candidate.text}</span>
                  <span className="mt-0.5 block truncate text-[11px] text-ink-4">{candidate.sourceType}{candidate.reason ? ` · ${candidate.reason}` : ''}</span>
                </div>
                <button onClick={() => api.approveProjectMemoryCandidate(candidate.id).then(reloadCandidates).catch(() => {})} className="shrink-0 rounded-md bg-positive/15 px-2 py-1 text-[12px] font-medium text-positive hover:bg-positive/25">{tx("Approve")}</button>
                <button onClick={() => api.rejectProjectMemoryCandidate(candidate.id).then(reloadCandidates).catch(() => {})} className="shrink-0 rounded-md border border-line px-2 py-1 text-[12px] text-ink-3 hover:text-danger">{tx("Reject")}</button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
