import { useEffect, useMemo, useState } from 'react'
import clsx from 'clsx'
import { api } from '../../lib/ipc'
import { Icon } from '../../lib/icons'
import { Button, Field, Input } from '../ui'
import { Spinner, type SettingsSectionContext } from './shared'
import { cliToolState, cliToolStatusText } from '../../lib/cli-tool-display'
import type { CliToolStatus, Provider, Settings } from '../../lib/types'

type ToolLink = {
  docsUrl?: string
  headlessUrl?: string
  configUrl?: string
  installCommand?: string
  detail: string
}

const TOOL_LINKS: Record<string, ToolLink> = {
  codex: {
    docsUrl: 'https://github.com/openai/codex',
    configUrl: 'https://github.com/openai/codex',
    installCommand: 'npm install -g @openai/codex',
    detail: 'Runs Codex CLI sessions, resumable terminals, and OAuth-backed code tasks.',
  },
  claude: {
    docsUrl: 'https://docs.anthropic.com/en/docs/claude-code',
    configUrl: 'https://docs.anthropic.com/en/docs/claude-code/settings',
    installCommand: 'npm install -g @anthropic-ai/claude-code',
    detail: 'Runs Claude Code through your local Claude account and permission surface.',
  },
  cursor: {
    docsUrl: 'https://docs.cursor.com',
    installCommand: 'Install Cursor, then run cursor-agent from its CLI tools.',
    detail: 'Cursor Agent CLI candidate for ACP/headless coding sessions.',
  },
  gemini: {
    docsUrl: 'https://github.com/google-gemini/gemini-cli',
    configUrl: 'https://github.com/google-gemini/gemini-cli',
    installCommand: 'npm install -g @google/gemini-cli',
    detail: 'Gemini CLI candidate for code and research sessions.',
  },
  grok: {
    docsUrl: 'https://docs.x.ai',
    installCommand: 'Install the Grok CLI, then leave this path blank to use grok from PATH.',
    detail: 'Grok CLI candidate for xAI coding sessions.',
  },
  opencode: {
    docsUrl: 'https://opencode.ai/docs',
    configUrl: 'https://opencode.ai/docs/config',
    installCommand: 'npm install -g opencode-ai',
    detail: 'OpenCode CLI candidate for OpenCode-compatible coding sessions.',
  },
  kilo: {
    docsUrl: 'https://kilocode.ai',
    installCommand: 'Install Kilo, then leave this path blank to use kilo from PATH.',
    detail: 'Kilo CLI candidate for OpenCode-compatible coding sessions.',
  },
  pi: {
    docsUrl: 'https://github.com/earendil-works/pi',
    installCommand: 'npm install -g @earendil-works/pi-coding-agent',
    detail: 'Pi coding-agent harness for terminal-based agent sessions.',
  },
}

function providerTransportForTool(id: string): Provider['transport'] | null {
  if (id === 'codex') return 'codex-cli'
  if (id === 'claude') return 'claude-cli'
  return null
}

function defaultBinary(tool: CliToolStatus) {
  return tool.command?.trim().split(/\s+/)[0] ?? ''
}

function configuredBinary(draft: Settings, tool: CliToolStatus) {
  const transport = providerTransportForTool(tool.id)
  const providerPath = transport
    ? draft.providers.find((provider) => provider.transport === transport)?.cliPath?.trim()
    : ''
  const configured = providerPath || draft.cliToolOverrides?.[tool.id]?.binaryPath || ''
  const trimmed = configured.trim()
  const fallback = defaultBinary(tool)
  return trimmed && trimmed !== fallback && trimmed !== tool.command?.trim() ? configured : ''
}

function patchToolBinary(draft: Settings, tool: CliToolStatus, value: string): Settings {
  const transport = providerTransportForTool(tool.id)
  if (transport) {
    return {
      ...draft,
      providers: draft.providers.map((provider) => provider.transport === transport ? { ...provider, cliPath: value.trim() } : provider),
    }
  }
  const overrides = { ...(draft.cliToolOverrides ?? {}) }
  const trimmed = value.trim()
  if (trimmed) overrides[tool.id] = { ...(overrides[tool.id] ?? {}), binaryPath: value }
  else delete overrides[tool.id]
  return { ...draft, cliToolOverrides: overrides }
}

const STATE_TONE = { installed: 'text-ink-3', missing: 'text-ink-4', broken: 'text-danger' } as const

export function ProviderToolsPanel({ ctx }: { ctx: SettingsSectionContext }) {
  const [tools, setTools] = useState<CliToolStatus[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<string | null>('codex')
  const [copied, setCopied] = useState<string | null>(null)
  const [refreshNonce, setRefreshNonce] = useState(0)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    api.listCliTools()
      .then((value) => {
        if (cancelled) return
        setTools(value.filter((tool) => tool.id !== 'shell'))
      })
      .catch((e) => {
        if (cancelled) return
        setError(String(e?.message ?? e))
      })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [refreshNonce])

  const summary = useMemo(() => {
    const count = (state: string) => tools.filter((tool) => cliToolState(tool) === state).length
    const broken = count('broken')
    return `${count('installed')} installed · ${count('missing')} missing${broken ? ` · ${broken} broken` : ''}`
  }, [tools])

  const saveBinary = (tool: CliToolStatus, value: string) => {
    const next = patchToolBinary(ctx.draft, tool, value)
    ctx.commit(next)
    setRefreshNonce((nonce) => nonce + 1)
  }

  const copyInstall = async (tool: CliToolStatus) => {
    const command = TOOL_LINKS[tool.id]?.installCommand
    if (!command) return
    await api.writeClipboard(command)
    setCopied(tool.id)
    window.setTimeout(() => setCopied(null), 1400)
  }

  const openLink = (url?: string) => {
    if (url) void api.openExternal(url).catch(() => {})
  }

  return (
    <section className="mb-8">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-[15px] font-semibold tracking-[-0.01em] text-ink">Provider tools</h2>
          <p className="mt-0.5 text-[13px] text-ink-3">Installed CLIs, versions, docs, and binary overrides for local agent harnesses.</p>
        </div>
        <Button variant="default" size="sm" icon="refresh" onClick={() => setRefreshNonce((nonce) => nonce + 1)} disabled={loading}>
          {loading ? 'Checking' : 'Refresh'}
        </Button>
      </div>
      <div className="mt-3.5 rounded-xl border border-line bg-card shadow-card">
        <div className="border-b border-line px-3.5 py-3">
          <div className="text-[14px] font-semibold text-ink">Installed CLIs</div>
          <div className="mt-1 text-[13px] leading-snug text-ink-3">
            {loading ? 'Reviewing provider versions and local binaries.' : error ? error : summary}
          </div>
        </div>
        <div className="divide-y divide-line">
          {loading && tools.length === 0 ? (
            <div className="flex items-center gap-2 px-3.5 py-4 text-[13px] text-ink-3"><Spinner size={13} /> Checking provider tools</div>
          ) : tools.map((tool) => {
            const open = expanded === tool.id
            const links = TOOL_LINKS[tool.id]
            const configured = configuredBinary(ctx.draft, tool)
            const fallback = defaultBinary(tool)
            const disabled = !tool.command
            const state = cliToolState(tool)
            const status = cliToolStatusText(tool)
            // A broken CLI's error stands in for its description; opened, the full error sits below.
            const description = state === 'broken' && tool.problem && !open ? tool.problem : links?.detail ?? 'Provider CLI tool.'
            const linkButtons = links ? [
              { label: 'Docs', icon: 'arrowRight' as const, url: links.docsUrl },
              { label: 'Headless', icon: 'terminal' as const, url: links.headlessUrl },
              { label: 'Config', icon: 'settings' as const, url: links.configUrl },
            ].filter((link) => link.url) : []
            return (
              <div key={tool.id}>
                <button
                  type="button"
                  onClick={() => setExpanded(open ? null : tool.id)}
                  className={clsx('flex w-full items-center gap-3 px-3.5 py-3 text-left transition-colors hover:bg-ink-5/60', open && 'bg-ink-5/40')}
                >
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-line bg-card-2" style={{ color: tool.color }}>
                    <Icon name={(tool.icon as any) ?? 'terminal'} size={16} />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[14px] font-semibold text-ink" title={tool.name}>{tool.name}</span>
                    <span className="mt-0.5 block truncate text-[13px] text-ink-3" title={description}>{description}</span>
                  </span>
                  <span className={clsx('hidden max-w-[40%] shrink-0 truncate text-[13px] tabular-nums sm:block', STATE_TONE[state])} title={status.title}>{status.text}</span>
                  <Icon name={open ? 'chevronDown' : 'chevronRight'} size={15} className="shrink-0 text-ink-3" />
                </button>
                {open && (
                  <div className="min-w-0 space-y-3 px-3.5 pb-3.5">
                    {state === 'broken' && (
                      <div className="break-words text-[13px] leading-relaxed text-ink-3">
                        <span className="font-mono text-ink-2">{fallback || tool.short} --version</span> failed: {tool.problem || 'it printed no version'}
                      </div>
                    )}
                    {(linkButtons.length > 0 || links?.installCommand) && (
                      <div className="-ml-2.5 flex flex-wrap items-center gap-1">
                        {linkButtons.map((link) => (
                          <Button key={link.label} variant="ghost" size="xs" icon={link.icon} onClick={() => openLink(link.url)}>{link.label}</Button>
                        ))}
                        {links?.installCommand && (
                          <Button variant="ghost" size="xs" icon={copied === tool.id ? 'check' : 'copy'} onClick={() => void copyInstall(tool)} title={links.installCommand}>{copied === tool.id ? 'Copied' : 'Copy install command'}</Button>
                        )}
                      </div>
                    )}
                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-[1fr_auto]">
                      <Field label={`${tool.name} binary path`}>
                        <Input
                          value={configured}
                          onChange={(e) => ctx.setDraft((draft: Settings) => patchToolBinary(draft, tool, e.target.value))}
                          placeholder={fallback ? `${fallback} binary path` : 'No binary path'}
                          spellCheck={false}
                          disabled={disabled}
                        />
                      </Field>
                      <div className="flex items-end gap-2">
                        <Button variant="default" size="sm" icon="x" onClick={() => saveBinary(tool, '')} disabled={disabled || !configured}>Reset</Button>
                        <Button variant="default" size="sm" icon="check" onClick={() => saveBinary(tool, configured)} disabled={disabled}>Save</Button>
                      </div>
                    </div>
                    <div className="break-words text-[13px] leading-relaxed text-ink-3">
                      Leave blank to use <span className="font-mono text-ink-2">{fallback || tool.short}</span> from your PATH.
                      {tool.configuredPath && <span> Current effective command: <span className="font-mono text-ink-2">{tool.effectiveCommand}</span>.</span>}
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
