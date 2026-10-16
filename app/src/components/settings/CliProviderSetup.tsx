import { useCallback, useEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import { Icon } from '../../lib/icons'
import { api } from '../../lib/ipc'
import { Button, Field, Input, Segmented } from '../ui'
import { AdvancedSection } from './shared'
import type { Provider, ProviderAccountProfile } from '../../lib/types'

// The simplified Codex/Claude CLI add experience (replaces the old wall of fields).
//   single → detect the one installed login and connect in a click.
//   multi  → a step-by-step wizard to set up several subscription accounts.
type Transport = 'codex-cli' | 'claude-cli'

const META: Record<Transport, { label: string; bin: string; install: string }> = {
  'codex-cli': { label: 'Codex', bin: 'codex', install: 'npm i -g @openai/codex' },
  'claude-cli': { label: 'Claude', bin: 'claude', install: 'npm i -g @anthropic-ai/claude-code' },
}

let _rowSeq = 0
const rid = () => `row_${++_rowSeq}_${Math.random().toString(36).slice(2, 7)}`

interface AccountRow { id: string; name: string; cliHome: string; cliProfile: string }
interface Detect { state: 'checking' | 'ready' | 'needs-login' | 'missing' | 'error'; detail?: string; message?: string }

function defaultAccounts(transport: Transport): AccountRow[] {
  return transport === 'codex-cli'
    ? [
        { id: rid(), name: 'Personal Codex', cliHome: '~/.codex-personal', cliProfile: 'personal' },
        { id: rid(), name: 'Work Codex', cliHome: '~/.codex-work', cliProfile: 'work' },
      ]
    : [
        { id: rid(), name: 'Personal Claude', cliHome: '~/.claude-personal', cliProfile: '~/.claude-personal/settings.json' },
        { id: rid(), name: 'Work Claude', cliHome: '~/.claude-work', cliProfile: '~/.claude-work/settings.json' },
      ]
}

// Mirrors electron/agent — the exact shell the CLI needs to sign a given account in.
function loginCommand(transport: Transport, bin: string, cliHome: string): string {
  const home = (cliHome.trim() || (transport === 'codex-cli' ? '~/.codex' : '~/.claude'))
  return transport === 'codex-cli'
    ? `mkdir -p ${home} && CODEX_HOME=${home} ${bin} login`
    : `mkdir -p ${home} && CLAUDE_CONFIG_DIR=${home} CLAUDE_HOME=${home} ${bin} login`
}

function CopyCommand({ command }: { command: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      onClick={async () => { await api.writeClipboard(command); setCopied(true); window.setTimeout(() => setCopied(false), 1400) }}
      className="group flex w-full items-center gap-2 rounded-lg border border-line bg-card px-2.5 py-1.5 text-left font-mono text-[12px] text-ink-3 transition-colors hover:border-line-strong hover:text-ink-2"
      title="Copy the sign-in command"
    >
      <Icon name={copied ? 'check' : 'copy'} size={12} className="shrink-0" />
      <span className="truncate">{command}</span>
    </button>
  )
}

export function CliProviderSetup({ transport, cliSubscriptionMode, cliPath, cliProfile, onPatch, addCliProvider }: {
  transport: Transport
  cliSubscriptionMode: 'single' | 'multi'
  cliPath: string
  cliProfile: string
  onPatch: (patch: { cliSubscriptionMode?: 'single' | 'multi'; cliPath?: string; cliProfile?: string }) => void
  addCliProvider: (transport: Transport, accountSeeds: Partial<ProviderAccountProfile>[] | null) => Promise<Provider>
}) {
  const meta = META[transport]

  return (
    <div className="space-y-3">
      <Field label="How many subscriptions?">
        <Segmented<'single' | 'multi'>
          value={cliSubscriptionMode}
          onChange={(mode) => onPatch({ cliSubscriptionMode: mode })}
          options={[
            { value: 'single', label: 'Single', icon: 'shield' },
            { value: 'multi', label: 'Multiple', icon: 'layers' },
          ]}
        />
      </Field>

      {cliSubscriptionMode === 'single'
        ? <SingleSetup transport={transport} cliPath={cliPath} cliProfile={cliProfile} onPatch={onPatch} addCliProvider={addCliProvider} meta={meta} />
        : <MultiSetup transport={transport} addCliProvider={addCliProvider} meta={meta} />}
    </div>
  )
}

// ---------------------------------------------------------------- single
function SingleSetup({ transport, cliPath, cliProfile, onPatch, addCliProvider, meta }: {
  transport: Transport
  cliPath: string
  cliProfile: string
  onPatch: (patch: { cliPath?: string; cliProfile?: string }) => void
  addCliProvider: (transport: Transport, accountSeeds: Partial<ProviderAccountProfile>[] | null) => Promise<Provider>
  meta: { label: string; bin: string; install: string }
}) {
  const [detect, setDetect] = useState<Detect>({ state: 'checking' })
  const [busy, setBusy] = useState(false)
  const [added, setAdded] = useState<string | null>(null)
  const bin = cliPath.trim() || meta.bin

  const recheck = useCallback(async () => {
    setDetect({ state: 'checking' })
    try {
      const r = await api.checkCliTransport({ transport, cliPath: cliPath.trim() || undefined })
      if (!r.binaryPresent) setDetect({ state: 'missing', message: r.message })
      else if (r.loggedIn) setDetect({ state: 'ready', detail: r.detail })
      else setDetect({ state: 'needs-login', message: r.message })
    } catch (e: any) {
      setDetect({ state: 'error', message: String(e?.message ?? e) })
    }
  }, [transport, cliPath])

  // Detect on mount and whenever the binary override changes.
  useEffect(() => { void recheck() }, [recheck])

  const connect = async () => {
    setBusy(true)
    try { const p = await addCliProvider(transport, null); setAdded(p.name) } finally { setBusy(false) }
  }
  const addAndSignIn = async () => {
    setBusy(true)
    try {
      const p = await addCliProvider(transport, null)
      setAdded(p.name)
      const res = await api.loginCliProvider({ providerId: p.id })
      setDetect(res.loggedIn ? { state: 'ready', detail: res.detail } : { state: 'needs-login', message: res.message })
    } finally { setBusy(false) }
  }

  if (added) {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-positive/30 bg-positive/10 px-3 py-2.5 text-[13px] text-positive">
        <Icon name="check" size={15} className="shrink-0" />
        <span>Added <span className="font-medium">{added}</span> — pick it on any chat, room, agent, or automation.</span>
      </div>
    )
  }

  return (
    <div className="space-y-2.5">
      {detect.state === 'checking' && (
        <div className="flex items-center gap-2 rounded-lg border border-line bg-card-2 px-3 py-2.5 text-[13px] text-ink-3">
          <Icon name="refresh" size={14} className="shrink-0 animate-spin" />
          Looking for {meta.label} on this machine…
        </div>
      )}

      {detect.state === 'ready' && (
        <div className="rounded-lg border border-positive/30 bg-positive/10 px-3 py-2.5">
          <div className="flex items-center gap-2 text-[13px] text-positive">
            <Icon name="check" size={15} className="shrink-0" />
            <span>{meta.label} detected — signed in{detect.detail ? <> as <span className="font-medium">{detect.detail}</span></> : ''}.</span>
          </div>
          <div className="mt-2.5 flex items-center gap-2">
            <Button variant="primary" size="sm" icon="plug" onClick={connect} disabled={busy}>{busy ? 'Connecting' : 'Connect'}</Button>
            <button onClick={() => void recheck()} disabled={busy} className="text-[13px] text-ink-3 transition-colors hover:text-ink">Re-check</button>
          </div>
        </div>
      )}

      {detect.state === 'needs-login' && (
        <div className="rounded-lg border border-warn/30 bg-warn/10 px-3 py-2.5">
          <div className="flex items-center gap-2 text-[13px] text-warn">
            <Icon name="warning" size={15} className="shrink-0" />
            <span>{meta.label} is installed but not signed in.</span>
          </div>
          <div className="mt-2.5 flex flex-wrap items-center gap-2">
            <Button variant="primary" size="sm" icon="plug" onClick={addAndSignIn} disabled={busy}>{busy ? 'Signing in…' : 'Add & sign in'}</Button>
            <button onClick={() => void recheck()} disabled={busy} className="text-[13px] text-ink-3 transition-colors hover:text-ink">Re-check</button>
          </div>
          <div className="mt-2.5 text-[12px] text-ink-3">Sign-in opens your browser. Or run it yourself, then Re-check:</div>
          <div className="mt-1.5"><CopyCommand command={loginCommand(transport, bin, '')} /></div>
        </div>
      )}

      {detect.state === 'missing' && (
        <div className="rounded-lg border border-danger/30 bg-danger/5 px-3 py-2.5">
          <div className="flex items-center gap-2 text-[13px] text-danger">
            <Icon name="warning" size={15} className="shrink-0" />
            <span>Couldn't find <span className="font-mono">{bin}</span> on your PATH.</span>
          </div>
          <div className="mt-2 text-[13px] text-ink-3">Install it, then re-check:</div>
          <div className="mt-1.5"><CopyCommand command={meta.install} /></div>
          <div className="mt-2.5"><Button variant="default" size="sm" icon="refresh" onClick={() => void recheck()}>Re-check</Button></div>
        </div>
      )}

      {detect.state === 'error' && (
        <div className="flex items-center justify-between gap-2 rounded-lg border border-danger/30 bg-danger/5 px-3 py-2.5 text-[13px] text-danger">
          <span className="truncate">{detect.message ?? 'Detection failed.'}</span>
          <Button variant="default" size="sm" icon="refresh" onClick={() => void recheck()}>Re-check</Button>
        </div>
      )}

      <AdvancedSection title="Advanced" icon="settings">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <Field label="CLI binary" hint={`defaults to ${meta.bin}`}>
            <Input value={cliPath} onChange={(e) => onPatch({ cliPath: e.target.value })} placeholder={meta.bin} spellCheck={false} />
          </Field>
          <Field label={transport === 'codex-cli' ? 'Profile' : 'Settings file'} hint="optional">
            <Input value={cliProfile} onChange={(e) => onPatch({ cliProfile: e.target.value })} placeholder={transport === 'codex-cli' ? 'work' : '/path/to/settings.json'} spellCheck={false} />
          </Field>
        </div>
      </AdvancedSection>
    </div>
  )
}

// ---------------------------------------------------------------- multi (wizard)
function MultiSetup({ transport, addCliProvider, meta }: {
  transport: Transport
  addCliProvider: (transport: Transport, accountSeeds: Partial<ProviderAccountProfile>[] | null) => Promise<Provider>
  meta: { label: string; bin: string; install: string }
}) {
  const [rows, setRows] = useState<AccountRow[]>(() => defaultAccounts(transport))
  const [created, setCreated] = useState<Provider | null>(null)
  const [busy, setBusy] = useState(false)
  const [signin, setSignin] = useState<Record<string, 'busy' | 'ok' | 'fail'>>({})
  const [signinDetail, setSigninDetail] = useState<Record<string, string | undefined>>({})

  const patchRow = (id: string, patch: Partial<AccountRow>) => setRows((rs) => rs.map((r) => r.id === id ? { ...r, ...patch } : r))
  const removeRow = (id: string) => setRows((rs) => rs.filter((r) => r.id !== id))
  const addRow = () => setRows((rs) => [...rs, { id: rid(), name: `${meta.label} account ${rs.length + 1}`, cliHome: `~/.${meta.bin}-${rs.length + 1}`, cliProfile: '' }])

  const create = async () => {
    const seeds = rows
      .filter((r) => r.name.trim())
      .map((r) => ({ name: r.name.trim(), cliHome: r.cliHome.trim(), cliProfile: r.cliProfile.trim() }))
    if (!seeds.length) return
    setBusy(true)
    try { setCreated(await addCliProvider(transport, seeds)) } finally { setBusy(false) }
  }
  const signIn = async (account: ProviderAccountProfile) => {
    if (!created) return
    setSignin((s) => ({ ...s, [account.id]: 'busy' }))
    try {
      const res = await api.loginCliProvider({ providerId: created.id, accountId: account.id })
      setSignin((s) => ({ ...s, [account.id]: res.loggedIn ? 'ok' : 'fail' }))
      setSigninDetail((d) => ({ ...d, [account.id]: res.loggedIn ? res.detail : res.message }))
    } catch (e: any) {
      setSignin((s) => ({ ...s, [account.id]: 'fail' }))
      setSigninDetail((d) => ({ ...d, [account.id]: String(e?.message ?? e) }))
    }
  }

  // Step 2 — the provider + accounts exist, walk sign-in for each.
  if (created) {
    const accounts = created.accounts ?? []
    return (
      <div className="space-y-2.5">
        <div className="rounded-lg border border-positive/30 bg-positive/10 px-3 py-2 text-[13px] text-positive">
          <Icon name="check" size={13} className="mr-1.5 inline" />Added <span className="font-medium">{created.name}</span> with {accounts.length} account{accounts.length === 1 ? '' : 's'}. Sign in to each below.
        </div>
        {accounts.map((account) => {
          const state = signin[account.id]
          return (
            <div key={account.id} className="rounded-lg border border-line bg-card-2 px-3 py-2.5">
              <div className="flex items-center justify-between gap-2">
                <div className="min-w-0">
                  <div className="truncate text-[13px] font-medium text-ink">{account.name}</div>
                  <div className="truncate font-mono text-[12px] text-ink-4">{account.cliHome || '~/.' + meta.bin}</div>
                </div>
                {state === 'ok'
                  ? <span className="flex shrink-0 items-center gap-1 text-[13px] text-positive"><Icon name="check" size={13} />Signed in{signinDetail[account.id] ? ` · ${signinDetail[account.id]}` : ''}</span>
                  : <Button variant={state === 'fail' ? 'default' : 'primary'} size="sm" icon="plug" onClick={() => signIn(account)} disabled={state === 'busy'}>{state === 'busy' ? 'Signing in…' : state === 'fail' ? 'Retry' : 'Sign in'}</Button>}
              </div>
              {state === 'fail' && <div className="mt-2 text-[12px] text-ink-3">Browser sign-in didn't complete. Run it yourself, then Retry:</div>}
              {state !== 'ok' && <div className="mt-1.5"><CopyCommand command={loginCommand(transport, created.cliPath?.trim() || meta.bin, account.cliHome ?? '')} /></div>}
            </div>
          )
        })}
        <div className="text-[12px] text-ink-3">You can sign the rest in any time from the provider's card below.</div>
      </div>
    )
  }

  // Step 1 — define the accounts to create.
  return (
    <div className="space-y-2.5">
      <div className="text-[13px] leading-snug text-ink-3">Each subscription gets its own config home so their logins don't collide. Name them, then create — you'll sign in to each next.</div>
      {rows.map((row) => (
        <div key={row.id} className="rounded-lg border border-line bg-card-2 px-3 py-2.5">
          <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-[1.1fr_1.2fr_auto]">
            <Field label="Name">
              <Input value={row.name} onChange={(e) => patchRow(row.id, { name: e.target.value })} placeholder={`${meta.label} account`} spellCheck={false} />
            </Field>
            <Field label="Config home" hint={transport === 'codex-cli' ? 'CODEX_HOME' : 'CLAUDE config dir'}>
              <Input value={row.cliHome} onChange={(e) => patchRow(row.id, { cliHome: e.target.value })} placeholder={`~/.${meta.bin}-work`} spellCheck={false} />
            </Field>
            <div className="flex items-end">
              <button onClick={() => removeRow(row.id)} disabled={rows.length <= 1} className="flex h-9 w-9 items-center justify-center rounded-lg border border-line text-ink-3 transition-colors hover:border-line-strong hover:text-danger disabled:cursor-not-allowed disabled:opacity-40" aria-label="Remove account"><Icon name="trash" size={14} /></button>
            </div>
          </div>
        </div>
      ))}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Button variant="default" size="sm" icon="plus" onClick={addRow}>Add another</Button>
        <Button variant="primary" size="sm" icon="layers" onClick={create} disabled={busy || !rows.some((r) => r.name.trim())}>{busy ? 'Creating…' : `Create ${rows.filter((r) => r.name.trim()).length} account${rows.filter((r) => r.name.trim()).length === 1 ? '' : 's'}`}</Button>
      </div>
    </div>
  )
}
