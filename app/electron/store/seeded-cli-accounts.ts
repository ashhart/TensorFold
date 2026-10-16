import { existsSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import type { Provider } from '../agent/types'

// Earlier builds seeded Personal/Work Codex and Claude accounts whose config homes
// (~/.codex-personal, ~/.claude-work, …) no user had, so a signed-in CLI read as signed
// out. Drop only those exact seeds, and only while their home folder still doesn't
// exist: an account the user actually authorized has a home and is kept.
const SEEDED_ACCOUNTS: Record<string, string> = {
  'codex-personal': '~/.codex-personal',
  'codex-work': '~/.codex-work',
  'claude-personal': '~/.claude-personal',
  'claude-work': '~/.claude-work',
}

function expandHome(p: string, home: string): string {
  return p === '~' ? home : p.startsWith('~/') ? path.join(home, p.slice(2)) : p
}

export interface SeededAccountPrune { providers: Provider[]; removedIds: string[] }

/** Pure apart from `exists`; returns the providers without unused seeded accounts. */
export function dropUnusedSeededCliAccounts(
  providers: Provider[],
  exists: (p: string) => boolean = existsSync,
  home = os.homedir(),
): SeededAccountPrune {
  const removedIds: string[] = []
  const next = providers.map((provider) => {
    if (provider.transport !== 'codex-cli' && provider.transport !== 'claude-cli') return provider
    const accounts = provider.accounts ?? []
    const kept = accounts.filter((account) => {
      const seededHome = SEEDED_ACCOUNTS[account.id]
      const unused = !!seededHome && account.cliHome === seededHome && !exists(expandHome(seededHome, home))
      if (unused) removedIds.push(account.id)
      return !unused
    })
    if (kept.length === accounts.length) return provider
    const defaultAccountId = provider.defaultAccountId && removedIds.includes(provider.defaultAccountId)
      ? (kept[0]?.id ?? null)
      : provider.defaultAccountId
    return { ...provider, accounts: kept, defaultAccountId }
  })
  return { providers: next, removedIds }
}
