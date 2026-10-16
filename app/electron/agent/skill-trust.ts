// Project-local skill trust (VST-010). Skills that live INSIDE a repo
// (`<project>/.vontra/skills/<name>/SKILL.md`) are repo-authored content — anyone who can
// commit to the repo can write them — so they are UNTRUSTED until the user explicitly
// trusts each one, and a trusted skill whose content changes drops back to untrusted
// (hash-pinned trust, like a lockfile) until re-reviewed. Pure: hashing is injected.

export interface SkillTrustRecord {
  projectId: string
  skillId: string
  // Hash of the skill content that was reviewed. Trust applies to CONTENT, not the name.
  contentHash: string
  trustedAt: number
}

export interface SkillTrustRegistry {
  version: 1
  records: SkillTrustRecord[]
}

export type SkillTrustState = 'trusted' | 'untrusted' | 'changed'

export function emptySkillTrustRegistry(): SkillTrustRegistry {
  return { version: 1, records: [] }
}

// Defensive load: garbage in → empty registry, never a throw.
export function normalizeSkillTrustRegistry(raw: unknown): SkillTrustRegistry {
  if (!raw || typeof raw !== 'object') return emptySkillTrustRegistry()
  const records = Array.isArray((raw as { records?: unknown }).records) ? (raw as { records: unknown[] }).records : []
  return {
    version: 1,
    records: records.filter((record): record is SkillTrustRecord => {
      const r = record as Partial<SkillTrustRecord> | null
      return !!r && typeof r.projectId === 'string' && !!r.projectId
        && typeof r.skillId === 'string' && !!r.skillId
        && typeof r.contentHash === 'string' && !!r.contentHash
        && typeof r.trustedAt === 'number'
    }).map((record) => ({ ...record })),
  }
}

/**
 * Trust state for one scanned project skill: 'trusted' when the reviewed hash matches the
 * live content, 'changed' when the skill WAS trusted but its content differs now (the
 * dangerous case — a repo edit after review), 'untrusted' when never reviewed.
 */
export function evaluateSkillTrust(registry: SkillTrustRegistry, projectId: string, skillId: string, contentHash: string): SkillTrustState {
  const record = registry.records.find((candidate) => candidate.projectId === projectId && candidate.skillId === skillId)
  if (!record) return 'untrusted'
  return record.contentHash === contentHash ? 'trusted' : 'changed'
}

/** Trust the CURRENT content of a project skill (replaces any prior record). */
export function trustProjectSkill(registry: SkillTrustRegistry, input: { projectId: string; skillId: string; contentHash: string; now?: number }): SkillTrustRegistry {
  const rest = registry.records.filter((record) => !(record.projectId === input.projectId && record.skillId === input.skillId))
  return {
    version: 1,
    records: [...rest, { projectId: input.projectId, skillId: input.skillId, contentHash: input.contentHash, trustedAt: input.now ?? Date.now() }],
  }
}

/** Withdraw trust (the skill returns to untrusted regardless of content). */
export function revokeProjectSkillTrust(registry: SkillTrustRegistry, projectId: string, skillId: string): SkillTrustRegistry {
  return {
    version: 1,
    records: registry.records.filter((record) => !(record.projectId === projectId && record.skillId === skillId)),
  }
}
