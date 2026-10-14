import type { Command, Skill, SkillSuggestion } from './types'

export type SkillCurationSeverity = 'high' | 'medium' | 'low'

export type SkillCurationFindingKind =
  | 'duplicate_skill'
  | 'duplicate_command'
  | 'name_collision'
  | 'empty_skill'
  | 'empty_command'
  | 'stale_skill'
  | 'pending_improvement'

export interface SkillCurationFinding {
  kind: SkillCurationFindingKind
  severity: SkillCurationSeverity
  title: string
  evidence: string
  recommendation: string
  artifactIds: string[]
}

export interface SkillCurationReport {
  totals: {
    skills: number
    enabledSkills: number
    commands: number
    pendingSuggestions: number
  }
  findings: SkillCurationFinding[]
}

export interface SkillCurationInput {
  skills?: Skill[]
  commands?: Command[]
  pendingSuggestions?: SkillSuggestion[]
  now?: number
  staleAfterDays?: number
}

const DEFAULT_STALE_AFTER_DAYS = 90
const MIN_USEFUL_BODY_CHARS = 12

export function buildSkillCurationReport(input: SkillCurationInput = {}): SkillCurationReport {
  const skills = input.skills ?? []
  const commands = input.commands ?? []
  const pendingSuggestions = input.pendingSuggestions ?? []
  const now = Number.isFinite(input.now) ? Number(input.now) : Date.now()
  const staleAfterDays = normalizeStaleAfterDays(input.staleAfterDays)
  const staleBefore = now - staleAfterDays * 24 * 60 * 60 * 1000
  const findings: SkillCurationFinding[] = []

  for (const group of duplicateGroups(skills, (skill) => skill.name)) {
    findings.push({
      kind: 'duplicate_skill',
      severity: 'high',
      title: 'Duplicate skill name',
      evidence: `Skills named "${group.label}" share the same invocation key: ${group.items.map((skill) => skill.id).join(', ')}.`,
      recommendation: 'Merge the best instructions into one enabled skill and delete or disable the duplicates.',
      artifactIds: group.items.map((skill) => skill.id),
    })
  }

  for (const group of duplicateGroups(commands, (command) => command.name)) {
    findings.push({
      kind: 'duplicate_command',
      severity: 'high',
      title: 'Duplicate command name',
      evidence: `Commands named "/${group.label}" share the same invocation key: ${group.items.map((command) => command.id).join(', ')}.`,
      recommendation: 'Keep one slash command per invocation name, then fold any useful prompt text into it.',
      artifactIds: group.items.map((command) => command.id),
    })
  }

  const commandsByName = groupByName(commands, (command) => command.name)
  for (const [key, skillGroup] of groupByName(skills, (skill) => skill.name)) {
    const commandGroup = commandsByName.get(key)
    if (!commandGroup?.length) continue
    findings.push({
      kind: 'name_collision',
      severity: 'medium',
      title: 'Skill and command name collision',
      evidence: `Skill $${skillGroup[0].name} and command /${commandGroup[0].name} normalize to "${key}".`,
      recommendation: 'Rename one side so invocation intent is obvious in the composer.',
      artifactIds: [...skillGroup.map((skill) => skill.id), ...commandGroup.map((command) => command.id)],
    })
  }

  for (const skill of skills) {
    if (skill.body.trim().length < MIN_USEFUL_BODY_CHARS || !skill.description.trim()) {
      findings.push({
        kind: 'empty_skill',
        severity: 'high',
        title: 'Thin skill',
        evidence: `$${skill.name} has ${skill.body.trim().length} instruction characters and ${skill.description.trim() ? 'a description' : 'no description'}.`,
        recommendation: 'Either expand it into concrete repeatable guidance or delete it to reduce prompt surface noise.',
        artifactIds: [skill.id],
      })
    }
    if (skill.enabled !== false && skill.updatedAt > 0 && skill.updatedAt < staleBefore) {
      findings.push({
        kind: 'stale_skill',
        severity: 'low',
        title: 'Stale skill needs validation',
        evidence: `$${skill.name} has not been updated for at least ${staleAfterDays} days.`,
        recommendation: 'Review it against current project conventions before relying on it for new work.',
        artifactIds: [skill.id],
      })
    }
  }

  for (const command of commands) {
    if (command.prompt.trim().length < MIN_USEFUL_BODY_CHARS) {
      findings.push({
        kind: 'empty_command',
        severity: 'high',
        title: 'Empty command',
        evidence: `/${command.name} has ${command.prompt.trim().length} prompt characters.`,
        recommendation: 'Fill in the command prompt or delete the command.',
        artifactIds: [command.id],
      })
    }
  }

  for (const suggestion of pendingSuggestions.filter((item) => item.status === 'pending' && item.targetSkillId)) {
    const skill = skills.find((candidate) => candidate.id === suggestion.targetSkillId)
    findings.push({
      kind: 'pending_improvement',
      severity: 'low',
      title: 'Skill already has a pending improvement',
      evidence: skill
        ? `$${skill.name} already has a pending improvement suggestion (${suggestion.id}).`
        : `A pending improvement suggestion targets missing skill ${suggestion.targetSkillId} (${suggestion.id}).`,
      recommendation: 'Review the existing suggestion before filing another change for the same skill.',
      artifactIds: [suggestion.targetSkillId!, suggestion.id],
    })
  }

  return {
    totals: {
      skills: skills.length,
      enabledSkills: skills.filter((skill) => skill.enabled !== false).length,
      commands: commands.length,
      pendingSuggestions: pendingSuggestions.filter((suggestion) => suggestion.status === 'pending').length,
    },
    findings: findings.sort(compareFindings),
  }
}

export function formatSkillCurationReport(report: SkillCurationReport): string {
  const lines = [
    'Skill curator',
    `Artifacts: ${report.totals.enabledSkills}/${report.totals.skills} skills enabled; ${report.totals.commands} commands; ${report.totals.pendingSuggestions} pending suggestions`,
  ]
  if (!report.findings.length) return [...lines, 'No skill or command curation issues found.'].join('\n')
  lines.push(`Findings: ${report.findings.length}`)
  for (const finding of report.findings) {
    lines.push(`- [${capitalize(finding.severity)}] ${finding.title}: ${finding.evidence}`)
    lines.push(`  Recommendation: ${finding.recommendation}`)
  }
  return lines.join('\n')
}

function duplicateGroups<T>(items: T[], labelOf: (item: T) => string): Array<{ label: string; items: T[] }> {
  return [...groupByName(items, labelOf).entries()]
    .filter(([, group]) => group.length > 1)
    .map(([label, group]) => ({ label, items: group }))
}

function groupByName<T>(items: T[], labelOf: (item: T) => string): Map<string, T[]> {
  const groups = new Map<string, T[]>()
  for (const item of items) {
    const key = normalizeName(labelOf(item))
    if (!key) continue
    groups.set(key, [...(groups.get(key) ?? []), item])
  }
  return groups
}

function normalizeName(value: string): string {
  return value.trim().toLowerCase().replace(/^[$/]/, '').replace(/[\s_]+/g, '-')
}

function normalizeStaleAfterDays(value: unknown): number {
  const parsed = Math.floor(Number(value))
  if (!Number.isFinite(parsed)) return DEFAULT_STALE_AFTER_DAYS
  return Math.max(1, Math.min(3650, parsed))
}

function compareFindings(a: SkillCurationFinding, b: SkillCurationFinding): number {
  const severityRank: Record<SkillCurationSeverity, number> = { high: 0, medium: 1, low: 2 }
  return severityRank[a.severity] - severityRank[b.severity] || a.title.localeCompare(b.title)
}

function capitalize(value: string): string {
  return value.slice(0, 1).toUpperCase() + value.slice(1)
}
