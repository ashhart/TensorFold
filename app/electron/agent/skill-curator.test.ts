import assert from 'node:assert/strict'
import { buildToolSchemas, executeTool, READ_ONLY_TOOLS, TOOL_SCHEMAS } from './tools.ts'
import { buildSkillCurationReport, formatSkillCurationReport } from './skill-curator'
import type { Command, Skill, SkillSuggestion } from './types'

const now = Date.UTC(2026, 5, 27)
const old = now - 120 * 24 * 60 * 60 * 1000

const skills: Skill[] = [
  { id: 's1', name: 'Release Flow', description: 'Ship a release.', body: 'Run tests, build, package, and summarize evidence.', enabled: true, createdAt: old, updatedAt: old },
  { id: 's2', name: 'release-flow', description: 'Duplicate release instructions.', body: 'Duplicate body.', enabled: true, createdAt: now, updatedAt: now },
  { id: 's3', name: 'Tiny', description: '', body: 'todo', enabled: true, createdAt: old, updatedAt: old },
  { id: 's4', name: 'Old Disabled', description: 'Legacy.', body: 'Legacy body.', enabled: false, createdAt: old, updatedAt: old },
]

const commands: Command[] = [
  { id: 'c1', name: 'release-flow', description: 'Run release.', prompt: 'Do release work.', createdAt: now, updatedAt: now },
  { id: 'c2', name: 'empty-command', description: '', prompt: ' ', createdAt: now, updatedAt: now },
]

const pendingSuggestions: SkillSuggestion[] = [
  {
    id: 'sg1',
    kind: 'skill',
    name: 'Tiny',
    description: 'Better tiny skill.',
    body: 'A complete replacement.',
    rationale: 'Already queued.',
    targetSkillId: 's3',
    status: 'pending',
    createdAt: now,
    updatedAt: now,
  },
]

const report = buildSkillCurationReport({ skills, commands, pendingSuggestions, now, staleAfterDays: 90 })
assert.equal(report.totals.skills, 4)
assert.equal(report.totals.commands, 2)
assert.equal(report.totals.pendingSuggestions, 1)
assert.ok(report.findings.some((finding) => finding.kind === 'duplicate_skill' && finding.artifactIds.includes('s1') && finding.artifactIds.includes('s2')))
assert.ok(report.findings.some((finding) => finding.kind === 'name_collision' && finding.artifactIds.includes('s1') && finding.artifactIds.includes('c1')))
assert.ok(report.findings.some((finding) => finding.kind === 'empty_command' && finding.artifactIds.includes('c2')))
assert.ok(report.findings.some((finding) => finding.kind === 'stale_skill' && finding.artifactIds.includes('s1')))
assert.ok(report.findings.some((finding) => finding.kind === 'pending_improvement' && finding.artifactIds.includes('s3')))
assert.match(formatSkillCurationReport(report), /Skill curator/)
assert.match(formatSkillCurationReport(report), /Duplicate skill name/)
assert.match(formatSkillCurationReport(report), /already has a pending improvement/)
console.log('ok skill curator reports duplicate, stale, empty, and pending-improvement artifacts')

assert.ok(READ_ONLY_TOOLS.has('skill_curator'))
assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'skill_curator'))
assert.ok(buildToolSchemas().some((tool: any) => tool.function?.name === 'skill_curator'))

const toolResult = await executeTool('skill_curator', { stale_after_days: 90 }, {
  cwd: '/repo',
  skills,
  commands,
  skillSuggestions: pendingSuggestions,
})
assert.equal(toolResult.ok, true)
assert.equal(toolResult.title, 'Skill curator')
assert.match(toolResult.output, /Duplicate skill name/)
assert.match(toolResult.output, /empty-command/)
console.log('ok skill_curator is exposed as a read-only tool')
