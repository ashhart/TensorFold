// Run: ./node_modules/.bin/jiti electron/agent/agent-skills.test.ts
import assert from 'node:assert/strict'
import { mkdtemp, mkdir, realpath, rm, symlink, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { tmpdir } from 'node:os'
import { DEFAULT_AGENT_SKILL_MAX_BYTES, scanAgentSkills } from './agent-skills'

async function main() {
  const base = await mkdtemp(path.join(tmpdir(), 'vontra-agent-skills-'))
  const external = await mkdtemp(path.join(tmpdir(), 'vontra-agent-skills-external-'))

  try {
    const alphaRoot = path.join(base, 'alpha')
    await mkdir(path.join(alphaRoot, 'scripts'), { recursive: true })
    await mkdir(path.join(alphaRoot, 'references'))
    await mkdir(path.join(alphaRoot, 'assets'))
    await writeFile(path.join(alphaRoot, 'SKILL.md'), [
      '---',
      'name: Alpha Runner',
      'description: Runs alpha tasks safely.',
      '---',
      '# Ignored title',
      '',
      'Ignored fallback paragraph.',
    ].join('\n'))

    const fallbackRoot = path.join(base, 'fallback-skill')
    await mkdir(fallbackRoot)
    await writeFile(path.join(fallbackRoot, 'SKILL.md'), [
      '# Fallback Skill',
      '',
      'Uses markdown title and the first paragraph when front matter is absent.',
      '',
      'Second paragraph is ignored.',
    ].join('\n'))

    await mkdir(path.join(base, 'not-a-skill'))
    await writeFile(path.join(base, 'loose.md'), '# Not a skill')

    const externalSkill = path.join(external, 'escape')
    await mkdir(externalSkill)
    await writeFile(path.join(externalSkill, 'SKILL.md'), '---\nname: Escaped\n---\n')
    await symlink(externalSkill, path.join(base, 'escaped-link'), 'dir')

    const escapedSkillFile = path.join(external, 'escaped-SKILL.md')
    await writeFile(escapedSkillFile, '---\nname: Escaped File\n---\n')
    const fileEscapeRoot = path.join(base, 'file-escape')
    await mkdir(fileEscapeRoot)
    await symlink(escapedSkillFile, path.join(fileEscapeRoot, 'SKILL.md'))

    const largeRoot = path.join(base, 'too-large')
    await mkdir(largeRoot)
    await writeFile(path.join(largeRoot, 'SKILL.md'), 'x'.repeat(DEFAULT_AGENT_SKILL_MAX_BYTES + 1))

    const emptyRoot = path.join(base, 'empty')
    await mkdir(emptyRoot)
    await writeFile(path.join(emptyRoot, 'SKILL.md'), '')

    const malformedRoot = path.join(base, 'malformed')
    await mkdir(malformedRoot)
    await writeFile(path.join(malformedRoot, 'SKILL.md'), '---\nname: Unclosed front matter\n# Markdown Still Works\n\nBody fallback.')

    const skills = await scanAgentSkills(base)
    const ids = skills.map((skill) => skill.id).sort()

    assert.deepEqual(ids, ['alpha', 'empty', 'fallback-skill', 'malformed'])
    assert.ok(!ids.includes('not-a-skill'))
    assert.ok(!ids.includes('escaped-link'))
    assert.ok(!ids.includes('file-escape'))
    assert.ok(!ids.includes('too-large'))

    const alpha = skills.find((skill) => skill.id === 'alpha')
    assert.ok(alpha)
    assert.equal(alpha.name, 'Alpha Runner')
    assert.equal(alpha.description, 'Runs alpha tasks safely.')
    assert.equal(alpha.rootPath, await realpath(alphaRoot))
    assert.equal(alpha.skillPath, await realpath(path.join(alphaRoot, 'SKILL.md')))
    assert.equal(alpha.hasScripts, true)
    assert.equal(alpha.hasReferences, true)
    assert.equal(alpha.hasAssets, true)
    assert.equal(typeof alpha.updatedAt, 'number')
    assert.ok(alpha.updatedAt > 0)

    const fallback = skills.find((skill) => skill.id === 'fallback-skill')
    assert.ok(fallback)
    assert.equal(fallback.name, 'Fallback Skill')
    assert.equal(fallback.description, 'Uses markdown title and the first paragraph when front matter is absent.')
    assert.equal(fallback.hasScripts, false)
    assert.equal(fallback.hasReferences, false)
    assert.equal(fallback.hasAssets, false)

    const empty = skills.find((skill) => skill.id === 'empty')
    assert.ok(empty)
    assert.equal(empty.name, 'Empty')
    assert.equal(empty.description, '')

    const malformed = skills.find((skill) => skill.id === 'malformed')
    assert.ok(malformed)
    assert.equal(malformed.name, 'Markdown Still Works')
    assert.equal(malformed.description, 'Body fallback.')
  } finally {
    await rm(base, { recursive: true, force: true })
    await rm(external, { recursive: true, force: true })
  }

  console.log('agent-skills tests passed')
}

void main()
