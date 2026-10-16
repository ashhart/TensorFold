// Run: jiti electron/agent/skill-access.test.ts
import assert from 'node:assert/strict'
import { listSkills, parseSkillSections, renderSkillView, skillsForContext, viewSkill } from './skill-access.ts'
import type { Skill } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

const skill = (over: Partial<Skill>): Skill => ({ id: over.name || 'id', name: 'Deploy', description: 'how to deploy', body: '1. build\n2. ship', enabled: true, createdAt: 0, updatedAt: 0, ...over })

console.log('skill-access tests:')

{
  const skills = [skill({ name: 'Deploy' }), skill({ name: 'Lint', description: 'run lint' }), skill({ name: 'Off', enabled: false })]
  const list = listSkills(skills)
  assert.deepEqual(list, [{ name: 'Deploy', description: 'how to deploy' }, { name: 'Lint', description: 'run lint' }])
  assert.equal(listSkills(undefined).length, 0)
  ok('listSkills: enabled name+description only (disabled excluded)')
}

{
  const skills = [skill({ name: 'Deploy', body: 'steps' }), skill({ name: 'Off', enabled: false })]
  assert.equal(viewSkill(skills, '$deploy')?.body, 'steps', '$-prefix + case-insensitive match')
  assert.equal(viewSkill(skills, 'DEPLOY')?.name, 'Deploy')
  assert.equal(viewSkill(skills, 'Off'), null, 'disabled not viewable')
  assert.equal(viewSkill(skills, 'nope'), null)
  assert.equal(viewSkill([], 'x'), null)
  ok('viewSkill: enabled-only, $-prefix + case-insensitive, null when absent')
}

{
  const out = renderSkillView(skill({ name: 'Deploy', body: 'do the thing' }))
  assert.match(out, /<skill_body>[\s\S]*do the thing[\s\S]*<\/skill_body>/)
  assert.match(out, /security policy|doesn't conflict/i, 'framed as untrusted reference')
  ok('renderSkillView: body wrapped + untrusted framing')
}

// ---- VST-010: scoped context, trigger summaries, structured sections ----

{
  const globalSkill = { id: 'g1', name: 'lint', description: 'lint things', body: 'x', enabled: true, createdAt: 1, updatedAt: 1 } as Skill
  const projA = { id: 'pa', name: 'deploy-a', description: '', body: 'x', enabled: true, scope: 'project', projectId: 'A', createdAt: 1, updatedAt: 1 } as Skill
  const projB = { id: 'pb', name: 'deploy-b', description: '', body: 'x', enabled: true, scope: 'project', projectId: 'B', createdAt: 1, updatedAt: 1 } as Skill
  const inA = skillsForContext([globalSkill, projA, projB], 'A')
  assert.deepEqual(inA.map((s) => s.id), ['g1', 'pa'], "project B's skill never loads in project A")
  const inChat = skillsForContext([globalSkill, projA, projB], null)
  assert.deepEqual(inChat.map((s) => s.id), ['g1'], 'project skills never load outside a project')
  ok('skillsForContext: global everywhere, project-local only at home')
}

{
  const withTrigger = { id: 't1', name: 'release', description: 'Ship it safely', trigger: 'the user asks to deploy or release', body: 'x', enabled: true, createdAt: 1, updatedAt: 1 } as Skill
  const summaries = listSkills([withTrigger])
  assert.match(summaries[0].description, /Ship it safely — Use when: the user asks to deploy or release/)
  ok('listSkills: the trigger rides the summary so the agent picks well before viewing')
}

{
  const body = [
    'Do the main thing carefully.',
    '',
    '## References',
    '- docs/deploy.md',
    '',
    '## Not A Known Section',
    'stays in instructions',
    '',
    '## Tests',
    'curl /healthz expects 200',
  ].join('\n')
  const parsed = parseSkillSections(body)
  assert.match(parsed.instructions, /main thing carefully/)
  assert.match(parsed.instructions, /Not A Known Section/, 'unknown headings stay in the instruction flow')
  assert.match(parsed.sections.references ?? '', /docs\/deploy\.md/)
  assert.match(parsed.sections.tests ?? '', /healthz/)

  const rendered = renderSkillView({ id: 's', name: 'deploy', description: '', trigger: 'shipping time', body, enabled: true, createdAt: 1, updatedAt: 1 } as Skill)
  assert.match(rendered, /Trigger: shipping time/)
  assert.match(rendered, /\[references\]/)
  assert.match(rendered, /\[tests\]/)
  ok('sections: lifted from the body and rendered structurally, trigger shown')
}

console.log(`\n${passed} passed`)
