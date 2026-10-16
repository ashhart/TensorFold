// Run: jiti electron/agent/skill-self-improvement.test.ts
import assert from 'node:assert/strict'
import { buildSkillReviewMessages, parseSkillReview, skillsToReview, reviewSkillsAfterRun } from './skill-self-improvement.ts'
import type { Skill } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('skill-self-improvement tests:')

function skill(over: Partial<Skill>): Skill {
  return { id: 'sk', name: 'Cloudflare setup', description: 'how to', body: 'old body', enabled: true, createdAt: 1, updatedAt: 1, ...over }
}

{
  const msgs = buildSkillReviewMessages(skill({}), 'User asked to add a DNS record; the skill omitted the zone step.')
  assert.ok(msgs.length >= 2, 'has a system + user message')
  const joined = msgs.map((m) => m.content).join('\n')
  assert.match(joined, /Cloudflare setup/, 'includes the skill name')
  assert.match(joined, /old body/, 'includes the current body')
  assert.match(joined, /zone step/, 'includes the run digest')
  assert.match(joined, /json|JSON/, 'asks for a JSON verdict')
  ok('buildSkillReviewMessages: frames the skill, the run, and a JSON contract')
}

{
  assert.equal(parseSkillReview('{"action":"keep"}').improve, false, 'keep')
  const fenced = parseSkillReview('```json\n{"action":"improve","body":"new body","rationale":"adds the zone step"}\n```')
  assert.equal(fenced.improve, true)
  assert.equal(fenced.body, 'new body')
  assert.equal(fenced.rationale, 'adds the zone step')
  assert.equal(parseSkillReview('not json at all').improve, false, 'garbage defaults to keep (safe)')
  assert.equal(parseSkillReview('{"action":"improve","body":"   "}').improve, false, 'empty body is not an improvement')
  ok('parseSkillReview: tolerant parse, defaults to keep, rejects empty body')
}

{
  const skills = [skill({ id: 'a', name: 'Cloudflare setup' }), skill({ id: 'b', name: 'Deploy' })]
  const picked = skillsToReview(skills, ['$Cloudflare Setup', 'nonexistent'])
  assert.deepEqual(picked.map((s) => s.id), ['a'], 'only viewed + existing skills, name-normalized')
  ok('skillsToReview: matches viewed names to real skills (case/$/space-insensitive)')
}

{
  // Orchestrator: drives the injected completer per reviewable skill, files improvements only.
  const skills = [skill({ id: 'a', name: 'Cloudflare setup', body: 'old body' })]
  const calls: string[] = []
  const complete = async (messages: { role: string; content: string }[]) => {
    calls.push(messages[messages.length - 1].content)
    return '{"action":"improve","body":"new better body","rationale":"adds the missing zone step"}'
  }
  const out = await reviewSkillsAfterRun({ skills, viewedNames: ['Cloudflare setup'], digest: 'd', complete, sourceThreadId: 't1' })
  assert.equal(calls.length, 1, 'reviewed the one viewed skill')
  assert.equal(out.length, 1)
  assert.equal(out[0].kind, 'skill')
  assert.equal(out[0].targetSkillId, 'a', 'targets the existing skill for in-place update')
  assert.equal(out[0].body, 'new better body')
  assert.equal(out[0].name, 'Cloudflare setup')
  assert.equal(out[0].sourceThreadId, 't1')
  ok('reviewSkillsAfterRun: turns an improve verdict into a targeted skill suggestion')
}

{
  // No proposal when the model keeps, when the body is unchanged, or when the call throws.
  const skills = [skill({ id: 'a', name: 'A', body: 'same' }), skill({ id: 'b', name: 'B', body: 'x' }), skill({ id: 'c', name: 'C', body: 'y' })]
  const complete = async (_m: { role: string; content: string }[]): Promise<string> => {
    // first → keep, second → unchanged body, third → throw
    const n = ++count
    if (n === 1) return '{"action":"keep"}'
    if (n === 2) return '{"action":"improve","body":"x"}'
    throw new Error('model down')
  }
  let count = 0
  const out = await reviewSkillsAfterRun({ skills, viewedNames: ['A', 'B', 'C'], digest: 'd', complete })
  assert.equal(out.length, 0, 'keep, no-op change, and errors all produce nothing')
  ok('reviewSkillsAfterRun: skips keeps, unchanged bodies, and completer failures')
}

{
  // Caps how many skills it reviews per run, so a skill-heavy run never fans out.
  const skills = Array.from({ length: 5 }, (_v, i) => skill({ id: `s${i}`, name: `S${i}`, body: 'b' }))
  let n = 0
  const complete = async (_m: { role: string; content: string }[]) => { n++; return '{"action":"keep"}' }
  await reviewSkillsAfterRun({ skills, viewedNames: skills.map((s) => s.name), digest: 'd', complete, max: 2 })
  assert.equal(n, 2, 'reviewed at most `max` skills')
  ok('reviewSkillsAfterRun: respects the per-run review cap')
}

console.log(`\n${passed} passed`)
