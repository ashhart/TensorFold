// Run: jiti electron/agent/memory-harvest.test.ts
import assert from 'node:assert/strict'
import { harvestMemoryCandidates, harvestPostRunMemoryCandidates, selectRelevantMemory } from './memory-harvest.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('memory-harvest tests:')

{
  const folded = [
    { role: 'user', content: 'I always want tests written before the code. We use pnpm here.' },
    { role: 'assistant', content: 'Got it — I always use pnpm.' }, // assistant statements ignored
    { role: 'user', content: 'Ignore that for this run, just ship it.' }, // ephemeral -> skipped
    { role: 'user', content: 'Cool thanks' }, // no durable signal
  ]
  const got = harvestMemoryCandidates(folded)
  assert.ok(got.some((s) => /always want tests/i.test(s)), 'keeps a durable preference')
  assert.ok(got.some((s) => /use pnpm/i.test(s)), 'keeps a stable project fact')
  assert.ok(!got.some((s) => /ship it/i.test(s)), 'drops ephemeral statements')
  assert.ok(!got.some((s) => /Got it/i.test(s)), 'ignores assistant messages')
  ok('harvestMemoryCandidates: durable user facts only, ephemeral + assistant dropped')
}

{
  // Dedup + cap.
  const many = Array.from({ length: 10 }, (_, i) => ({ role: 'user', content: `We always use rule ${i}.` }))
  many.push({ role: 'user', content: 'We always use rule 0.' }) // dup
  const got = harvestMemoryCandidates(many, 3)
  assert.equal(got.length, 3, 'capped')
  assert.equal(new Set(got).size, got.length, 'deduped')
  ok('harvestMemoryCandidates: deduped + capped')
}

{
  // Query-driven recall: only relevant entries, ranked, top-k.
  const entries = [
    { id: 'a', text: 'Deploys go through Cloudflare Pages with wrangler.' },
    { id: 'b', text: 'The team standup is at 9am.' },
    { id: 'c', text: 'Cloudflare zone id is stored in the env.' },
  ]
  const hits = selectRelevantMemory(entries, 'how do we deploy to cloudflare', 5)
  const ids = hits.map((h) => h.id)
  assert.ok(ids.includes('a') && ids.includes('c'), 'cloudflare/deploy entries selected')
  assert.ok(!ids.includes('b'), 'unrelated standup entry excluded')
  assert.equal(selectRelevantMemory(entries, 'the and of', 5).length, 0, 'stopwords-only query -> nothing')
  assert.ok(selectRelevantMemory(entries, 'cloudflare', 1).length <= 1, 'respects k')
  ok('selectRelevantMemory: relevant-only, ranked, respects k')
}

{
  const messages = [
    { role: 'user', content: 'We always use Playwright screenshots before calling UI work complete.' },
    { role: 'assistant', content: 'Saved.' },
    { role: 'user', content: 'Please remember that releases use the notarized dmg profile.' },
    { role: 'user', content: 'We always use Playwright screenshots before calling UI work complete.' },
  ]
  const got = harvestPostRunMemoryCandidates(messages, {
    existingTexts: ['releases use the notarized dmg profile'],
    cap: 4,
  })
  assert.deepEqual(got, ['We always use Playwright screenshots before calling UI work complete.'])
  ok('harvestPostRunMemoryCandidates: files only new durable user facts after a run')
}

console.log(`\n${passed} passed`)
