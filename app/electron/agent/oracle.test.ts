// Run: jiti electron/agent/oracle.test.ts
import assert from 'node:assert/strict'
import { parseOracleArgs, buildOracleUserMessage, buildOracleSystemPrompt } from './oracle.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('oracle tests:')

// ---- parseOracleArgs ----
{
  assert.deepEqual(parseOracleArgs({ question: 'why does X deadlock?' }), { question: 'why does X deadlock?' })
  // `query` alias accepted
  assert.deepEqual(parseOracleArgs({ query: 'pick a cache strategy' }), { question: 'pick a cache strategy' })
  // context carried when present + trimmed
  assert.deepEqual(
    parseOracleArgs({ question: ' q ', context: '  some code  ' }),
    { question: 'q', context: 'some code' },
  )
  // empty / missing question → null (no consult)
  assert.equal(parseOracleArgs({}), null)
  assert.equal(parseOracleArgs({ question: '   ' }), null)
  assert.equal(parseOracleArgs({ context: 'only context' }), null)
  // blank context is dropped (not carried as '')
  assert.deepEqual(parseOracleArgs({ question: 'q', context: '   ' }), { question: 'q' })
  ok('parseOracleArgs: question/query alias, trims, drops blank context, null without a question')
}

// ---- buildOracleUserMessage ----
{
  assert.equal(buildOracleUserMessage({ question: 'q' }), 'q')
  assert.equal(
    buildOracleUserMessage({ question: 'q', context: 'CTX' }),
    'q\n\nRelevant context:\nCTX',
  )
  ok('buildOracleUserMessage folds context into the user turn')
}

// ---- buildOracleSystemPrompt ----
{
  const p = buildOracleSystemPrompt('/work/repo')
  assert.ok(p.includes('/work/repo'), 'mentions the workspace')
  assert.ok(/read-only/i.test(p), 'states read-only')
  assert.ok(/Oracle/.test(p), 'identifies as the Oracle')
  assert.ok(/cannot edit files/i.test(p), 'forbids writes')
  ok('buildOracleSystemPrompt: workspace, read-only grounding, no-write')
}

console.log(`\n${passed} checks passed.`)
