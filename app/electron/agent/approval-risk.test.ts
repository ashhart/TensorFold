// Run: jiti electron/agent/approval-risk.test.ts
import assert from 'node:assert/strict'
import { fullAccessAutoAllows, riskBandForClassification, shouldAutoAllowInFullAuto, type RiskBand } from './approval-risk'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('approval-risk tests:')

// --- command class -> risk band matrix ---
{
  const cases: Array<[string, RiskBand]> = [
    ['read-only', 'low'],
    ['test', 'low'],
    ['build', 'low'],
    ['empty-command', 'low'],
    ['git-read', 'medium'],
    ['network-fetch', 'medium'],
    ['shell', 'medium'],
    ['git-mutate', 'high'],
    ['network-shell', 'high'],
    ['package-install', 'high'],
    ['compound-shell', 'high'],
    ['destructive-filesystem', 'critical'],
    ['secret-read', 'critical'],
    ['system-control', 'critical'],
  ]
  for (const [klass, expected] of cases) {
    const { band } = riskBandForClassification(klass, 'command')
    assert.equal(band, expected, `command ${klass} -> ${expected}, got ${band}`)
  }
  // Unknown command class defaults to medium (not low).
  assert.equal(riskBandForClassification('unknown', 'command').band, 'medium', 'unknown command -> medium')
  ok('command class -> risk band matrix')
}

// --- connector class -> risk band matrix ---
{
  const cases: Array<[string, RiskBand]> = [
    ['read', 'low'],
    ['draft', 'low'],
    ['approval-resolution', 'medium'],
    ['calendar-create', 'medium'],
    ['calendar-update', 'high'],
    ['reply', 'high'],
    ['send', 'high'],
    ['browser-submit', 'high'],
    ['delete', 'critical'],
    ['account-permission-change', 'critical'],
    ['payment-purchase', 'critical'],
  ]
  for (const [klass, expected] of cases) {
    const { band } = riskBandForClassification(klass, 'connector')
    assert.equal(band, expected, `connector ${klass} -> ${expected}, got ${band}`)
  }
  // Unknown connector class defaults to high (fail closed — external state).
  assert.equal(riskBandForClassification('unknown', 'connector').band, 'high', 'unknown connector -> high')
  ok('connector class -> risk band matrix')
}

// --- generic tool classification ---
{
  // File edits (write_file/edit_file/apply_patch) -> medium (git-reversible).
  assert.equal(riskBandForClassification('write_file', 'tool').band, 'medium', 'write_file -> medium')
  assert.equal(riskBandForClassification('edit_file', 'tool').band, 'medium', 'edit_file -> medium')
  // MCP tools and unknown -> high (opaque).
  assert.equal(riskBandForClassification('mcp__evil__do_thing', 'tool').band, 'high', 'mcp tool -> high')
  assert.equal(riskBandForClassification('unknown', 'tool').band, 'high', 'unknown tool -> high')
  ok('generic tool risk classification')
}

// --- reason is human-readable and includes the class ---
{
  const { reason } = riskBandForClassification('package-install', 'command')
  assert.ok(reason.includes('high risk'), 'reason includes band')
  assert.ok(reason.includes('package install'), 'reason includes class')
  ok('reason is human-readable')
}

// --- null/empty classification handled safely ---
{
  assert.equal(riskBandForClassification(null, 'command').band, 'medium', 'null command -> medium')
  assert.equal(riskBandForClassification('', 'command').band, 'medium', 'empty command -> medium')
  assert.equal(riskBandForClassification(undefined, 'connector').band, 'high', 'undefined connector -> high')
  ok('null/empty classification handled safely')
}

// --- shouldAutoAllowInFullAuto: default ceiling 'low' ---
{
  // Default: only low auto-allows.
  assert.equal(shouldAutoAllowInFullAuto('low', 'low', false), true, 'low always auto-allows')
  assert.equal(shouldAutoAllowInFullAuto('medium', 'low', false), false, 'medium blocked at low ceiling without rule')
  assert.equal(shouldAutoAllowInFullAuto('medium', 'low', true), false, 'medium blocked at low ceiling even with rule')
  assert.equal(shouldAutoAllowInFullAuto('high', 'low', false), false, 'high never auto-allows')
  assert.equal(shouldAutoAllowInFullAuto('critical', 'low', false), false, 'critical never auto-allows')
  ok('default ceiling low: only low auto-allows')
}

// --- shouldAutoAllowInFullAuto: ceiling raised to 'medium' ---
{
  // Medium ceiling: low still auto-allows; medium auto-allows ONLY with a saved rule.
  assert.equal(shouldAutoAllowInFullAuto('low', 'medium', false), true, 'low auto-allows regardless')
  assert.equal(shouldAutoAllowInFullAuto('medium', 'medium', false), false, 'medium needs saved rule even at medium ceiling')
  assert.equal(shouldAutoAllowInFullAuto('medium', 'medium', true), true, 'medium auto-allows with saved rule at medium ceiling')
  assert.equal(shouldAutoAllowInFullAuto('high', 'medium', true), false, 'high blocked even at medium ceiling')
  assert.equal(shouldAutoAllowInFullAuto('critical', 'medium', true), false, 'critical never auto-allows')
  ok('ceiling medium: medium needs saved rule, high/critical blocked')
}

// --- shouldAutoAllowInFullAuto: high/critical never auto-allow regardless of ceiling/rule ---
{
  // Even raising the ceiling to high (not exposed in UI, but the function must hold).
  assert.equal(shouldAutoAllowInFullAuto('high', 'high', true), false, 'high never auto-allows')
  assert.equal(shouldAutoAllowInFullAuto('critical', 'critical', true), false, 'critical never auto-allows')
  ok('high/critical never auto-allow (hard floor)')
}

{
  assert.equal(fullAccessAutoAllows('command', 'run_command', 'high'), true, 'commands are pre-screened by their approval reason')
  assert.equal(fullAccessAutoAllows('tool', 'write_file', 'medium'), true)
  assert.equal(fullAccessAutoAllows('tool', 'web_fetch', 'medium'), true)
  assert.equal(fullAccessAutoAllows('connector', 'email_compose', 'low'), true, 'drafts and reads')
  assert.equal(fullAccessAutoAllows('connector', 'email_compose', 'high'), false, 'sends still ask')
  assert.equal(fullAccessAutoAllows('tool', 'mcp__github__create_issue', 'high'), false)
  assert.equal(fullAccessAutoAllows('tool', 'controller_action', 'high'), false)
  assert.equal(fullAccessAutoAllows('tool', 'browser_automation', 'high'), false)
  ok('Full access runs sandboxed tools unasked; outbound connectors, MCP and desktop/browser control ask')
}

console.log(`\n${passed} checks passed.`)
console.log('ALL TESTS PASSED')
