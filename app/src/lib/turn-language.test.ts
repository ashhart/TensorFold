import assert from 'node:assert/strict'
import { configureUiLanguageReader } from './preferences/current-locale'
import { authoredText } from './preferences/language-packs'
import { workRowLabel, stepLabel, type TurnSummary } from './turn-blocks'
import { turnStatsSegments, turnStatsTitle, cortheonSummary } from './turn-stats-line'
const original = 'Settings / Cancel / 日本語.txt'
const turn: TurnSummary = { key: 'synthetic', index: 1, work: [{ id: 'read', kind: 'tool', tool: 'read_file', title: 'Original diagnostic', path: original, state: 'running', createdAt: 0 }], steps: 3, startedAt: 0, endedAt: 12000, answerId: null, edits: [], running: true }
let locale = 'zh-Hans'
const restore = configureUiLanguageReader(() => locale)
try {
  const live = workRowLabel(turn, { now: 5000 })
  assert.ok(live.includes(authoredText(locale, 'Working')))
  assert.doesNotMatch(live, /Working|Reading|Worked|steps/)
  assert.ok(stepLabel(turn.work[0]).includes('日本語.txt'), 'Filenames remain original data')
  const finished = workRowLabel({ ...turn, running: false }, { now: 12000 })
  assert.ok(finished.includes(authoredText(locale, 'Steps: {count}', { count: 3 })))
  assert.doesNotMatch(finished, /Worked|steps/)
  const unknown = { ...turn.work[0], tool: 'custom_tool', title: original }
  assert.equal(stepLabel(unknown), original, 'Unknown tool title is not translated as UI text')
  const stats = { model: 'synthetic', providerId: 'local', modelCalls: 1, toolCalls: 2, firstTokenMs: 450, decodeTokensPerSecond: 20, promptTokens: 1200, cachedTokens: 1000, completionTokens: 10, decodeSeconds: 0.5 }
  const parts = turnStatsSegments(stats, original)
  assert.equal(parts[0], original, 'Model titles remain original data')
  assert.ok(parts.includes(authoredText(locale, 'Tool calls: {count}', { count: 2 })))
  assert.doesNotMatch(parts.join(' '), /tool calls|first token|decode/)
  assert.ok(turnStatsTitle(stats)?.includes(authoredText(locale, 'Prompt: {count} tokens', { count: 1200 })))
  assert.ok(cortheonSummary({ ...stats, cortheon: { searches: 2, pagesRead: 1, sources: [] } } as any)?.includes(authoredText(locale, 'Searches: {count}', { count: 2 })))
  locale = 'pl'
  assert.ok(workRowLabel({ ...turn, running: false }, { now: 12000 }).includes(authoredText(locale, 'Steps: {count}', { count: 3 })), 'Pure formatters read the current profile after switching')
  locale = 'en'
  assert.equal(workRowLabel({ ...turn, running: false }, { now: 12000 }), 'Worked for 12s · 3 steps')
  assert.deepEqual(turnStatsSegments({ model: 'm', providerId: 'local', modelCalls: 1, toolCalls: 1 }, original), [original, '1 tool call'])
} finally { restore() }
console.log('Actual turn work, live tool labels, measured stats and titles follow the profile language while preserving raw data and English compatibility')
