// Run: npx jiti src/lib/activity-grouping.test.ts
import assert from 'node:assert/strict'
import {
  groupToolRuns,
  activityGroupLabel,
  activityGroupStateSummary,
  reasoningDisplayState,
} from './activity-grouping'
import type { TimelineItem } from './types'

let id = 0
function tool(name: string, state?: TimelineItem['state']): TimelineItem {
  return { id: `t${id++}`, kind: 'tool', tool: name, title: name, state }
}
function item(kind: TimelineItem['kind'], text = ''): TimelineItem {
  return { id: `i${id++}`, kind, text }
}

// ── groupToolRuns ───────────────────────────────────────────────────────────

// 1. Single non-tool item passes through as a length-1 group.
{
  const out = groupToolRuns([item('user', 'hi')])
  assert.equal(out.length, 1)
  assert.equal(out[0].length, 1)
  assert.equal(out[0][0].kind, 'user')
}

// 2. Three consecutive identical tools group into one run.
{
  const out = groupToolRuns([tool('web_search'), tool('web_search'), tool('web_search')])
  assert.equal(out.length, 1)
  assert.equal(out[0].length, 3)
  assert.ok(out[0].every((i) => i.tool === 'web_search'))
}

// 3. Different tools form separate runs.
{
  const out = groupToolRuns([tool('web_search'), tool('web_search'), tool('read_file'), tool('read_file')])
  assert.equal(out.length, 2)
  assert.equal(out[0].length, 2)
  assert.equal(out[1].length, 2)
  assert.equal(out[0][0].tool, 'web_search')
  assert.equal(out[1][0].tool, 'read_file')
}

// 4. A non-tool item interrupts a run.
{
  const out = groupToolRuns([tool('search'), tool('search'), item('assistant', 'done'), tool('search')])
  assert.equal(out.length, 3)
  assert.equal(out[0].length, 2)
  assert.equal(out[1][0].kind, 'assistant')
  assert.equal(out[2].length, 1)
  assert.equal(out[2][0].tool, 'search')
}

// 5. Mixed timeline groups tools but keeps boundaries.
{
  const out = groupToolRuns([
    item('user', 'go'),
    item('reasoning', 'thinking'),
    tool('web_search'), tool('web_search'), tool('web_search'),
    item('assistant', 'here'),
    tool('read_file'), tool('read_file'),
  ])
  assert.equal(out.length, 5)
  assert.equal(out[0][0].kind, 'user')
  assert.equal(out[1][0].kind, 'reasoning')
  assert.equal(out[2].length, 3)
  assert.equal(out[3][0].kind, 'assistant')
  assert.equal(out[4].length, 2)
}

// 6. Empty timeline -> empty groups.
{
  assert.deepEqual(groupToolRuns([]), [])
}

// ── activityGroupLabel ──────────────────────────────────────────────────────

{
  assert.equal(activityGroupLabel([tool('web_search'), tool('web_search'), tool('web_search')]), 'Search the web · 3 queries')
  assert.equal(activityGroupLabel([tool('web_search')]), 'Search the web · 1 query')
  assert.equal(activityGroupLabel([tool('write_file'), tool('write_file')]), 'Create files · 2 changes')
  assert.equal(activityGroupLabel([tool('edit_file'), tool('edit_file')]), 'Edit files · 2 changes')
  assert.equal(activityGroupLabel([tool('run_command'), tool('run_command')]), 'Run commands · 2 commands')
  assert.equal(activityGroupLabel([tool('read_file'), tool('read_file')]), 'Read files · 2 files')
  assert.equal(activityGroupLabel([tool('web_fetch'), tool('web_fetch')]), 'Fetch URLs · 2 requests')
  assert.equal(activityGroupLabel([tool('list_dir'), tool('list_dir')]), 'List directories · 2 listings')
  // Unknown tool falls back to the item title + step count.
  assert.equal(activityGroupLabel([tool('mystery'), tool('mystery')]), 'mystery · 2 steps')
}

// ── activityGroupStateSummary ───────────────────────────────────────────────

// All-ok run reports no badge text (the icon implies success).
{
  const s = activityGroupStateSummary([tool('search', 'ok'), tool('search', 'ok')])
  assert.equal(s.ok, 2)
  assert.equal(s.error, 0)
  assert.equal(s.awaiting, false)
  assert.equal(s.text, '')
}

// Mixed states produce a badge and flag awaiting.
{
  const s = activityGroupStateSummary([tool('run_command', 'ok'), tool('run_command', 'ok'), tool('run_command', 'awaiting')])
  assert.equal(s.ok, 2)
  assert.equal(s.pending, 1)
  assert.equal(s.awaiting, true)
  assert.equal(s.text, '2 ok, 1 pending')
}

// A pure error run still surfaces a badge.
{
  const s = activityGroupStateSummary([tool('read_file', 'error')])
  assert.equal(s.error, 1)
  assert.equal(s.text, '1 error')
}

// Denied is counted and surfaced.
{
  const s = activityGroupStateSummary([tool('run_command', 'denied'), tool('run_command', 'ok')])
  assert.equal(s.denied, 1)
  assert.equal(s.text, '1 ok, 1 denied')
}

// ── reasoningDisplayState (Task #34) ────────────────────────────────────────

// Active + collapse-by-default -> compact indicator, no expanded text.
{
  const s = reasoningDisplayState({ active: true, forceExpanded: false })
  assert.equal(s.expanded, false)
  assert.equal(s.compact, true)
}

// Active + forceExpanded -> full text (legacy / toggle off).
{
  const s = reasoningDisplayState({ active: true, forceExpanded: true })
  assert.equal(s.expanded, true)
  assert.equal(s.compact, false)
}

// Active + forceExpanded undefined -> backward compatible (expanded).
{
  const s = reasoningDisplayState({ active: true })
  assert.equal(s.expanded, true)
  assert.equal(s.compact, false)
}

// A user click expands even when active + collapse-by-default.
{
  const s = reasoningDisplayState({ open: true, active: true, forceExpanded: false })
  assert.equal(s.expanded, true)
  assert.equal(s.compact, false)
}

// Inactive + not clicked -> header only (collapsed, no compact spinner).
{
  const s = reasoningDisplayState({ active: false })
  assert.equal(s.expanded, false)
  assert.equal(s.compact, false)
}

// Inactive + clicked open -> expanded.
{
  const s = reasoningDisplayState({ open: true, active: false })
  assert.equal(s.expanded, true)
  assert.equal(s.compact, false)
}

console.log('activity-grouping tests passed')
