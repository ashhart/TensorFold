// Generated thread recap (VST-040): a deterministic "what happened in this thread" summary
// built from the timeline — files touched, commands run, the last outcome — so a returning
// user (or a handoff target) gets the gist without rereading the whole transcript. Pure: no
// model call, so it's instant and testable; a model-written prose recap can layer on top.

import type { TimelineItem } from './types'

export interface ThreadRecap {
  turns: number
  filesChanged: string[]
  filesRead: string[]
  commands: number
  lastError: string | null
  // A one-line headline (the most recent assistant sentence, trimmed).
  headline: string | null
  // The full rendered recap text (markdown), or '' when nothing has happened yet.
  text: string
}

const EDIT_TOOLS = new Set(['write_file', 'edit_file', 'apply_patch', 'str_replace', 'create_file'])
const READ_TOOLS = new Set(['read_file', 'view_file', 'cat'])

function firstSentence(text: string, max = 140): string {
  const clean = text.replace(/```[\s\S]*?```/g, '').replace(/\s+/g, ' ').trim()
  if (!clean) return ''
  const end = clean.search(/[.!?](\s|$)/)
  const sentence = end >= 0 ? clean.slice(0, end + 1) : clean
  return sentence.length > max ? `${sentence.slice(0, max - 1)}…` : sentence
}

// Build a recap from a thread's timeline. Counts user turns, distinct edited/read files
// (by the tool's `path`), commands, and the last error; the headline is the newest
// assistant message's first sentence.
export function buildThreadRecap(timeline: TimelineItem[]): ThreadRecap {
  let turns = 0
  let commands = 0
  let lastError: string | null = null
  let headline: string | null = null
  const changed = new Set<string>()
  const read = new Set<string>()

  for (const item of timeline) {
    if (item.kind === 'user') turns++
    else if (item.kind === 'error' && item.text) lastError = firstSentence(item.text, 180)
    else if (item.kind === 'assistant' && item.text) {
      const sentence = firstSentence(item.text)
      if (sentence) headline = sentence // keep the LAST one (freshest)
    } else if (item.kind === 'tool') {
      const tool = item.tool ?? ''
      if (item.command) commands++
      if (item.path) {
        if (EDIT_TOOLS.has(tool)) changed.add(item.path)
        else if (READ_TOOLS.has(tool)) read.add(item.path)
      }
    }
  }

  const filesChanged = [...changed].sort()
  const filesRead = [...read].sort()
  const parts: string[] = []
  if (turns > 0) parts.push(`${turns} turn${turns === 1 ? '' : 's'}`)
  if (filesChanged.length) parts.push(`${filesChanged.length} file${filesChanged.length === 1 ? '' : 's'} changed`)
  if (commands) parts.push(`${commands} command${commands === 1 ? '' : 's'} run`)

  const lines: string[] = []
  if (headline) lines.push(headline)
  if (parts.length) lines.push(`_${parts.join(' · ')}_`)
  if (filesChanged.length) lines.push(`**Changed:** ${filesChanged.slice(0, 8).join(', ')}${filesChanged.length > 8 ? `, +${filesChanged.length - 8} more` : ''}`)
  if (lastError) lines.push(`**Last error:** ${lastError}`)

  return {
    turns,
    filesChanged,
    filesRead,
    commands,
    lastError,
    headline,
    text: lines.join('\n\n'),
  }
}
