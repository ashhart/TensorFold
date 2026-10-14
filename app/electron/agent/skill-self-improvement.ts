// Aux-model skill self-improvement. After a run, a cheaper model reviews each skill the
// agent actually consulted, judged against how the run went, and proposes a sharper
// version — filed as a consent-first improvement suggestion (never auto-applied). This
// module is the pure brain: prompt, parse, gate, and a dependency-injected orchestrator.
// The live model call + persistence are wired in the runtime.
import type { Skill } from './types'
import type { SuggestionInput } from './suggestions'

export interface SkillReviewVerdict {
  improve: boolean
  description?: string
  body?: string
  rationale?: string
}

interface ReviewMessage {
  role: 'system' | 'user'
  content: string
}

const REVIEW_SYSTEM = [
  'You refine reusable skills (saved instruction snippets) for an AI coding/assistant agent.',
  'You are given one skill and a short digest of the run that just used it.',
  'Judge ONLY whether the skill\'s body would have served that run better with a concrete edit.',
  'Be conservative: most skills need no change. Improve only for a real, specific gain',
  '(a missing step, a wrong detail, a stale instruction) — never for style or padding.',
  'Reply with STRICT JSON and nothing else:',
  '{"action":"keep"} when it is already good, or',
  '{"action":"improve","body":"<full revised skill body>","description":"<optional revised one-line>","rationale":"<one line: what changed and why>"}',
  'The body must be the COMPLETE replacement text, not a diff.',
].join('\n')

export function buildSkillReviewMessages(skill: Pick<Skill, 'name' | 'description' | 'body'>, digest: string): ReviewMessage[] {
  const user = [
    `SKILL NAME: ${skill.name}`,
    `SKILL DESCRIPTION: ${skill.description || '(none)'}`,
    'CURRENT BODY:',
    '"""',
    skill.body,
    '"""',
    '',
    'RUN DIGEST (how this skill was just used and how the run went):',
    digest || '(no digest available)',
    '',
    'Return your JSON verdict now.',
  ].join('\n')
  return [
    { role: 'system', content: REVIEW_SYSTEM },
    { role: 'user', content: user },
  ]
}

// Pull the first balanced JSON object out of the reply (tolerates code fences / prose
// around it). Returns null when there is nothing parseable.
function extractJsonObject(text: string): any | null {
  const fence = text.match(/```(?:json)?\s*([\s\S]*?)```/i)
  const candidate = fence ? fence[1] : text
  const start = candidate.indexOf('{')
  if (start < 0) return null
  let depth = 0
  for (let i = start; i < candidate.length; i++) {
    const ch = candidate[i]
    if (ch === '{') depth++
    else if (ch === '}') {
      depth--
      if (depth === 0) {
        try { return JSON.parse(candidate.slice(start, i + 1)) } catch { return null }
      }
    }
  }
  return null
}

export function parseSkillReview(text: string): SkillReviewVerdict {
  const obj = extractJsonObject(text || '')
  if (!obj || obj.action !== 'improve') return { improve: false }
  const body = typeof obj.body === 'string' ? obj.body.trim() : ''
  if (!body) return { improve: false } // an "improvement" with no body is not one
  return {
    improve: true,
    body,
    description: typeof obj.description === 'string' ? obj.description.trim() : undefined,
    rationale: typeof obj.rationale === 'string' ? obj.rationale.trim() : undefined,
  }
}

function normName(name: string): string {
  return (name || '').trim().toLowerCase().replace(/^[$/]/, '').replace(/\s+/g, '-')
}

/** Skills the agent actually viewed this run (name-normalized), deduped, in skill order. */
export function skillsToReview(skills: Skill[], viewedNames: string[]): Skill[] {
  const viewed = new Set(viewedNames.map(normName).filter(Boolean))
  const seen = new Set<string>()
  const out: Skill[] = []
  for (const s of skills) {
    const key = normName(s.name)
    if (!viewed.has(key) || seen.has(s.id)) continue
    seen.add(s.id)
    out.push(s)
  }
  return out
}

const DEFAULT_MAX_REVIEWS = 3

/**
 * Drive the injected completer over each reviewable skill and return a targeted
 * improvement suggestion for every genuine, changed proposal. Pure orchestration: no IO,
 * no model knowledge — `complete` is the only side-effecting dependency, and its failures
 * are swallowed per-skill so one bad call never sinks the rest.
 */
export async function reviewSkillsAfterRun(opts: {
  skills: Skill[]
  viewedNames: string[]
  digest: string
  complete: (messages: ReviewMessage[]) => Promise<string>
  sourceThreadId?: string | null
  max?: number
}): Promise<SuggestionInput[]> {
  const targets = skillsToReview(opts.skills, opts.viewedNames).slice(0, opts.max ?? DEFAULT_MAX_REVIEWS)
  const out: SuggestionInput[] = []
  for (const skill of targets) {
    let verdict: SkillReviewVerdict
    try {
      const reply = await opts.complete(buildSkillReviewMessages(skill, opts.digest))
      verdict = parseSkillReview(reply)
    } catch {
      continue // model hiccup on one skill shouldn't block the others
    }
    if (!verdict.improve || !verdict.body) continue
    if (verdict.body.trim() === skill.body.trim()) continue // no-op edit
    out.push({
      kind: 'skill',
      name: skill.name,
      description: verdict.description ?? skill.description,
      body: verdict.body,
      rationale: verdict.rationale || 'Refined from how this skill was just used.',
      targetSkillId: skill.id,
      sourceThreadId: opts.sourceThreadId ?? null,
    })
  }
  return out
}
