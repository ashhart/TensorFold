import { createHash } from 'node:crypto'
import type { ContentBlock } from './types'

export interface SystemPromptSections {
  base: string
  persona?: string
  goal?: string
  capabilities?: string
}

export interface SystemPromptSnapshot {
  sections: Required<SystemPromptSections>
  fingerprint: string
  createdAt: number
  updatedAt: number
}

export interface SystemPromptMessage {
  role: string
  content: string | ContentBlock[] | null
}

export function buildSystemPromptSnapshot(sections: SystemPromptSections, now = Date.now()): SystemPromptSnapshot {
  const normalized = normalizeSystemPromptSections(sections)
  return {
    sections: normalized,
    fingerprint: systemPromptFingerprint(normalized),
    createdAt: now,
    updatedAt: now,
  }
}

export function refreshSystemPromptSnapshot(
  current: SystemPromptSnapshot | null | undefined,
  sections: SystemPromptSections,
  now = Date.now(),
): SystemPromptSnapshot {
  const normalized = normalizeSystemPromptSections(sections)
  const fingerprint = systemPromptFingerprint(normalized)
  if (current?.fingerprint === fingerprint) return current
  return {
    sections: normalized,
    fingerprint,
    createdAt: now,
    updatedAt: now,
  }
}

export function renderSystemPromptSnapshot(snapshot: Pick<SystemPromptSnapshot, 'sections'>): string {
  return renderSystemPromptSections(snapshot.sections)
}

export function renderSystemPromptSections(sections: SystemPromptSections): string {
  const normalized = normalizeSystemPromptSections(sections)
  return [
    normalized.persona,
    normalized.base,
    normalized.goal,
    normalized.capabilities,
  ].filter(Boolean).join('\n\n')
}

export function systemPromptFingerprint(sections: SystemPromptSections): string {
  const normalized = normalizeSystemPromptSections(sections)
  const payload = JSON.stringify([
    ['persona', normalized.persona],
    ['base', normalized.base],
    ['goal', normalized.goal],
    ['capabilities', normalized.capabilities],
  ])
  return createHash('sha256').update(payload).digest('hex')
}

export function messagesWithSystemPromptSnapshot<T extends SystemPromptMessage>(
  messages: T[],
  snapshot: SystemPromptSnapshot,
): T[] {
  const content = renderSystemPromptSnapshot(snapshot)
  const systemIndex = messages.findIndex((message) => message.role === 'system')
  if (systemIndex < 0) return [{ role: 'system', content } as T, ...messages]
  return messages.map((message, index) => index === systemIndex ? { ...message, content } : message)
}

export function normalizeSystemPromptSections(sections: SystemPromptSections): Required<SystemPromptSections> {
  return {
    persona: normalizeSection(sections.persona),
    base: normalizeSection(sections.base),
    goal: normalizeSection(sections.goal),
    capabilities: normalizeSection(sections.capabilities),
  }
}

function normalizeSection(value: string | null | undefined): string {
  return String(value ?? '').replace(/\r\n?/g, '\n').trim()
}
