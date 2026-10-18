import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { authoredText } from '../../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../../lib/store'
import { useState } from 'react'
import clsx from 'clsx'
import { Toggle } from '../ui'
import { Section, SettingsPane, type SettingsSectionContext } from './shared'
import type { PersonaProfile } from '../../lib/types'

// One-click voices — they just fill the persona body for the selected scope.
const PRESETS: { name: string; body: string }[] = [
  { name: 'Concise', body: 'You are a concise, no-nonsense assistant. Lead with the answer, no preamble or filler. Short sentences, minimal markdown, bottom line first.' },
  { name: 'Warm', body: 'You are a warm, encouraging assistant. Friendly and human with a little personality, but always clear and genuinely useful. Briefly acknowledge wins.' },
  { name: 'Technical', body: 'You are a precise technical expert. Exact, dense, and correct. Use proper terminology, cite specifics, and call out trade-offs and edge cases.' },
  { name: 'Playful', body: 'You are a playful, witty assistant. Light humour and the occasional aside are welcome — never at the expense of being clear and genuinely helpful.' },
]

// Scopes resolve most-specific-first: connector > assistant > workspace > global.
const SCOPES: { value: string; label: string }[] = [
  { value: 'global', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Everywhere") } },
  { value: 'assistant', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Assistant") } },
  { value: 'connector:imessage', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "iMessage") } },
  { value: 'connector:whatsapp', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "WhatsApp") } },
  { value: 'connector:slack', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Slack") } },
  { value: 'connector:discord', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Discord") } },
  { value: 'connector:email', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Email") } },
]

const TEXTAREA = 'w-full resize-y rounded-lg border border-line bg-card px-3 py-2 text-[13px] leading-relaxed text-ink outline-none transition-colors focus:border-line-strong'

export function PersonaSettings({ ctx }: { ctx: SettingsSectionContext }) {
  const tx = useAuthoredText()
  const { draft, setDraft, commit } = ctx
  const [scope, setScope] = useState('global')
  const personas: PersonaProfile[] = draft.personas ?? []
  const current = personas.find((p) => p.scope === scope) ?? null

  // Edit live via setDraft; persist on blur/toggle via commit (matches the other panels).
  const writePersonas = (next: PersonaProfile[], persist = false) => {
    const d = { ...draft, personas: next }
    if (persist) commit(d); else setDraft(d)
  }
  const upsert = (patch: Partial<PersonaProfile>, persist = false) => {
    const now = Date.now()
    const exists = personas.some((p) => p.scope === scope)
    const next = exists
      ? personas.map((p) => (p.scope === scope ? { ...p, ...patch, updatedAt: now } : p))
      : [...personas, { id: scope, scope, name: SCOPES.find((s) => s.value === scope)?.label ?? scope, body: '', enabled: true, updatedAt: now, ...patch }]
    writePersonas(next, persist)
  }
  const clearScope = () => writePersonas(personas.filter((p) => p.scope !== scope), true)

  return (
    <SettingsPane>
      <Section title={tx("Voice")} desc={tx("The assistant's voice and personality. Edits apply to the very next message — no restart. Give a different voice per surface or connector; the most specific one wins.")}>
        {/* Scope picker */}
        <div className="mb-2 flex flex-wrap gap-1.5">
          {SCOPES.map((s) => {
            const set = personas.some((p) => p.scope === s.value && p.enabled && p.body.trim())
            return (
              <button
                key={s.value}
                onClick={() => setScope(s.value)}
                className={clsx('rounded-md px-2.5 py-1 text-[13px] font-medium transition-colors', scope === s.value ? 'bg-card text-ink shadow-card' : 'text-ink-3 hover:bg-ink-5 hover:text-ink')}
              >
                {s.label}{set ? <span className="ml-1 inline-block h-1.5 w-1.5 rounded-full bg-positive align-middle" /> : null}
              </button>
            )
          })}
        </div>

        {/* Presets */}
        <div className="mb-2 flex flex-wrap items-center gap-1.5">
          <span className="mr-1 text-[12px] text-ink-3">{tx("Start from:")}</span>
          {PRESETS.map((p) => (
            <button
              key={p.name}
              onClick={() => upsert({ body: p.body, enabled: true }, true)}
              className="rounded-md border border-line px-2 py-0.5 text-[13px] text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink"
            >
              {p.name}
            </button>
          ))}
        </div>

        <textarea
          value={current?.body ?? ''}
          onChange={(e) => upsert({ body: e.target.value })}
          onBlur={() => commit(draft)}
          rows={6}
          placeholder={tx("How should the assistant sound for \"{value0}\"? e.g. warm and brief, technical and exact, playful…", { value0: SCOPES.find((s) => s.value === scope)?.label })}
          className={TEXTAREA}
        />
        <div className="mt-2 flex items-center justify-between">
          <Toggle checked={current ? current.enabled : false} onChange={(enabled) => upsert({ enabled }, true)} label={tx("Use this persona")} />
          {current?.body.trim() ? (
            <button onClick={clearScope} className="text-[13px] text-ink-3 transition-colors hover:text-danger">{tx("Clear")}</button>
          ) : null}
        </div>
      </Section>

      <Section title={tx("Your profile")}>
        <p className="mb-1.5 text-[13px] leading-relaxed text-ink-3">
          {tx("A short note about you the assistant keeps in mind across conversations — your name, role, and how you like replies. Treated as reference only, never run as instructions.")}</p>
        <textarea
          value={draft.userProfile ?? ''}
          onChange={(e) => setDraft({ ...draft, userProfile: e.target.value })}
          onBlur={() => commit(draft)}
          rows={4}
          placeholder={tx("e.g. Founder. Ships fast, hates fluff. Prefer the bottom line first, then bullets.")}
          className={TEXTAREA}
        />
      </Section>
    </SettingsPane>
  )
}
