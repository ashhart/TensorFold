import { useEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import { SETTINGS_SECTION_REQUEST_EVENT, useStore, consumeSettingsSection } from '../lib/store'
import { registerUpdatePreparationBlocker } from '../lib/update-safety'
import { createSettingsDraftPersistence, type SettingsDraftState } from '../lib/settings-draft'
import { Icon } from '../lib/icons'
import { Button } from './ui'
import { ThemePanel } from './ThemePanel'
import { SkillsCommands } from './SkillsCommands'
import type { Settings as TSettings } from '../lib/types'
import { ALL_SETTINGS_SECTIONS, groupOfSection, SETTINGS_GROUPS, Section, SettingsPane, type SettingsSection } from './settings/shared'
import { RuntimeSettings } from './settings/RuntimeSettings'
import { OrchestratorSettings } from './settings/OrchestratorSettings'
import { VoiceSettings } from './settings/VoiceSettings'
import { ModelsSettings } from './settings/ModelsSettings'
import { SecuritySettings } from './settings/SecuritySettings'
import { ConnectorsSettings } from './settings/ConnectorsSettings'
import { PersonaSettings } from './settings/PersonaSettings'
import { ReadinessSettings } from './settings/ReadinessSettings'
import { useProviderSettings } from './settings/useProviderSettings'
import { useConnectorSources } from './settings/useConnectorSources'
import { useBotConnectors } from './settings/useBotConnectors'
import { useRemoteSecurity } from './settings/useRemoteSecurity'
import { AppSettings } from './settings/AppSettings'
import { ModelFoldersSettings } from './settings/ModelFoldersSettings'
import { PhoneSettings } from './settings/PhoneSettings'

function consumeRequestedSettingsSection(): SettingsSection | null {
  const requested = consumeSettingsSection()
  return requested && ALL_SETTINGS_SECTIONS.some((section) => section.id === requested) ? requested as SettingsSection : null
}

export function Settings({ onClose, initialSection, onDraftDirtyChange }: { onClose?: () => void; initialSection?: SettingsSection; onDraftDirtyChange?: (dirty: boolean) => void } = {}) {
  const settings = useStore((s) => s.settings)!
  const save = useStore((s) => s.saveSettings)
  const purgeLocalMemory = useStore((s) => s.purgeLocalMemory)
  const setView = useStore((s) => s.setView)
  const updateStatus = useStore((s) => s.updateStatus)
  const checkForUpdates = useStore((s) => s.checkForUpdates)
  const openUpdateDownload = useStore((s) => s.openUpdateDownload)
  const [draft, setDraft] = useState<TSettings>(settings)
  // Land on the requested section if a surface (e.g. Assistant Sources) asked for
  // one, else the prop, else Models. Consumed once so it doesn't stick.
  const [activeSection, setActiveSection] = useState<SettingsSection>(() => {
    return consumeRequestedSettingsSection() ?? initialSection ?? 'models'
  })
  useEffect(() => {
    const onRequestedSection = () => {
      const requested = consumeRequestedSettingsSection()
      if (requested) setActiveSection(requested)
    }
    window.addEventListener(SETTINGS_SECTION_REQUEST_EVENT, onRequestedSection)
    return () => window.removeEventListener(SETTINGS_SECTION_REQUEST_EVENT, onRequestedSection)
  }, [])
  useEffect(() => {
    if (initialSection) setActiveSection(initialSection)
  }, [initialSection])
  // Which sidebar groups are expanded. The active section's group is always
  // open; the rest persist across launches via localStorage so the nav
  // remembers the user's browsing shape.
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(() => {
    const saved = (() => { try { return localStorage.getItem('vontra.settingsGroups') } catch { return null } })()
    const initial = new Set<string>(['ai']) // AI & Models open by default
    const activeGroup = groupOfSection(activeSection)
    if (activeGroup) initial.add(activeGroup.id)
    if (saved) { try { for (const id of JSON.parse(saved) as string[]) initial.add(id) } catch { /* ignore */ } }
    return initial
  })
  const toggleGroup = (id: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id); else next.add(id)
      try { localStorage.setItem('vontra.settingsGroups', JSON.stringify([...next])) } catch { /* ignore */ }
      return next
    })
  }
  // Expanding a section's group when it becomes active (e.g. deep-link, or the
  // agent opens Settings) keeps the active item visible.
  useEffect(() => {
    const g = groupOfSection(activeSection)
    if (g && !expandedGroups.has(g.id)) setExpandedGroups((prev) => new Set(prev).add(g.id))
  }, [activeSection, expandedGroups])
  const close = () => {
    if (dirtyRef.current) {
      useStore.getState().setNotice({ kind: 'info', message: 'Save or discard your settings changes before closing.' })
      return
    }
    if (onClose) onClose(); else setView({ name: 'home' })
  }

  // True while the local draft holds edits the user has not committed yet. A save can land from
  // a background path (agent change_setting, connector status push, a Connect that calls
  // commitProviders) while the modal is open; saveSettings replaces the store settings with a
  // fresh redacted object, so an unconditional resync would wipe in-progress typing — including
  // re-typed secret fields that come back masked. Skip the resync while edits are pending.
  const dirtyRef = useRef(false)
  const [draftState, setDraftState] = useState<SettingsDraftState>({ dirty: false, pending: false, failed: false })
  const draftPersistence = useRef<ReturnType<typeof createSettingsDraftPersistence<TSettings>> | null>(null)
  if (!draftPersistence.current) draftPersistence.current = createSettingsDraftPersistence<TSettings>({
    dirty: dirtyRef,
    setDraft,
    save: async (next) => { await save(next); return useStore.getState().settings ?? next },
    changed: (state) => { setDraftState(state); onDraftDirtyChange?.(state.dirty) },
  })
  useEffect(() => registerUpdatePreparationBlocker(() => dirtyRef.current ? 'Save or discard your settings draft before installing and restarting.' : null), [])
  useEffect(() => {
    if (dirtyRef.current) return
    setDraft(settings)
  }, [settings])

  // Latest-draft ref so long-lived effect handlers (e.g. the WhatsApp status push) can
  // persist against the user's current edits instead of a stale closure captured at subscribe time.
  const draftRef = useRef(draft)
  draftRef.current = draft

  // Uncommitted edit: mark the draft dirty so an external save can't clobber it.
  const editDraft: typeof setDraft = (update) => { draftPersistence.current!.edited(); setDraft(update) }
  // The store already reports persistence failures; consumed rejections keep automatic
  // control changes from becoming unhandled errors while the draft remains retryable.
  const commit = (next: TSettings) => { void draftPersistence.current!.persist(next).catch(() => {}) }
  const persistDraft = (next: TSettings) => draftPersistence.current!.persist(next)
  const patch = (p: Partial<TSettings>) => commit({ ...draft, ...p })

  const providerApi = useProviderSettings({ draft, editDraft, setDraft, commit, patch, save: persistDraft })
  const connectorApi = useConnectorSources({ draft, editDraft, setDraft, commit, patch })
  const botApi = useBotConnectors({ draft, editDraft, setDraft, commit, patch, draftRef }, connectorApi.flashConnectorStatus)
  const remoteSecurityApi = useRemoteSecurity({ draft, editDraft, setDraft, commit, patch }, purgeLocalMemory)

  const [voiceNames, setVoiceNames] = useState<string[]>([])
  const commitVoice = (voice: TSettings['voice']) => commit({ ...draft, voice })
  useEffect(() => {
    if (!('speechSynthesis' in window)) return
    const load = () => setVoiceNames(window.speechSynthesis.getVoices().map((voice) => voice.name).filter(Boolean))
    load()
    window.speechSynthesis.onvoiceschanged = load
    return () => { window.speechSynthesis.onvoiceschanged = null }
  }, [])

  const activeMeta = ALL_SETTINGS_SECTIONS.find((section) => section.id === activeSection) ?? ALL_SETTINGS_SECTIONS[0]
  const sectionCtx = {
    draft,
    setDraft: editDraft,
    goToSection: setActiveSection,
    patch,
    commit,
    ...providerApi,
    ...connectorApi,
    ...botApi,
    ...remoteSecurityApi,
    updateStatus,
    checkForUpdates,
    openUpdateDownload,
    voiceNames,
    commitVoice,
  }

  // Full-page takeover: the settings nav swaps into the sidebar
  // position on the frosted shell, and the sections render on the content card.
  return (
    <div className="flex h-full w-full overflow-hidden">
      <aside className="hidden w-[240px] shrink-0 flex-col sm:flex">
        <div className="drag flex h-[52px] shrink-0 items-end px-3 pb-1">
          <div className="h-1 w-[58px] shrink-0" aria-hidden="true" />
        </div>
        <div className="px-2 pb-1">
          <button
            onClick={close}
            className="no-drag flex h-8 w-full items-center gap-2.5 rounded-lg px-2.5 text-[13px] text-ink/85 transition-colors hover:bg-surface-hover hover:text-ink"
          >
            <Icon name="back" size={14} className="text-ink-2" />
            Back to app
          </button>
        </div>
        <nav className="scroll-thin min-h-0 flex-1 overflow-y-auto px-2 pb-2">
          {SETTINGS_GROUPS.map((group) => {
            const isOpen = expandedGroups.has(group.id)
            const hasActive = group.sections.some((s) => s.id === activeSection)
            return (
              <div key={group.id}>
                <button
                  onClick={() => toggleGroup(group.id)}
                  className={clsx('mt-2 flex w-full items-center gap-1.5 rounded-lg px-2.5 py-1 text-left text-[13px] font-normal transition-colors hover:text-ink', hasActive ? 'text-ink-2' : 'text-ink-3')}
                >
                  <span className="min-w-0 flex-1 truncate">{group.label}</span>
                  <Icon name="chevronDown" size={12} className={clsx('shrink-0 text-ink-4 transition-transform', isOpen ? '' : '-rotate-90')} />
                </button>
                {isOpen && (
                  <div className="mt-0.5 space-y-px">
                    {group.sections.map((section) => (
                      <button
                        key={section.id}
                        onClick={() => setActiveSection(section.id)}
                        className={clsx('flex h-8 w-full items-center gap-2.5 rounded-lg px-2.5 text-left transition-colors', activeSection === section.id ? 'bg-surface-hover text-ink' : 'text-ink/80 hover:bg-surface-hover hover:text-ink')}
                      >
                        <Icon name={section.icon} size={15} className="shrink-0 text-ink-2" />
                        <span className="min-w-0 flex-1 truncate text-[13px] font-normal">{section.label}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </nav>
        <div className="px-4 py-3 text-[12px] leading-relaxed text-ink-4">
          Settings and conversations are stored locally on your machine.
        </div>
      </aside>
      <section className="content-card relative flex min-w-0 flex-1 flex-col">
        <div className="drag flex min-h-11 shrink-0 items-center justify-between gap-3 px-3 py-1">
          <label className="no-drag flex min-w-0 flex-1 items-center sm:hidden">
            <span className="sr-only">Settings section</span>
            <select aria-label="Settings section" value={activeSection} onChange={(event) => setActiveSection(event.target.value as SettingsSection)} className="h-9 w-full min-w-0 rounded-lg border border-line bg-card px-2 text-[13px] text-ink outline-none focus:border-accent">
              {SETTINGS_GROUPS.map((group) => <optgroup key={group.id} label={group.label}>{group.sections.map((section) => <option key={section.id} value={section.id}>{section.label}</option>)}</optgroup>)}
            </select>
          </label>
          <Button variant="ghost" size="sm" icon="x" onClick={close} className="no-drag sm:ml-auto">Close</Button>
        </div>
        {draftState.dirty && <div role="status" className="no-drag mx-6 mb-4 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-line bg-card px-4 py-3 text-[13px]">
          <span className={draftState.failed ? 'text-danger' : 'text-ink-2'}>{draftState.pending ? 'Saving changes…' : draftState.failed ? 'Changes were not saved. Retry or discard this draft.' : 'You have unsaved changes.'}</span>
          <div className="flex gap-2">
            <Button size="sm" disabled={draftState.pending} onClick={() => draftPersistence.current!.discard(useStore.getState().settings ?? settings)}>Discard draft</Button>
            <Button size="sm" variant="primary" disabled={draftState.pending} onClick={() => commit(draftRef.current)}>{draftState.failed ? 'Retry save' : 'Save draft'}</Button>
          </div>
        </div>}
        <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-4 pb-8 sm:px-8">
          <div className="mx-auto w-full max-w-[760px]">
            <div className="pb-5">
              <h1 className="text-[22px] font-semibold tracking-[-0.02em] text-ink">{activeMeta.label}</h1>
              <p className="mt-1 text-[13px] text-ink-2">{activeMeta.desc}</p>
            </div>
            {activeSection === 'appearance' && (
              <SettingsPane>
                <ThemePanel />
              </SettingsPane>
            )}

            {activeSection === 'app' && <AppSettings blocked={draftState.dirty} />}
            {activeSection === 'library' && <ModelFoldersSettings />}
            {activeSection === 'phone' && <PhoneSettings />}

            {activeSection === 'agents' && (
              <SettingsPane>
                <Section title="Your bots" desc="Save each bot’s instructions, model and permissions in one place; the same bots can fill room roles.">
                  <Button icon="spark" disabled={draftState.dirty} onClick={() => setView({ name: 'bots' })}>Open bots</Button>
                  {draftState.dirty && <p className="mt-2 text-[13px] text-ink-3">Save or discard your settings draft first.</p>}
                </Section>
              </SettingsPane>
            )}

            {activeSection === 'agent' && <RuntimeSettings ctx={sectionCtx} />}

            {activeSection === 'orchestrator' && <OrchestratorSettings ctx={sectionCtx} />}

            {activeSection === 'voice' && <VoiceSettings ctx={sectionCtx} />}

            {activeSection === 'readiness' && <ReadinessSettings ctx={sectionCtx} />}

            {activeSection === 'models' && <ModelsSettings ctx={sectionCtx} />}

            {activeSection === 'security' && <SecuritySettings ctx={sectionCtx} />}

            {activeSection === 'connectors' && <ConnectorsSettings ctx={sectionCtx} />}
            {activeSection === 'persona' && <PersonaSettings ctx={sectionCtx} />}

            {activeSection === 'skills' && (
              <SettingsPane>
        <Section title="Skills & commands" desc="Reusable instructions and saved prompts. Invoke them in the composer with $ for skills and / for commands.">
          <SkillsCommands />
        </Section>
              </SettingsPane>
            )}
          </div>
        </div>
      </section>
    </div>
  )
}
