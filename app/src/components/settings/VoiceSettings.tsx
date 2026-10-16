import { useState } from 'react'
import { Icon } from '../../lib/icons'
import { api } from '../../lib/ipc'
import { Field, Input, Toggle } from '../ui'
import { AdvancedSection, Section, SettingsPane, type SettingsSectionContext } from './shared'
import type { LocalTtsDiscovery, VoiceProvider } from '../../lib/types'
import { KokoroVoiceCard } from './KokoroVoiceCard'

export function VoiceSettings({ ctx }: { ctx: SettingsSectionContext }) {
  const { draft, setDraft, commit, voiceNames, commitVoice } = ctx

  // Local TTS engine discovery (reachability + installable expressive voices). Kept in
  // component state — it's a point-of-use probe, not persisted settings.
  const [localTtsStatus, setLocalTtsStatus] = useState<LocalTtsDiscovery | null>(null)
  const [localTtsBusy, setLocalTtsBusy] = useState(false)
  const localTtsVoices = localTtsStatus?.voices ?? []

  const checkLocalTts = async () => {
    setLocalTtsBusy(true)
    try {
      const result = await api.discoverLocalTts(draft.voice.localTtsBaseUrl || undefined)
      setLocalTtsStatus(result)
    } catch (e: any) {
      setLocalTtsStatus({ found: false, baseUrl: draft.voice.localTtsBaseUrl || '', voices: [], message: e?.message ?? 'Could not reach the local TTS engine.' })
    } finally {
      setLocalTtsBusy(false)
    }
  }

  return (
<SettingsPane>
        <Section title="Voice" desc="Local-first speech input and fast readback for long chat, code, and room output.">
          <div className="space-y-3">
            <div className="rounded-xl border border-line bg-card p-3.5 shadow-card">
              <div className="mb-3 flex items-start justify-between gap-3">
                <div className="flex min-w-0 items-start gap-2.5">
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-ink-5 text-ink-2"><Icon name="speaker" size={17} /></span>
                  <div className="min-w-0">
                    <div className="text-[13px] font-medium text-ink">Speech bridge</div>
                    <div className="mt-0.5 text-[13px] leading-snug text-ink-2">Use your voice in the composer and read long model output aloud at your preferred speed.</div>
                  </div>
                </div>
                <Toggle label="Speech bridge" checked={draft.voice.enabled} onChange={(enabled) => commitVoice({ ...draft.voice, enabled })} />
              </div>

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <Field label="Text-to-speech">
                  <div className="space-y-2">
                    <div className="flex items-center gap-3 rounded-lg border border-line bg-card-2 px-3 py-2.5">
                      <span className="flex-1 text-[13px] leading-snug text-ink-2">Read assistant replies, briefs, and room status aloud.</span>
                      <Toggle label="Text-to-speech" checked={draft.voice.ttsEnabled} onChange={(ttsEnabled) => commitVoice({ ...draft.voice, ttsEnabled })} />
                    </div>
                    <select aria-label="Speech playback provider" value={draft.voice.ttsProvider} onChange={(event) => commitVoice({ ...draft.voice, ttsProvider: event.target.value as VoiceProvider })} className="no-drag h-9 w-full cursor-pointer rounded-lg border border-line bg-card px-3 text-[13px] text-ink outline-none focus:border-accent focus:ring-2 focus:ring-accent/15">
                      <option value="browser">System voice</option><option value="kokoro">Kokoro · offline voices</option><option value="local">External local engine</option>
                      {['openai', 'custom', 'elevenlabs'].includes(draft.voice.ttsProvider) && <option value={draft.voice.ttsProvider}>Cloud voice disabled</option>}
                    </select>
                  </div>
                </Field>

                <Field label="Speech-to-text">
                  <div className="space-y-2">
                    <div className="flex items-center gap-3 rounded-lg border border-line bg-card-2 px-3 py-2.5">
                      <span className="flex-1 text-[13px] leading-snug text-ink-2">Dictate into the composer, then send or guide it.</span>
                      <Toggle label="Speech-to-text" checked={draft.voice.sttEnabled} onChange={(sttEnabled) => commitVoice({ ...draft.voice, sttEnabled })} />
                    </div>
                    <select aria-label="Dictation provider" value={draft.voice.sttProvider} onChange={(event) => commitVoice({ ...draft.voice, sttProvider: event.target.value as VoiceProvider })} className="no-drag h-9 w-full cursor-pointer rounded-lg border border-line bg-card px-3 text-[13px] text-ink outline-none focus:border-accent focus:ring-2 focus:ring-accent/15">
                      <option value="browser">System dictation, when available</option>
                      {draft.voice.sttProvider !== 'browser' && <option value={draft.voice.sttProvider}>Unavailable transcription provider</option>}
                    </select>
                  </div>
                </Field>

                <Field label="Playback speed" hint={`${draft.voice.rate.toFixed(1)}x`}>
                  <input
                    type="range" min={0.7} max={2.5} step={0.1} value={draft.voice.rate}
                    onChange={(e) => commitVoice({ ...draft.voice, rate: parseFloat(e.target.value) })}
                    className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-ink-5 accent-[var(--accent)]"
                  />
                  <div className="mt-1 flex justify-between text-[12px] text-ink-4"><span>0.7x</span><span>fast readback</span><span>2.5x</span></div>
                </Field>

                {draft.voice.ttsProvider === 'browser' && <Field label="System voice">
                  <select
                    value={draft.voice.voiceName}
                    onChange={(e) => commitVoice({ ...draft.voice, voiceName: e.target.value })}
                    className="no-drag h-9 w-full cursor-pointer rounded-lg border border-line bg-card px-3 text-[13px] text-ink outline-none transition-colors focus:border-accent focus:ring-2 focus:ring-accent/15"
                  >
                    <option value="">System default</option>
                    {voiceNames.map((name) => <option key={name} value={name}>{name}</option>)}
                  </select>
                </Field>}

                {draft.voice.ttsProvider === 'kokoro' && <KokoroVoiceCard settings={draft.voice} onChange={commitVoice} />}

                <Field label="Brief before reading" hint={draft.voice.autoSummarizeBeforeRead ? 'On' : 'Off'}>
                  <div className="flex items-center gap-3 rounded-lg border border-line bg-card-2 px-3 py-2.5">
                    <span className="flex-1 text-[13px] leading-snug text-ink-2">Compress long replies before reading so huge outputs do not become a wall of audio.</span>
                    <Toggle label="Brief before reading" checked={draft.voice.autoSummarizeBeforeRead} onChange={(autoSummarizeBeforeRead) => commitVoice({ ...draft.voice, autoSummarizeBeforeRead })} />
                  </div>
                </Field>

                {draft.voice.autoSummarizeBeforeRead && (
                  <AdvancedSection title="Brief length" icon="sliders">
                    <Field label="Brief length" hint={`${draft.voice.summaryMaxChars} chars`}>
                      <input
                        type="range" min={300} max={2000} step={100} value={draft.voice.summaryMaxChars}
                        onChange={(e) => commitVoice({ ...draft.voice, summaryMaxChars: parseInt(e.target.value, 10) })}
                        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-ink-5 accent-[var(--accent)]"
                      />
                    </Field>
                  </AdvancedSection>
                )}

                <Field label="Room updates" hint={draft.voice.autoReadRoomUpdates ? 'Auto' : 'Manual'}>
                  <div className="flex items-center gap-3 rounded-lg border border-line bg-card-2 px-3 py-2.5">
                    <span className="flex-1 text-[13px] leading-snug text-ink-2">Automatically read room asks, errors, and final updates. Manual Tell me is always available.</span>
                    <Toggle label="Read room updates automatically" checked={draft.voice.autoReadRoomUpdates} onChange={(autoReadRoomUpdates) => commitVoice({ ...draft.voice, autoReadRoomUpdates })} />
                  </div>
                </Field>

                <Field label="Volume" hint={`${Math.round(draft.voice.volume * 100)}%`}>
                  <input
                    type="range" min={0} max={1} step={0.05} value={draft.voice.volume}
                    onChange={(e) => commitVoice({ ...draft.voice, volume: parseFloat(e.target.value) })}
                    className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-ink-5 accent-[var(--accent)]"
                  />
                </Field>

                {draft.voice.ttsProvider === 'local' && (
                  <div className="sm:col-span-2 space-y-2.5 rounded-lg border border-line bg-card-2 p-3">
                    <div className="flex items-start gap-2.5">
                      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-ink-5 text-ink-2"><Icon name="cpu" size={15} /></span>
                      <div className="min-w-0">
                        <div className="text-[13px] font-medium text-ink">Local expressive engine</div>
                        <div className="mt-0.5 text-[12px] leading-snug text-ink-2">Connect a local speech server you already run. For a voice TensorFold downloads and manages for you, choose Kokoro above.</div>
                      </div>
                    </div>
                    <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                      <Field label="Local TTS endpoint">
                        <Input value={draft.voice.localTtsBaseUrl} onChange={(e) => setDraft((d) => ({ ...d, voice: { ...d.voice, localTtsBaseUrl: e.target.value } }))} onBlur={() => commit(draft)} placeholder="http://127.0.0.1:8080" spellCheck={false} />
                      </Field>
                      <Field label="Voice" hint={localTtsVoices.length ? `${localTtsVoices.length} found` : 'optional'}>
                        {localTtsVoices.length ? (
                          <select value={draft.voice.localTtsVoice} onChange={(e) => commitVoice({ ...draft.voice, localTtsVoice: e.target.value })} className="no-drag h-9 w-full cursor-pointer rounded-lg border border-line bg-card px-3 text-[13px] text-ink outline-none transition-colors focus:border-accent focus:ring-2 focus:ring-accent/15">
                            <option value="">Engine default</option>
                            {localTtsVoices.map((name) => <option key={name} value={name}>{name}</option>)}
                          </select>
                        ) : (
                          <Input value={draft.voice.localTtsVoice} onChange={(e) => setDraft((d) => ({ ...d, voice: { ...d.voice, localTtsVoice: e.target.value } }))} onBlur={() => commit(draft)} placeholder="af_heart" spellCheck={false} />
                        )}
                      </Field>
                      <Field label="Emotion / style" hint="optional">
                        <Input value={draft.voice.localTtsEmotion} onChange={(e) => setDraft((d) => ({ ...d, voice: { ...d.voice, localTtsEmotion: e.target.value } }))} onBlur={() => commit(draft)} placeholder="cheerful, calm, excited…" spellCheck={false} />
                      </Field>
                      <Field label="Engine status" hint={localTtsStatus?.found ? 'reachable' : localTtsBusy ? 'checking' : 'unknown'}>
                        <button
                          type="button"
                          onClick={checkLocalTts}
                          disabled={localTtsBusy}
                          className="no-drag inline-flex h-9 w-full items-center justify-center gap-1.5 rounded-lg border border-line bg-card px-3 text-[13px] font-medium text-ink-2 shadow-card transition-colors hover:bg-ink-5 hover:text-ink disabled:opacity-50"
                        >
                          <Icon name="refresh" size={13} className={localTtsBusy ? 'animate-spin' : undefined} />
                          {localTtsBusy ? 'Checking…' : 'Check / find voices'}
                        </button>
                      </Field>
                    </div>
                    {localTtsStatus && (
                      <div className={`flex items-center gap-2 rounded-md border px-2.5 py-2 text-[12px] ${localTtsStatus.found ? 'border-line bg-card text-ink-2' : 'border-line bg-card text-ink-2'}`}>
                        <Icon name={localTtsStatus.found ? 'check' : 'warning'} size={13} /><span>{localTtsStatus.message}</span>
                      </div>
                    )}
                    <div className="rounded-md border border-line bg-card px-2.5 py-2 text-[12px] leading-snug text-ink-2">
                      No engine yet? Install a local OpenAI-compatible TTS such as <span className="font-mono text-ink-2">Kokoro-FastAPI</span> (expressive multi-voice) or <span className="font-mono text-ink-2">Piper</span>, start it on the endpoint above, then Check.
                    </div>
                  </div>
                )}

                {(['openai', 'custom', 'elevenlabs'].includes(draft.voice.ttsProvider) || draft.voice.sttProvider !== 'browser') && <p className="sm:col-span-2 text-[13px] leading-relaxed text-ink-2">This voice provider is unavailable in this build, choose System, Kokoro or an external local engine above.</p>}
              </div>
            </div>
          </div>
        </Section>
              </SettingsPane>

  )
}
