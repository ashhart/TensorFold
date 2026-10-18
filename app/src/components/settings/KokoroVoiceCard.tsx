import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useRef, useState } from 'react'
import { api } from '../../lib/ipc'
import { Icon } from '../../lib/icons'
import { speakText, stopSpeaking } from '../../lib/voice'
import type { KokoroStatus, VoiceSettings } from '../../lib/types'
import { Field } from '../ui'

export function KokoroVoiceCard({ settings, onChange }: { settings: VoiceSettings; onChange: (settings: VoiceSettings) => void }) {
  const tx = useAuthoredText()
  const [status, setStatus] = useState<KokoroStatus | null>(null)
  const [working, setWorking] = useState(false)
  const [removing, setRemoving] = useState(false)
  const [previewing, setPreviewing] = useState(false)
  const [notice, setNotice] = useState('')
  const mounted = useRef(true)
  const installing = status?.state === 'downloading' || status?.state === 'verifying'
  const downloadBusy = !removing && (working || installing)
  const progress = status?.totalBytes ? Math.min(100, Math.round(status.downloadedBytes / status.totalBytes * 100)) : 0

  useEffect(() => {
    mounted.current = true
    void api.getKokoroStatus().then((next) => { if (mounted.current) setStatus(next) }).catch(() => { if (mounted.current) setNotice(tx("Could not check Kokoro. Reopen Voice settings to retry.")) })
    return () => { mounted.current = false; stopSpeaking() }
  }, [])

  useEffect(() => {
    if (!downloadBusy) return
    const timer = setInterval(() => {
      void api.getKokoroStatus().then((next) => { if (mounted.current) setStatus(next) }).catch(() => {})
    }, 700)
    return () => clearInterval(timer)
  }, [downloadBusy])

  async function manage(action: 'install' | 'cancel' | 'remove') {
    setNotice(''); setWorking(true); setRemoving(action === 'remove'); stopSpeaking(); setPreviewing(false)
    try {
      const next = await (action === 'install' ? api.installKokoro() : action === 'cancel' ? api.cancelKokoroInstall() : api.removeKokoro())
      if (mounted.current) { setStatus(next); if (next.error) setNotice(next.error) }
    } catch { if (mounted.current) setNotice(tx("Could not update Kokoro. Retry in a moment.")) }
    finally { if (mounted.current) { setWorking(false); setRemoving(false) } }
  }

  function preview() {
    if (previewing) { stopSpeaking(); setPreviewing(false); return }
    setNotice(''); setPreviewing(true)
    const started = speakText('Hello, I am your TensorFold voice. Your replies stay on this device, even when I read them aloud.', { ...settings, autoSummarizeBeforeRead: false }, {
      onEnd: () => { if (mounted.current) setPreviewing(false) },
      onError: (message) => { if (mounted.current) { setPreviewing(false); setNotice(message) } },
    })
    if (!started) setPreviewing(false)
  }

  return <div className="sm:col-span-2 space-y-3 rounded-xl border border-line bg-card-2 p-3.5">
    <div className="flex items-start gap-2.5">
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-ink-5 text-ink-2"><Icon name="speaker" size={17} /></span>
      <div className="min-w-0 flex-1">
        <div className="text-[13px] font-medium text-ink">{tx("Kokoro offline voices")}</div>
        <p className="mt-1 text-[13px] leading-relaxed text-ink-2">{tx("Download the 92 MB speech model once, then read replies locally with nine American English voices.")}</p>
        <p className="mt-1 text-[12px] leading-relaxed text-ink-2">{tx("No API key or server setup. Speech text stays on your device.")}</p>
      </div>
      {status?.installed && <span className="shrink-0 rounded-full border border-line bg-card px-2 py-1 text-[12px] text-ink-2">{tx("Offline ready")}</span>}
    </div>
    <div role="status" aria-live="polite" className="text-[13px] text-ink-2">{removing ? tx("Removing Kokoro...") : status?.message || tx("Checking Kokoro...")}</div>
    {downloadBusy && <div className="space-y-1.5">
      <progress aria-label={tx("Kokoro model download")} value={status?.downloadedBytes || 0} max={status?.totalBytes || 92364770} className="h-2 w-full accent-[var(--accent)]" />
      <div className="flex justify-between gap-2 text-[12px] text-ink-2"><span>{status?.state === 'verifying' ? tx("Checking model integrity") : tx("Downloading model")}</span><span>{progress}%</span></div>
    </div>}
    <div className="flex flex-wrap gap-2">
      {downloadBusy ? <button type="button" onClick={() => void manage('cancel')} className="no-drag rounded-lg border border-line bg-card px-3 py-2 text-[13px] font-medium text-ink-2 hover:bg-ink-5">{tx("Cancel download")}</button>
        : !status?.installed && <button type="button" disabled={!status} onClick={() => void manage('install')} className="no-drag inline-flex items-center gap-1.5 rounded-lg bg-accent px-3 py-2 text-[13px] font-medium text-white hover:opacity-90 disabled:opacity-50"><Icon name="download" size={14} />{status?.state === 'error' ? tx("Retry download") : tx("Download Kokoro")}</button>}
      {status?.installed && !downloadBusy && <>
        <button type="button" disabled={working || !settings.enabled || !settings.ttsEnabled} onClick={preview} className="no-drag inline-flex items-center gap-1.5 rounded-lg border border-line bg-card px-3 py-2 text-[13px] font-medium text-ink-2 hover:bg-ink-5 disabled:opacity-50"><Icon name={previewing ? 'stop' : 'speaker'} size={14} />{previewing ? tx("Stop preview") : tx("Preview voice")}</button>
        <button type="button" disabled={working} onClick={() => void manage('remove')} className="no-drag rounded-lg border border-line bg-card px-3 py-2 text-[13px] text-ink-2 hover:bg-ink-5 hover:text-ink disabled:opacity-50">{removing ? tx("Removing...") : tx("Remove download")}</button>
      </>}
    </div>
    <Field label={tx("Kokoro voice")} hint={tx("English")}>
      <select aria-label={tx("Kokoro voice")} value={settings.kokoroVoice || 'af_heart'} disabled={!status?.installed || downloadBusy || removing} onChange={(event) => { stopSpeaking(); setPreviewing(false); onChange({ ...settings, kokoroVoice: event.target.value }) }} className="no-drag h-9 w-full cursor-pointer rounded-lg border border-line bg-card px-3 text-[13px] text-ink outline-none focus:border-accent focus:ring-2 focus:ring-accent/15 disabled:opacity-50">
        {(status?.voices.length ? status.voices : [{ id: 'af_heart', name: 'Heart', language: 'en-US' }]).map((voice) => <option key={voice.id} value={voice.id}>{voice.name} · {voice.language === 'en-GB' ? tx("British English") : tx("American English")}</option>)}
      </select>
    </Field>
    {(notice || status?.error) && <p role="alert" className="rounded-lg border border-line bg-card px-3 py-2 text-[13px] leading-relaxed text-ink-2">{notice || status?.error}</p>}
  </div>
}
