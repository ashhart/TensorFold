import { useEffect, useRef, useState } from 'react'
import type { QuickAskApi, QuickAskSnapshot } from '../../../shared/quick-ask'

export function useAskVoice(api: QuickAskApi, snapshot: QuickAskSnapshot | null, append: (text: string) => void, error: (error: unknown) => void) {
  const [listening, setListening] = useState(false), [speaking, setSpeaking] = useState(false)
  const recognition = useRef<any>(null), audio = useRef<HTMLAudioElement | null>(null), generation = useRef(0)
  const Recognition = (window as any).SpeechRecognition ?? (window as any).webkitSpeechRecognition
  const dictationAvailable = !!Recognition && snapshot?.voice.input === 'browser'
  const stopSpeech = () => { generation.current++; audio.current?.pause(); audio.current = null; window.speechSynthesis?.cancel(); setSpeaking(false) }
  function dictate() {
    if (listening) { recognition.current?.stop(); return }
    if (!dictationAvailable) { error(new Error('System dictation is unavailable here. You can use your operating system’s dictation shortcut in the question box.')); return }
    stopSpeech()
    const current = new Recognition()
    current.continuous = false; current.interimResults = false
    current.lang = snapshot?.appearance.uiPreferences?.locale || navigator.language
    current.onresult = (event: any) => { const text = event.results?.[0]?.[0]?.transcript; if (typeof text === 'string') append(text) }
    current.onerror = () => { setListening(false); error(new Error('Dictation could not start. Check microphone access or type your question.')) }
    current.onend = () => { recognition.current = null; setListening(false) }
    recognition.current = current
    try { current.start(); setListening(true) } catch (failure) { error(failure); setListening(false) }
  }
  async function speak(text: string) {
    if (speaking) { stopSpeech(); return }
    stopSpeech(); setSpeaking(true)
    const token = generation.current
    try {
      if (snapshot?.voice.output === 'unavailable') throw new Error('Enable voice output in TensorFold settings.')
      if (snapshot?.voice.output === 'configured') {
        const result = await api.speak(text.slice(0, 5000))
        if (token !== generation.current) return
        if (!result.ok || !result.audioBase64 || !/^audio\/(wav|mpeg|mp3|ogg|webm|mp4)$/.test(result.mimeType ?? '')) throw new Error(result.error ?? 'Speech output is unavailable.')
        const element = new Audio(`data:${result.mimeType};base64,${result.audioBase64}`)
        audio.current = element; element.onended = () => setSpeaking(false); element.onerror = () => { setSpeaking(false); error(new Error('The spoken answer could not play.')) }
        await element.play()
      } else {
        if (!window.speechSynthesis) throw new Error('System speech is unavailable on this computer.')
        const utterance = new SpeechSynthesisUtterance(text.slice(0, 5000))
        const voices = window.speechSynthesis.getVoices().filter(voice => voice.localService)
        if (!voices.length) throw new Error('No offline system voice is available. Configure local speech in TensorFold settings.')
        utterance.lang = snapshot?.appearance.uiPreferences?.locale || navigator.language
        utterance.voice = voices.find(voice => voice.lang.startsWith(utterance.lang.split('-')[0])) ?? voices[0]
        utterance.onend = () => setSpeaking(false); utterance.onerror = () => setSpeaking(false)
        window.speechSynthesis.speak(utterance)
      }
    } catch (failure) { if (token === generation.current) { setSpeaking(false); error(failure) } }
  }
  useEffect(() => () => { recognition.current?.abort(); generation.current++; audio.current?.pause(); window.speechSynthesis?.cancel() }, [])
  return { dictate, speak, listening, speaking, stopSpeech, dictationAvailable }
}
