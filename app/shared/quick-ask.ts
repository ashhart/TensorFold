import type { Attachment, ThemePalette, Thread, VoiceSynthesisResult } from '../electron/agent/types'
import type { UiPreferences } from './ui-preferences'

export interface QuickAskRoute {
  id: string
  providerId: string
  accountId: string | null
  model: string
  label: string
  destination: 'device' | 'remote' | 'cloud'
  location: string
  hostId?: string
  state: 'ready' | 'load-required'
  vision: 'observed' | 'unknown' | 'unsupported'
  binding: string
  upstreamBinding?: string
}

export interface QuickAskPreset { id: string; name: string; command: string; prompt: string; routeId: string | null }
export interface QuickAskPreferences { enabled: boolean; shortcut: string; presets: QuickAskPreset[] }
export interface QuickAskDraft { revision: number; text: string; attachments: Attachment[]; routeId: string | null; presetId: string; web: boolean }
export interface QuickAskTurn {
  id: string
  text: string
  attachments: Attachment[]
  answer: string
  reasoning: string
  status: 'running' | 'complete' | 'cancelled' | 'error'
  tools: Array<{ name: string; detail: string; status: 'running' | 'complete' | 'error' }>
  error?: string
  at: number
}
export interface QuickAskSession {
  id: string
  draft: QuickAskDraft
  route: QuickAskRoute | null
  prompt: string | null
  presetId?: string
  turns: QuickAskTurn[]
  handoffThreadId?: string
}
export interface QuickAskSnapshot {
  session: QuickAskSession
  preferences: QuickAskPreferences
  routes: QuickAskRoute[]
  routeErrors: string[]
  shortcutError: string | null
  appearance: { palette: ThemePalette; dark: boolean; uiPreferences?: UiPreferences }
  voice: { input: 'browser' | 'unavailable'; output: 'browser' | 'configured' | 'unavailable'; destination: string }
  platform: string
}
export type QuickAskEvent =
  | { kind: 'turn'; sessionId: string; turn: QuickAskTurn }
  | { kind: 'refresh' }
  | { kind: 'focus' }
  | { kind: 'flush'; requestId: string }

export interface QuickAskApi {
  resize(height: number): Promise<void>
  pickWindow(): Promise<QuickAskWindowSelection>
  useWindow(token: string): Promise<Attachment[]>
  snapshot(options?: { refreshRoutes?: boolean }): Promise<QuickAskSnapshot>
  saveDraft(draft: QuickAskDraft): Promise<QuickAskDraft>
  configure(preferences: QuickAskPreferences): Promise<QuickAskPreferences>
  recordShortcut(active: boolean): Promise<void>
  send(input: { revision: number; approvedRouteId: string | null }): Promise<{ requestId: string }>
  cancel(): Promise<void>
  reset(): Promise<void>
  clipboard(): Promise<Attachment[]>
  displays(): Promise<Array<{ id: string; label: string }>>
  capture(displayId: string): Promise<Attachment>
  loadModel(routeId: string): Promise<void>
  handoff(): Promise<{ threadId: string }>
  copy(text: string): Promise<void>
  openLink(url: string): Promise<void>
  hide(): Promise<void>
  acknowledgeFlush(input: { requestId: string; error?: string }): Promise<void>
  speak(text: string): Promise<VoiceSynthesisResult>
  onEvent(callback: (event: QuickAskEvent) => void): () => void
}
export interface QuickAskWindowChoice { token: string; title: string; thumbnail: string }
export type QuickAskWindowSelection = { kind: 'cancelled' } | { kind: 'attached'; attachments: Attachment[] } | { kind: 'choose'; windows: QuickAskWindowChoice[] }
export interface QuickAskLauncher {
  open(): Promise<void>
  onHandoff(callback: () => void): () => void
  consumeHandoff(): Promise<{ thread: Thread; draft: QuickAskDraft | null } | null>
}
declare global {
  interface Window { tensorfoldQuickAsk?: QuickAskApi; tensorfoldQuickAskLauncher?: QuickAskLauncher }
}

export function quickAskDefaults(): QuickAskPreferences {
  return { enabled: false, shortcut: 'CommandOrControl+Shift+Space', presets: [
    { id: 'ask', name: 'Ask', command: 'ask', prompt: 'Answer the question clearly and directly. Be concise unless detail is needed.', routeId: null },
    { id: 'rewrite', name: 'Rewrite', command: 'rewrite', prompt: 'Rewrite the supplied text to be clear and natural. Preserve its meaning and voice. Return the rewrite.', routeId: null },
    { id: 'summarise', name: 'Summarise', command: 'summarise', prompt: 'Summarise the supplied material accurately. Preserve important facts, decisions and uncertainties.', routeId: null },
    { id: 'translate', name: 'Translate', command: 'translate', prompt: 'Translate the supplied text into the language requested. If no target language is given, ask which language. Preserve tone and formatting.', routeId: null },
    { id: 'explain', name: 'Explain', command: 'explain', prompt: 'Explain the supplied material in plain language. Give a concrete example when useful.', routeId: null },
  ] }
}
