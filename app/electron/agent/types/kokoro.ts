export interface KokoroVoice {
  id: string
  name: string
  language: 'en-US' | 'en-GB'
}

export interface KokoroStatus {
  state: 'missing' | 'downloading' | 'verifying' | 'ready' | 'error'
  installed: boolean
  downloadedBytes: number
  totalBytes: number
  message: string
  error?: string
  voices: KokoroVoice[]
  modelRevision: string
}

export interface KokoroSynthesisInput {
  text: string
  requestId: string
  voice?: string
  rate?: number
}
