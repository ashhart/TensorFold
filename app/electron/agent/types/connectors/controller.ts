

// ---- Controller: attach current app/window context on demand ----
export interface ControllerSettings {
  enabled: boolean
  includeAccessibilityText: boolean
  maxTextChars: number
  lastCapturedAt: number | null
  lastError: string | null
}

export interface ControllerStatus {
  platform: NodeJS.Platform | string
  supported: boolean
  enabled: boolean
  accessibilityTrusted: boolean
  canCapture: boolean
  message: string
}

export interface ControllerSnapshot {
  appName: string
  bundleId: string | null
  windowTitle: string
  capturedAt: number
  platform: NodeJS.Platform | string
  text: string
  textCharCount: number
  truncated: boolean
  source: 'accessibility' | 'demo'
  warning: string | null
}
