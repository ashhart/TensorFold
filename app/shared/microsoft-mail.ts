export interface MicrosoftMailSignIn {
  id: string
  userCode: string
  verificationUri: string
  expiresAt: number
  intervalMs: number
}

export type MicrosoftMailPoll = { status: 'pending'; intervalMs: number } | { status: 'connected' }
