export interface MobileBridgeInterface {
  name: string
  address: string
}

export interface MobileBridgeConfig {
  version: 1
  enabled: boolean
  host: string
  port: number
}

export interface MobileBridgeStatus {
  enabled: boolean
  listening: boolean
  url: string | null
  host: string | null
  port: number | null
  certificateSha256: string | null
  certificateExpiresAt: number | null
  interfaces: MobileBridgeInterface[]
  clients: number
  lastError: string | null
}

export interface MobileBridgePairing {
  link: string
  expiresAt: number
  certificateSha256: string
}
