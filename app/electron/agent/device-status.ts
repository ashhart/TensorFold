import type { EnginePerformance, EngineStatus, SecurityRole } from './types'

export interface RemoteDeviceStatus {
  scope: 'paired' | 'owner'
  role: SecurityRole
  capabilities: {
    taskStart: boolean; taskGuide: boolean; taskStop: boolean;
    approvalResolve: boolean; transcriptReview: boolean; changesReview: boolean; engineControl: false
  }
  engine: {
    available: boolean; state: EngineStatus['state']; model: string | null; drafter: string | null;
    hardware: EngineStatus['hardware'] | null; contextTokens: number | null; totalMemoryBytes: number; memoryLimitBytes: number;
    performance: EnginePerformance | null
  } | null
}

/** Pick operational fields only; device status never reveals paths, launch keys, logs or prompts. */
export function remoteDeviceStatus(scope: RemoteDeviceStatus['scope'], role: SecurityRole, status: EngineStatus | null, performance: EnginePerformance | null = null): RemoteDeviceStatus {
  const canWrite = role === 'operator' || role === 'owner'
  const ownerView = scope === 'owner'
  return {
    scope, role,
    capabilities: { taskStart: canWrite, taskGuide: canWrite, taskStop: canWrite, approvalResolve: canWrite && ownerView, transcriptReview: ownerView, changesReview: ownerView, engineControl: false },
    engine: status ? {
      available: status.available, state: status.state, model: status.model, drafter: status.drafter,
      hardware: status.hardware ? { platform: status.hardware.platform, architecture: status.hardware.architecture, chip: status.hardware.chip } : null,
      contextTokens: status.contextTokens, totalMemoryBytes: status.totalMemoryBytes, memoryLimitBytes: status.memoryLimitBytes,
      performance: performance ? { model: performance.model, measuredAt: performance.measuredAt, decodeTokensPerSecond: performance.decodeTokensPerSecond, prefillTokensPerSecond: performance.prefillTokensPerSecond, firstTokenMs: performance.firstTokenMs, completionTokens: performance.completionTokens } : null,
    } : null,
  }
}
