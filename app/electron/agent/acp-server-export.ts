export type AcpExportKind = 'thread' | 'mission'
export type AcpRequestId = string | number | null

export interface AcpExportTarget {
  kind: AcpExportKind
  id: string
  title: string
  cwd?: string | null
  description?: string | null
}

export interface AcpJsonRpcResult<T> {
  jsonrpc: '2.0'
  id: AcpRequestId
  result: T
}

export interface AcpJsonRpcError {
  jsonrpc: '2.0'
  id: AcpRequestId
  error: {
    code: number
    message: string
  }
}

export function acpSessionIdForTarget(target: AcpExportTarget): string {
  return `vontra:${target.kind}:${target.id}`
}

export function buildAcpInitializeResult(target: AcpExportTarget) {
  return {
    protocolVersion: 1,
    agentCapabilities: {
      loadSession: true,
      promptCapabilities: { image: false, audio: false, embeddedContext: true },
      mcpCapabilities: { http: false, sse: false },
      sessionCapabilities: {
        prompt: true,
        cancel: true,
      },
      auth: {},
    },
    authMethods: [],
    agentInfo: {
      name: `TensorFold ${target.kind}`,
      version: '0.1.0',
    },
    _meta: {
      vontra: {
        exportKind: target.kind,
        targetId: target.id,
        title: target.title,
        cwd: target.cwd ?? null,
      },
    },
  }
}

export function buildAcpNewSessionResult(target: AcpExportTarget) {
  return {
    sessionId: acpSessionIdForTarget(target),
    sessionMode: {
      currentModeId: 'default',
      availableModes: [{ id: 'default', name: 'TensorFold', description: target.description ?? target.title }],
    },
    _meta: {
      vontra: {
        exportKind: target.kind,
        targetId: target.id,
      },
    },
  }
}

export function acpResult<T>(id: AcpRequestId, result: T): AcpJsonRpcResult<T> {
  return { jsonrpc: '2.0', id, result }
}

export function acpError(id: AcpRequestId, code: number, message: string): AcpJsonRpcError {
  return { jsonrpc: '2.0', id, error: { code, message } }
}

export function exportedAcpMethods(): string[] {
  return ['initialize', 'session/new', 'session/load', 'session/prompt', 'session/cancel']
}
