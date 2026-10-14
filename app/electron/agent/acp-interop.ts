export type AcpRuntimeRole = 'client' | 'server'
export type AcpPrototypeId =
  | 'host_external_agent'
  | 'timeline_mapping'
  | 'approval_bridge'
  | 'cancel_guide_audit'
  | 'server_export'

export interface AcpBoundary {
  id: 'auth' | 'model_selection' | 'tools' | 'policy' | 'filesystem' | 'timeline'
  rule: string
}

export interface AcpExplorationSpike {
  sourceUrl: string
  protocolVersion: 1
  transport: 'json-rpc-stdio'
  firstRole: AcpRuntimeRole
  laterRoles: AcpRuntimeRole[]
  reason: string
  boundaries: AcpBoundary[]
  prototypes: AcpPrototypeId[]
}

export function buildAcpExplorationSpike(): AcpExplorationSpike {
  return {
    sourceUrl: 'https://github.com/agentclientprotocol/agent-client-protocol',
    protocolVersion: 1,
    transport: 'json-rpc-stdio',
    firstRole: 'client',
    laterRoles: ['server'],
    reason: 'TensorFold should first host external coding agents as subprocess runtimes inside a normal thread or room lane. That fits the editor-as-client ACP model while preserving TensorFold approvals, model routing, provider-account isolation, timeline evidence, and cancellation semantics before exposing TensorFold missions outward as an ACP server.',
    boundaries: [
      {
        id: 'auth',
        rule: 'External agent credentials must live on the provider/account profile that launched the runtime; ACP sessions do not inherit unrelated provider, MCP, connector, or shell secrets.',
      },
      {
        id: 'model_selection',
        rule: 'ACP runtime selection chooses an agent transport, while model/account selection stays in TensorFold provider routing and CLI profile isolation.',
      },
      {
        id: 'tools',
        rule: 'External agent tool requests are evidence until mapped to a TensorFold tool capability and passed through normal approval policy.',
      },
      {
        id: 'policy',
        rule: 'Mutating filesystem, connector, browser, MCP, package-install, network-shell, and publish actions remain blocked unless TensorFold policy grants the exact scoped action.',
      },
      {
        id: 'filesystem',
        rule: 'ACP sessions receive the workspace root and optional selected files only; home-directory and cross-project access require explicit capability policy.',
      },
      {
        id: 'timeline',
        rule: 'Agent messages, edits, tool calls, errors, approvals, cancellation, and user guidance must be mirrored into TensorFold timeline items for audit and resume.',
      },
    ],
    prototypes: ['host_external_agent', 'timeline_mapping', 'approval_bridge', 'cancel_guide_audit', 'server_export'],
  }
}

export function acpFirstImplementation(spike: AcpExplorationSpike = buildAcpExplorationSpike()): string {
  return spike.firstRole === 'client'
    ? 'ACP client first: host external agents inside TensorFold threads and rooms before exporting TensorFold as a server.'
    : 'ACP server first: expose TensorFold missions to another editor before hosting external agents.'
}

export function acpBoundaryIds(spike: AcpExplorationSpike = buildAcpExplorationSpike()): string[] {
  return spike.boundaries.map((boundary) => boundary.id)
}
