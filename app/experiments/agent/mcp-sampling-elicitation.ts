// MCP client-side handlers for the two SERVER→CLIENT requests an MCP server can make once the
// client advertises the matching capability on `initialize`:
//
//   • sampling/createMessage — the server asks the CLIENT to run a model turn on its behalf
//     (agentic MCP servers that want an LLM completion without holding their own API key). The
//     client is the human-in-the-loop: the request is approval-gated, then handed to an injected
//     model function.
//   • elicitation/create — the server needs a missing structured input from the user (a param it
//     couldn't infer). The client surfaces it and returns an action + typed content.
//
// mcp-client.ts already sends `capabilities: {}` on initialize and answers every inbound
// server request with a -32601 "unsupported" error (sendUnsupportedResponse). This module is the
// contained core that turns those two methods into real handlers. It is PURE / dependency-injected:
// no process, no stdin, no clock, no model — every side-effecting dependency (approval, model,
// elicit UI) is injected, so the whole thing is unit-testable. mcp-client.ts wires it by (a) adding
// `sampling: {}` and `elicitation: {}` to the initialize capabilities and (b) dispatching matching
// inbound requests to `handleClientRequest`, writing the returned JSON-RPC envelope back to stdin.

export const SAMPLING_METHOD = 'sampling/createMessage'
export const ELICITATION_METHOD = 'elicitation/create'

// JSON-RPC error codes (subset used here). -32602 = invalid params; -32603 = internal error;
// -32601 = method not found (returned by the caller for anything this module doesn't own).
export const JSONRPC_INVALID_PARAMS = -32602
export const JSONRPC_INTERNAL_ERROR = -32603

export type JsonRpcId = string | number

// ---- Sampling (sampling/createMessage) ------------------------------------------------------

export interface SamplingTextContent {
  type: 'text'
  text: string
}
export interface SamplingImageContent {
  type: 'image'
  data: string
  mimeType: string
}
export interface SamplingAudioContent {
  type: 'audio'
  data: string
  mimeType: string
}
export type SamplingContent = SamplingTextContent | SamplingImageContent | SamplingAudioContent

export interface SamplingMessage {
  role: 'user' | 'assistant'
  content: SamplingContent
}

export interface SamplingModelPreferences {
  hints?: { name?: string }[]
  costPriority?: number
  speedPriority?: number
  intelligencePriority?: number
}

// The params an MCP server sends with sampling/createMessage (MCP 2025-06-18).
export interface SamplingRequestParams {
  messages: SamplingMessage[]
  systemPrompt?: string
  includeContext?: 'none' | 'thisServer' | 'allServers'
  temperature?: number
  maxTokens?: number
  stopSequences?: string[]
  modelPreferences?: SamplingModelPreferences
  metadata?: Record<string, unknown>
}

// What the injected model fn is asked to produce, and what it returns. Keeping this narrow (rather
// than reusing the whole app model client) is what makes the handler testable with a stub.
export interface SamplingModelInput {
  serverId: string
  serverName: string
  messages: SamplingMessage[]
  systemPrompt?: string
  temperature?: number
  maxTokens?: number
  stopSequences?: string[]
  modelPreferences?: SamplingModelPreferences
  signal?: AbortSignal
}

export interface SamplingModelResult {
  model: string
  role?: 'assistant'
  content: SamplingContent
  stopReason?: 'endTurn' | 'stopSequence' | 'maxTokens' | string
}

// The shape sampling/createMessage must resolve to (CreateMessageResult in the MCP schema).
export interface SamplingResult {
  model: string
  role: 'assistant'
  content: SamplingContent
  stopReason?: string
}

// ---- Elicitation (elicitation/create) -------------------------------------------------------

// A restricted JSON-Schema subset: MCP elicitation only permits a flat object of primitive props.
export interface ElicitationSchema {
  type: 'object'
  properties: Record<string, unknown>
  required?: string[]
}

export interface ElicitationRequestParams {
  message: string
  requestedSchema: ElicitationSchema
}

// What the injected UI/prompt hook is handed. `serverId`/`serverName` let the UI attribute the ask.
export interface ElicitationPromptInput {
  serverId: string
  serverName: string
  message: string
  requestedSchema: ElicitationSchema
  signal?: AbortSignal
}

// The user's answer. `accept` returns the typed content; `decline`/`cancel` return no content.
export type ElicitationAction = 'accept' | 'decline' | 'cancel'
export interface ElicitationPromptResult {
  action: ElicitationAction
  content?: Record<string, unknown>
}

// The shape elicitation/create must resolve to (ElicitResult in the MCP schema).
export interface ElicitationResult {
  action: ElicitationAction
  content?: Record<string, unknown>
}

// ---- Dependencies (injected) ----------------------------------------------------------------

export interface ClientRequestDeps {
  serverId: string
  serverName: string
  // Approval mode for this server. 'trusted' skips the sampling gate; 'ask' requires approval.
  approvalMode: 'ask' | 'trusted'
  // Gate a sampling request through the user. Resolve true to run the model, false to deny.
  // Omitted → treated as denied unless approvalMode==='trusted' (fail closed).
  approveSampling?: (input: SamplingModelInput) => Promise<boolean>
  // Run one model turn. REQUIRED to serve sampling — absent → sampling is reported unsupported.
  runModel?: (input: SamplingModelInput) => Promise<SamplingModelResult>
  // Surface the elicitation to the user and collect a typed answer. REQUIRED to serve
  // elicitation — absent → elicitation is reported unsupported.
  promptElicitation?: (input: ElicitationPromptInput) => Promise<ElicitationPromptResult>
  // Cancel in-flight model/UI when the run is torn down.
  signal?: AbortSignal
}

// ---- JSON-RPC envelope --------------------------------------------------------------------

export interface JsonRpcResult {
  jsonrpc: '2.0'
  id: JsonRpcId
  result: unknown
}
export interface JsonRpcErrorEnvelope {
  jsonrpc: '2.0'
  id: JsonRpcId
  error: { code: number; message: string; data?: unknown }
}
export type JsonRpcResponse = JsonRpcResult | JsonRpcErrorEnvelope

function ok(id: JsonRpcId, result: unknown): JsonRpcResult {
  return { jsonrpc: '2.0', id, result }
}
function err(id: JsonRpcId, code: number, message: string, data?: unknown): JsonRpcErrorEnvelope {
  const error: JsonRpcErrorEnvelope['error'] = { code, message }
  if (data !== undefined) error.data = data
  return { jsonrpc: '2.0', id, error }
}

// The capabilities the client advertises on `initialize` once these handlers are wired. Only
// advertise what the injected deps can actually serve, so a server never sends a request we then
// have to refuse — advertising a capability you can't serve is a protocol lie.
export function clientSamplingCapabilities(deps: Pick<ClientRequestDeps, 'runModel' | 'promptElicitation'>): {
  sampling?: Record<string, never>
  elicitation?: Record<string, never>
} {
  const caps: { sampling?: Record<string, never>; elicitation?: Record<string, never> } = {}
  if (deps.runModel) caps.sampling = {}
  if (deps.promptElicitation) caps.elicitation = {}
  return caps
}

// True if this inbound method is one this module owns (mcp-client dispatches only these here).
export function isClientRequestMethod(method: unknown): method is typeof SAMPLING_METHOD | typeof ELICITATION_METHOD {
  return method === SAMPLING_METHOD || method === ELICITATION_METHOD
}

// ---- Validation (pure) --------------------------------------------------------------------

function isRecord(value: unknown): value is Record<string, any> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}

function validSamplingContent(value: unknown): value is SamplingContent {
  if (!isRecord(value)) return false
  if (value.type === 'text') return typeof value.text === 'string'
  if (value.type === 'image' || value.type === 'audio') return typeof value.data === 'string' && typeof value.mimeType === 'string'
  return false
}

export function parseSamplingParams(params: unknown): { ok: true; value: SamplingRequestParams } | { ok: false; error: string } {
  if (!isRecord(params)) return { ok: false, error: 'sampling/createMessage requires an object params.' }
  const messages = params.messages
  if (!Array.isArray(messages) || messages.length === 0) return { ok: false, error: 'sampling/createMessage requires a non-empty messages array.' }
  const parsed: SamplingMessage[] = []
  for (const m of messages) {
    if (!isRecord(m) || (m.role !== 'user' && m.role !== 'assistant')) return { ok: false, error: 'each sampling message needs role user|assistant.' }
    if (!validSamplingContent(m.content)) return { ok: false, error: 'each sampling message needs valid text/image/audio content.' }
    parsed.push({ role: m.role, content: m.content as SamplingContent })
  }
  const value: SamplingRequestParams = { messages: parsed }
  if (typeof params.systemPrompt === 'string') value.systemPrompt = params.systemPrompt
  if (typeof params.temperature === 'number' && Number.isFinite(params.temperature)) value.temperature = params.temperature
  if (typeof params.maxTokens === 'number' && Number.isFinite(params.maxTokens) && params.maxTokens > 0) value.maxTokens = Math.floor(params.maxTokens)
  if (Array.isArray(params.stopSequences)) value.stopSequences = params.stopSequences.filter((s): s is string => typeof s === 'string')
  if (isRecord(params.modelPreferences)) value.modelPreferences = params.modelPreferences as SamplingModelPreferences
  if (params.includeContext === 'none' || params.includeContext === 'thisServer' || params.includeContext === 'allServers') value.includeContext = params.includeContext
  return { ok: true, value }
}

export function parseElicitationParams(params: unknown): { ok: true; value: ElicitationRequestParams } | { ok: false; error: string } {
  if (!isRecord(params)) return { ok: false, error: 'elicitation/create requires an object params.' }
  const message = params.message
  if (typeof message !== 'string' || !message.trim()) return { ok: false, error: 'elicitation/create requires a non-empty message.' }
  const schema = params.requestedSchema
  if (!isRecord(schema) || schema.type !== 'object' || !isRecord(schema.properties)) {
    return { ok: false, error: 'elicitation/create requires requestedSchema of type object with properties.' }
  }
  const value: ElicitationRequestParams = {
    message,
    requestedSchema: {
      type: 'object',
      properties: schema.properties,
      required: Array.isArray(schema.required) ? schema.required.filter((s): s is string => typeof s === 'string') : undefined,
    },
  }
  return { ok: true, value }
}

// Keep only the properties declared in the requested schema, and drop anything not a primitive
// (string/number/boolean) — MCP elicitation content is a flat primitive object. This is the
// client's guard against a malicious/buggy prompt hook returning arbitrary nested junk.
export function sanitizeElicitationContent(
  content: Record<string, unknown> | undefined,
  schema: ElicitationSchema,
): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  if (!content) return out
  const allowed = new Set(Object.keys(schema.properties))
  for (const [key, val] of Object.entries(content)) {
    if (!allowed.has(key)) continue
    if (typeof val === 'string' || typeof val === 'number' || typeof val === 'boolean') out[key] = val
  }
  return out
}

// ---- Handlers -------------------------------------------------------------------------------

async function handleSampling(id: JsonRpcId, params: unknown, deps: ClientRequestDeps): Promise<JsonRpcResponse> {
  if (!deps.runModel) return err(id, JSONRPC_INTERNAL_ERROR, 'This client does not support sampling/createMessage.')
  const parsed = parseSamplingParams(params)
  if (!parsed.ok) return err(id, JSONRPC_INVALID_PARAMS, parsed.error)

  const input: SamplingModelInput = {
    serverId: deps.serverId,
    serverName: deps.serverName,
    messages: parsed.value.messages,
    systemPrompt: parsed.value.systemPrompt,
    temperature: parsed.value.temperature,
    maxTokens: parsed.value.maxTokens,
    stopSequences: parsed.value.stopSequences,
    modelPreferences: parsed.value.modelPreferences,
    signal: deps.signal,
  }

  // Human-in-the-loop gate. 'trusted' servers skip it; otherwise an approval fn MUST say yes.
  // Fail closed: no approver on an 'ask' server = denied.
  if (deps.approvalMode !== 'trusted') {
    const approver = deps.approveSampling
    const approved = approver ? await approver(input).catch(() => false) : false
    if (!approved) return err(id, JSONRPC_INTERNAL_ERROR, `${deps.serverName} sampling request was denied.`)
  }

  let result: SamplingModelResult
  try {
    result = await deps.runModel(input)
  } catch (error: any) {
    return err(id, JSONRPC_INTERNAL_ERROR, `Sampling failed: ${String(error?.message || error)}`)
  }
  if (!result || !validSamplingContent(result.content) || typeof result.model !== 'string') {
    return err(id, JSONRPC_INTERNAL_ERROR, 'Sampling model returned an invalid result.')
  }
  const value: SamplingResult = {
    model: result.model,
    role: 'assistant',
    content: result.content,
    stopReason: typeof result.stopReason === 'string' ? result.stopReason : 'endTurn',
  }
  return ok(id, value)
}

async function handleElicitation(id: JsonRpcId, params: unknown, deps: ClientRequestDeps): Promise<JsonRpcResponse> {
  if (!deps.promptElicitation) return err(id, JSONRPC_INTERNAL_ERROR, 'This client does not support elicitation/create.')
  const parsed = parseElicitationParams(params)
  if (!parsed.ok) return err(id, JSONRPC_INVALID_PARAMS, parsed.error)

  let answer: ElicitationPromptResult
  try {
    answer = await deps.promptElicitation({
      serverId: deps.serverId,
      serverName: deps.serverName,
      message: parsed.value.message,
      requestedSchema: parsed.value.requestedSchema,
      signal: deps.signal,
    })
  } catch (error: any) {
    return err(id, JSONRPC_INTERNAL_ERROR, `Elicitation failed: ${String(error?.message || error)}`)
  }

  const action: ElicitationAction = answer?.action === 'accept' || answer?.action === 'decline' || answer?.action === 'cancel'
    ? answer.action
    : 'cancel'
  const value: ElicitationResult = { action }
  // Only 'accept' carries content, and only the schema-declared primitive fields survive.
  if (action === 'accept') value.content = sanitizeElicitationContent(answer.content, parsed.value.requestedSchema)
  return ok(id, value)
}

/**
 * Dispatch an inbound server→client request to the matching handler and return the JSON-RPC
 * envelope the caller should write back to the server's stdin. Only handles the two methods this
 * module owns; anything else resolves to a -32601 error so the caller can keep a single write path.
 * Never throws — every failure becomes a JSON-RPC error response.
 */
export async function handleClientRequest(
  id: JsonRpcId,
  method: string,
  params: unknown,
  deps: ClientRequestDeps,
): Promise<JsonRpcResponse> {
  try {
    if (method === SAMPLING_METHOD) return await handleSampling(id, params, deps)
    if (method === ELICITATION_METHOD) return await handleElicitation(id, params, deps)
    return err(id, -32601, `TensorFold does not implement client method ${method}.`)
  } catch (error: any) {
    return err(id, JSONRPC_INTERNAL_ERROR, String(error?.message || error))
  }
}
