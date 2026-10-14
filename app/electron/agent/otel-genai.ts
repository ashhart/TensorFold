// OpenTelemetry GenAI span export.
//
// Maps a run / turn / tool-call into OpenTelemetry GenAI semantic-convention
// spans (gen_ai.* attributes) as a plain, serializable structure, plus an
// OTLP/JSON emitter that ships them over an injected HTTP sender.
//
// PURE + DI. No imports of shared runtime. All I/O (the HTTP sender, the id
// generators, the clock) is injected so the whole module is unit-testable with
// `npx jiti electron/agent/otel-genai.test.ts`.

import { readBoundedResponseText } from './bounded-response'
import { discardResponseBody } from './pinned-fetch'
//
// Spec references (OTel Semantic Conventions for Generative AI, 1.29+):
//   - Span name: "{gen_ai.operation.name} {gen_ai.request.model}"
//   - gen_ai.system / gen_ai.operation.name / gen_ai.request.model
//   - gen_ai.usage.input_tokens / gen_ai.usage.output_tokens
//   - execute_tool spans carry gen_ai.tool.name / gen_ai.tool.call.id
//
// The mapper takes app-internal run/turn/tool shapes (kept structural here so
// the module compiles stand-alone) and produces `OtelSpan[]`. The emitter wraps
// those into the OTLP/JSON `resourceSpans` envelope and POSTs it.

// ---------------------------------------------------------------------------
// Serializable OTLP/JSON primitives
// ---------------------------------------------------------------------------

/** A single OTLP `AnyValue`, restricted to the subset GenAI attributes use. */
export type OtelAnyValue =
  | { stringValue: string }
  | { intValue: number }
  | { doubleValue: number }
  | { boolValue: boolean }
  | { arrayValue: { values: OtelAnyValue[] } }

export interface OtelKeyValue {
  key: string
  value: OtelAnyValue
}

/** OTLP span status codes. 0=UNSET, 1=OK, 2=ERROR. */
export type OtelStatusCode = 0 | 1 | 2

/** OTLP span kinds. We only ever emit CLIENT(3) or INTERNAL(1). */
export type OtelSpanKind = 1 | 3

export interface OtelSpan {
  traceId: string // 16-byte hex (32 chars)
  spanId: string // 8-byte hex (16 chars)
  parentSpanId?: string
  name: string
  kind: OtelSpanKind
  startTimeUnixNano: string
  endTimeUnixNano: string
  attributes: OtelKeyValue[]
  status: { code: OtelStatusCode; message?: string }
}

export interface OtelResourceSpans {
  resource: { attributes: OtelKeyValue[] }
  scopeSpans: Array<{
    scope: { name: string; version?: string }
    spans: OtelSpan[]
  }>
}

export interface OtlpTracePayload {
  resourceSpans: OtelResourceSpans[]
}

// ---------------------------------------------------------------------------
// App-internal input shapes (structural — no shared-type import)
// ---------------------------------------------------------------------------

export type GenAiOperation =
  | 'chat'
  | 'text_completion'
  | 'execute_tool'
  | 'embeddings'

export interface GenAiUsage {
  inputTokens?: number | null
  outputTokens?: number | null
}

/** One model turn (a `/chat/completions` round-trip). */
export interface TurnSpanInput {
  /** Stable id for this turn; used to derive a deterministic spanId when no id gen is injected. */
  id: string
  operation?: GenAiOperation // default 'chat'
  system: string // gen_ai.system, e.g. 'openai' | 'anthropic' | 'codex'
  requestModel?: string | null // gen_ai.request.model
  responseModel?: string | null // gen_ai.response.model
  temperature?: number | null
  maxTokens?: number | null
  usage?: GenAiUsage | null
  responseId?: string | null
  finishReason?: string | null
  startMs: number
  endMs: number
  error?: string | null
}

/** One tool / function call executed during a run. */
export interface ToolCallSpanInput {
  id: string
  toolName: string // gen_ai.tool.name
  callId?: string | null // gen_ai.tool.call.id
  toolType?: string | null // gen_ai.tool.type e.g. 'function' | 'mcp'
  description?: string | null
  startMs: number
  endMs: number
  error?: string | null
  /** The turn (`TurnSpanInput.id`) that requested this call, for parenting. */
  parentTurnId?: string | null
}

/** A whole run: the root span, with its turns and tool calls as children. */
export interface RunSpanInput {
  runId: string
  threadId?: string | null
  name?: string | null // human label; defaults to "gen_ai.run {runId}"
  system?: string | null // fallback gen_ai.system for the root
  startMs: number
  endMs: number
  error?: string | null
  turns?: TurnSpanInput[]
  toolCalls?: ToolCallSpanInput[]
}

export interface ResourceInfo {
  serviceName?: string // service.name; default 'vontra'
  serviceVersion?: string // service.version
  attributes?: Record<string, string | number | boolean>
}

// ---------------------------------------------------------------------------
// Deterministic id generation (DI-overridable)
// ---------------------------------------------------------------------------

/** FNV-1a 32-bit — small, deterministic, dependency-free. */
function fnv1a(input: string): number {
  let h = 0x811c9dc5
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i)
    h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0
  }
  return h >>> 0
}

function hex(n: number, width: number): string {
  return (n >>> 0).toString(16).padStart(8, '0').slice(-width)
}

/** Deterministic 32-char (16-byte) trace id from a seed string. */
export function deriveTraceId(seed: string): string {
  const a = fnv1a(seed)
  const b = fnv1a(`${seed}:1`)
  const c = fnv1a(`${seed}:2`)
  const d = fnv1a(`${seed}:3`)
  return `${hex(a, 8)}${hex(b, 8)}${hex(c, 8)}${hex(d, 8)}`
}

/** Deterministic 16-char (8-byte) span id from a seed string. */
export function deriveSpanId(seed: string): string {
  const a = fnv1a(`${seed}:s0`)
  const b = fnv1a(`${seed}:s1`)
  return `${hex(a, 8)}${hex(b, 8)}`
}

export interface IdGen {
  traceId(seed: string): string
  spanId(seed: string): string
}

const defaultIdGen: IdGen = { traceId: deriveTraceId, spanId: deriveSpanId }

// ---------------------------------------------------------------------------
// Attribute helpers
// ---------------------------------------------------------------------------

function kvString(key: string, value: string): OtelKeyValue {
  return { key, value: { stringValue: value } }
}
function kvInt(key: string, value: number): OtelKeyValue {
  return { key, value: { intValue: Math.trunc(value) } }
}
function kvDouble(key: string, value: number): OtelKeyValue {
  return { key, value: { doubleValue: value } }
}

/** ms → OTLP unix-nanos string. Guards against fractional / non-finite input. */
export function msToUnixNano(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '0'
  // ms * 1e6, done via BigInt to avoid float precision loss at large values.
  const whole = Math.trunc(ms)
  const frac = Math.round((ms - whole) * 1e6) // sub-ms → nanos
  return (BigInt(whole) * 1_000_000n + BigInt(frac)).toString()
}

// ---------------------------------------------------------------------------
// Span mapping (pure)
// ---------------------------------------------------------------------------

const SCOPE = { name: 'vontra.otel.genai', version: '1.0.0' }

function turnSpanName(op: GenAiOperation, model?: string | null): string {
  return model ? `${op} ${model}` : op
}

function statusFor(error?: string | null): { code: OtelStatusCode; message?: string } {
  return error ? { code: 2, message: error } : { code: 1 }
}

/** Map a single model turn to a CLIENT span with gen_ai.* attributes. */
export function mapTurnSpan(
  turn: TurnSpanInput,
  ctx: { traceId: string; parentSpanId?: string; ids?: IdGen },
): OtelSpan {
  const ids = ctx.ids ?? defaultIdGen
  const op = turn.operation ?? 'chat'
  const attrs: OtelKeyValue[] = [
    kvString('gen_ai.operation.name', op),
    kvString('gen_ai.system', turn.system),
  ]
  if (turn.requestModel) attrs.push(kvString('gen_ai.request.model', turn.requestModel))
  if (turn.responseModel) attrs.push(kvString('gen_ai.response.model', turn.responseModel))
  if (turn.responseId) attrs.push(kvString('gen_ai.response.id', turn.responseId))
  if (turn.finishReason) {
    attrs.push({
      key: 'gen_ai.response.finish_reasons',
      value: { arrayValue: { values: [{ stringValue: turn.finishReason }] } },
    })
  }
  if (typeof turn.temperature === 'number') attrs.push(kvDouble('gen_ai.request.temperature', turn.temperature))
  if (typeof turn.maxTokens === 'number') attrs.push(kvInt('gen_ai.request.max_tokens', turn.maxTokens))
  if (turn.usage?.inputTokens != null) attrs.push(kvInt('gen_ai.usage.input_tokens', turn.usage.inputTokens))
  if (turn.usage?.outputTokens != null) attrs.push(kvInt('gen_ai.usage.output_tokens', turn.usage.outputTokens))
  return {
    traceId: ctx.traceId,
    spanId: ids.spanId(`turn:${turn.id}`),
    parentSpanId: ctx.parentSpanId,
    name: turnSpanName(op, turn.requestModel),
    kind: 3, // CLIENT
    startTimeUnixNano: msToUnixNano(turn.startMs),
    endTimeUnixNano: msToUnixNano(turn.endMs),
    attributes: attrs,
    status: statusFor(turn.error),
  }
}

/** Map a tool call to an `execute_tool` INTERNAL span. */
export function mapToolCallSpan(
  call: ToolCallSpanInput,
  ctx: { traceId: string; parentSpanId?: string; ids?: IdGen },
): OtelSpan {
  const ids = ctx.ids ?? defaultIdGen
  const toolName = call.toolName
  const attrs: OtelKeyValue[] = [
    kvString('gen_ai.operation.name', 'execute_tool'),
    kvString('gen_ai.tool.name', toolName),
  ]
  if (call.callId) attrs.push(kvString('gen_ai.tool.call.id', call.callId))
  if (call.toolType) attrs.push(kvString('gen_ai.tool.type', call.toolType))
  if (call.description) attrs.push(kvString('gen_ai.tool.description', call.description))
  return {
    traceId: ctx.traceId,
    spanId: ids.spanId(`tool:${call.id}`),
    parentSpanId: ctx.parentSpanId,
    name: `execute_tool ${toolName}`,
    kind: 1, // INTERNAL
    startTimeUnixNano: msToUnixNano(call.startMs),
    endTimeUnixNano: msToUnixNano(call.endMs),
    attributes: attrs,
    status: statusFor(call.error),
  }
}

/**
 * Map a whole run into a flat `OtelSpan[]`: one root span, one span per turn,
 * one span per tool call. Tool calls parent onto their `parentTurnId` turn when
 * present, else onto the run root. All share one trace id derived from the run.
 */
export function mapRunToSpans(run: RunSpanInput, ids: IdGen = defaultIdGen): OtelSpan[] {
  const traceId = ids.traceId(`run:${run.runId}`)
  const rootSpanId = ids.spanId(`run:${run.runId}`)
  const rootAttrs: OtelKeyValue[] = [kvString('gen_ai.run.id', run.runId)]
  if (run.threadId) rootAttrs.push(kvString('gen_ai.conversation.id', run.threadId))
  if (run.system) rootAttrs.push(kvString('gen_ai.system', run.system))

  const root: OtelSpan = {
    traceId,
    spanId: rootSpanId,
    name: run.name || `gen_ai.run ${run.runId}`,
    kind: 1, // INTERNAL
    startTimeUnixNano: msToUnixNano(run.startMs),
    endTimeUnixNano: msToUnixNano(run.endMs),
    attributes: rootAttrs,
    status: statusFor(run.error),
  }

  const spans: OtelSpan[] = [root]
  const turnSpanIdById = new Map<string, string>()

  for (const turn of run.turns ?? []) {
    const span = mapTurnSpan(turn, { traceId, parentSpanId: rootSpanId, ids })
    turnSpanIdById.set(turn.id, span.spanId)
    spans.push(span)
  }
  for (const call of run.toolCalls ?? []) {
    const parentSpanId = (call.parentTurnId && turnSpanIdById.get(call.parentTurnId)) || rootSpanId
    spans.push(mapToolCallSpan(call, { traceId, parentSpanId, ids }))
  }
  return spans
}

// ---------------------------------------------------------------------------
// OTLP/JSON envelope
// ---------------------------------------------------------------------------

function resourceAttributes(info: ResourceInfo = {}): OtelKeyValue[] {
  const attrs: OtelKeyValue[] = [kvString('service.name', info.serviceName || 'vontra')]
  if (info.serviceVersion) attrs.push(kvString('service.version', info.serviceVersion))
  for (const [k, v] of Object.entries(info.attributes ?? {})) {
    if (typeof v === 'string') attrs.push(kvString(k, v))
    else if (typeof v === 'boolean') attrs.push({ key: k, value: { boolValue: v } })
    else if (Number.isInteger(v)) attrs.push(kvInt(k, v))
    else attrs.push(kvDouble(k, v))
  }
  return attrs
}

/** Wrap spans in the OTLP/JSON `resourceSpans` envelope. */
export function buildOtlpPayload(spans: OtelSpan[], info: ResourceInfo = {}): OtlpTracePayload {
  return {
    resourceSpans: [
      {
        resource: { attributes: resourceAttributes(info) },
        scopeSpans: [{ scope: SCOPE, spans }],
      },
    ],
  }
}

/** Convenience: run → OTLP/JSON payload in one call. */
export function buildRunOtlpPayload(
  run: RunSpanInput,
  info: ResourceInfo = {},
  ids: IdGen = defaultIdGen,
): OtlpTracePayload {
  return buildOtlpPayload(mapRunToSpans(run, ids), info)
}

// ---------------------------------------------------------------------------
// Emitter (DI HTTP sender)
// ---------------------------------------------------------------------------

/** Minimal HTTP response shape the emitter needs — matches `fetch`'s Response. */
export interface HttpSendResponse {
  status: number
  ok: boolean
  // Error details must be exposed as a stream so the emitter can enforce a byte limit before
  // allocation. A text()-only sender is intentionally not part of the contract.
  headers?: { get(name: string): string | null }
  body?: Response['body']
}

export const OTLP_ERROR_BODY_MAX_BYTES = 16 * 1024

/** The injected sender. Defaults to `fetch` at the wire site, never here. */
export type HttpSender = (
  url: string,
  init: { method: 'POST'; headers: Record<string, string>; body: string },
) => Promise<HttpSendResponse>

export interface OtlpEmitterOptions {
  endpoint: string // full OTLP/HTTP traces URL, e.g. http://localhost:4318/v1/traces
  send: HttpSender // injected HTTP sender (DI)
  headers?: Record<string, string> // extra headers, e.g. Authorization
  resource?: ResourceInfo
  ids?: IdGen
}

export interface EmitResult {
  ok: boolean
  status: number
  spanCount: number
  error?: string
  payload: OtlpTracePayload // the exact serialized structure that was sent
}

/**
 * Emit already-mapped spans over the injected sender as OTLP/JSON. Never
 * throws: transport failures come back as `{ ok:false, error }` so callers on
 * the agent hot path can fire-and-forget without a try/catch.
 */
export async function emitSpans(spans: OtelSpan[], opts: OtlpEmitterOptions): Promise<EmitResult> {
  const payload = buildOtlpPayload(spans, opts.resource)
  const headers: Record<string, string> = { 'content-type': 'application/json', ...(opts.headers ?? {}) }
  try {
    const res = await opts.send(opts.endpoint, { method: 'POST', headers, body: JSON.stringify(payload) })
    if (res.ok) {
      // A collector can legally return a body on success. Release it so a hostile or
      // misconfigured streaming response cannot retain one undici socket per export.
      await discardResponseBody({ body: res.body ?? null })
      return { ok: true, status: res.status, spanCount: spans.length, payload }
    }
    let detail = ''
    try {
      if (res.headers && res.body) {
        detail = await readBoundedResponseText({
          headers: res.headers as Headers,
          body: res.body,
          // The helper never calls text() while a streaming body is present.
          text: async () => '',
        }, 'OTLP collector error', OTLP_ERROR_BODY_MAX_BYTES)
      }
    } catch {
      /* ignore body read failure */
    }
    return {
      ok: false,
      status: res.status,
      spanCount: spans.length,
      error: `OTLP export failed: HTTP ${res.status}${detail ? ` ${detail}` : ''}`,
      payload,
    }
  } catch (err) {
    return {
      ok: false,
      status: 0,
      spanCount: spans.length,
      error: err instanceof Error ? err.message : String(err),
      payload,
    }
  }
}

/** Map a run to spans and emit them in one call. */
export async function emitRun(run: RunSpanInput, opts: OtlpEmitterOptions): Promise<EmitResult> {
  return emitSpans(mapRunToSpans(run, opts.ids ?? defaultIdGen), opts)
}
