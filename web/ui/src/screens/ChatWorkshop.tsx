// The Chat's workshop: tools the model may call, their calls checked against their schemas, results given back,
// and replies held to JSON or to a JSON schema, checked as they arrive.
import { useState } from 'react'
import { type Schema } from '../schema'
import { Field } from './ChatPresets'
export { check, checkJson, unchecked, type Schema } from '../schema'

export type ToolDef = { type: 'function'; function: { name: string; description?: string; parameters?: Schema } }
// `choice`: the request's tool_choice, 'auto', 'required', 'none' or a tool's name; `oneCall`: parallel_tool_calls false
export type Workshop = { tools: boolean; toolsText: string; auto: boolean; output: 'text' | 'json' | 'schema'; schemaText: string; choice?: string; oneCall?: boolean }

/** A forced tool that the new tool set no longer has goes back to 'auto'. */
function keepChoice(choice: string | undefined, tools: ToolDef[]): string | undefined {
  return !choice || ['auto', 'required', 'none'].includes(choice) || tools.some((t) => t.function.name === choice) ? choice : 'auto'
}

/** The request's tool_choice: forced only on the first round, so a reply can follow the tools' results. */
export function toolChoice(ws: Workshop, round: number): string | { type: 'function'; function: { name: string } } {
  const c = ws.choice ?? 'auto'
  if (round > 0 && c !== 'none') return 'auto'
  return c === 'auto' || c === 'required' || c === 'none' ? c : { type: 'function', function: { name: c } }
}

/** Tools that run in this page, so a chain of calls can go on by itself. */
const LOCAL: Record<string, (args: Record<string, unknown>) => unknown> = {
  calculator: (a) => {
    const expr = String(a.expression ?? '')
    if (!/^[\d\s+\-*/().,%^]+$/.test(expr)) return { error: 'only numbers and + - * / ( ) % ^' }
    // the expression holds numbers and operators only, so evaluating it runs no other code
    const value = Function(`"use strict"; return (${expr.replace(/\^/g, '**').replace(/,/g, '.')})`)() as number
    return { expression: expr, value }
  },
  current_time: (a) => ({ timezone: a.timezone ?? Intl.DateTimeFormat().resolvedOptions().timeZone,
    time: new Date().toLocaleString('en-GB', { timeZone: typeof a.timezone === 'string' ? a.timezone : undefined }) }),
  random_number: (a) => { const lo = Number(a.min ?? 1), hi = Number(a.max ?? 100); return { value: Math.floor(lo + Math.random() * (hi - lo + 1)) } },
}

export const TOOL_PRESETS: { name: string; tools: ToolDef[] }[] = [
  { name: 'Calculator, time, random (run here)', tools: [
    { type: 'function', function: { name: 'calculator', description: 'Evaluate an arithmetic expression exactly.', parameters: { type: 'object', properties: { expression: { type: 'string', description: 'For example (12.5 * 4) / 3' } }, required: ['expression'], additionalProperties: false } } },
    { type: 'function', function: { name: 'current_time', description: 'The current date and time in a time zone.', parameters: { type: 'object', properties: { timezone: { type: 'string', description: 'An IANA zone such as Europe/Paris' } }, additionalProperties: false } } },
    { type: 'function', function: { name: 'random_number', description: 'A random whole number between min and max.', parameters: { type: 'object', properties: { min: { type: 'integer' }, max: { type: 'integer' } }, required: ['min', 'max'], additionalProperties: false } } },
  ] },
  { name: 'Weather and email (you answer)', tools: [
    { type: 'function', function: { name: 'get_weather', description: 'Current weather for a city.', parameters: { type: 'object', properties: { city: { type: 'string' }, unit: { type: 'string', enum: ['celsius', 'fahrenheit'] } }, required: ['city'], additionalProperties: false } } },
    { type: 'function', function: { name: 'send_email', description: 'Send an email.', parameters: { type: 'object', properties: { to: { type: 'string' }, subject: { type: 'string' }, body: { type: 'string' } }, required: ['to', 'subject', 'body'], additionalProperties: false } } },
  ] },
]
export const SCHEMA_EXAMPLE = JSON.stringify({ type: 'object', properties: { city: { type: 'string' }, country: { type: 'string' }, population: { type: 'integer' } }, required: ['city', 'country', 'population'], additionalProperties: false }, null, 2)
export const DEFAULT_WORKSHOP: Workshop = { tools: false, toolsText: JSON.stringify(TOOL_PRESETS[0].tools, null, 2), auto: true, output: 'text', schemaText: SCHEMA_EXAMPLE }

export function parseTools(text: string): { tools: ToolDef[] | null; error: string | null } {
  try {
    const v = JSON.parse(text) as ToolDef[]
    if (!Array.isArray(v) || !v.every((t) => t?.type === 'function' && typeof t.function?.name === 'string')) return { tools: null, error: 'a list of {"type": "function", "function": {"name", "parameters"}}' }
    return { tools: v, error: null }
  } catch (e) { return { tools: null, error: `not JSON: ${(e as Error).message}` } }
}

export const isLocal = (name: string) => name in LOCAL

export function runLocal(name: string, args: Record<string, unknown>): unknown | undefined {
  const f = LOCAL[name]
  if (!f) return undefined
  try { return f(args) } catch (e) { return { error: String(e) } }
}

const SET_HELP = [
  <>The page runs these itself. Try “What time is it in Tokyo?” or “What is 17.5% of 2,340?”.</>,
  <>Made-up tools: when the model calls one, you type the result (“18 °C, cloudy”) and see what it does with it. No email is sent.</>,
]
const FORMAT_HELP: Record<Workshop['output'], string> = {
  text: 'Ordinary text, as in any conversation.',
  json: 'One JSON object and nothing else; each reply says whether it is valid JSON.',
  schema: 'JSON that must match the schema below; each reply says whether it does, and where not.',
}

/** Tools & JSON, in two panels like the conversation's settings: the tools the model may call, and the reply's
 *  format; the definitions and the wire details folded, for whoever wants them. */
export function WorkshopPanel({ ws, set }: { ws: Workshop; set: (w: Workshop) => void }) {
  const parsed = parseTools(ws.toolsText)
  let schemaError: string | null = null
  if (ws.output === 'schema') { try { JSON.parse(ws.schemaText) } catch (e) { schemaError = (e as Error).message } }
  const [preset, setPreset] = useState(0)
  const custom = ws.toolsText !== JSON.stringify(TOOL_PRESETS[preset].tools, null, 2)
  return (
    <div className="presetbox workshop">
      <div className="phead"><b>Tools &amp; JSON</b><span className="small muted">Test what an app asks of the model: calling tools, answering in JSON. With either on, replies come whole rather than word by word, and each one is checked.</span></div>
      <div className="pgroups">
        <section className="pgroup" aria-label="Tools">
          <h4>Tools the model may call</h4>
          <label className="pfield"><span><input type="checkbox" checked={ws.tools} onChange={(e) => set({ ...ws, tools: e.target.checked })} /> <b>Offer tools</b></span>
            <span className="small muted">The model may ask for a tool instead of answering; the page checks the call, gives the result back, and the model answers from it.</span></label>
          {ws.tools && <>
            <Field label="Tool set" help={custom ? 'Your own definitions, edited below.' : SET_HELP[preset]}>
              <select className="sel sm" value={preset} aria-label="Tool set" onChange={(e) => { const i = Number(e.target.value); setPreset(i); set({ ...ws, toolsText: JSON.stringify(TOOL_PRESETS[i].tools, null, 2), choice: keepChoice(ws.choice, TOOL_PRESETS[i].tools) }) }}>
                {TOOL_PRESETS.map((p, i) => <option key={p.name} value={i}>{p.name}</option>)}
              </select></Field>
            <Field label="Calls" help="Whether the model decides, must call at least one tool, may call none, or must call a given one. Forcing applies to the first round only, so it can then answer from the results.">
              <select className="sel sm" value={ws.choice ?? 'auto'} aria-label="Tool choice" onChange={(e) => set({ ...ws, choice: e.target.value })}>
                <option value="auto">as the model sees fit</option><option value="required">at least one, required</option><option value="none">none</option>
                {(parsed.tools ?? []).map((t) => <option key={t.function.name} value={t.function.name}>{t.function.name}, forced</option>)}</select></Field>
            <label className="pfield"><span><input type="checkbox" checked={!!ws.oneCall} onChange={(e) => set({ ...ws, oneCall: e.target.checked })} /> <b>One call at most</b></span>
              <span className="small muted">One tool call per round, instead of several at once.</span></label>
            <label className="pfield"><span><input type="checkbox" checked={ws.auto} onChange={(e) => set({ ...ws, auto: e.target.checked })} /> <b>Run the page's tools by themselves</b></span>
              <span className="small muted">The calculator, the time and the random number run at once; unticked, the page waits at each call and you type the result yourself.</span></label>
            <details className="pmore" open={!!parsed.error}>
              <summary className="small">Tool definitions (OpenAI format){parsed.error ? <span className="bad"> {parsed.error}</span> : <span className="muted"> {parsed.tools!.length} tools</span>}</summary>
              <p className="small muted" style={{ margin: '0 0 6px' }}>Describe your own tools here to see whether the model calls them right before plugging them into an app.</p>
              <textarea className="input mono" rows={8} value={ws.toolsText} onChange={(e) => { const p = parseTools(e.target.value); set({ ...ws, toolsText: e.target.value, choice: p.tools ? keepChoice(ws.choice, p.tools) : ws.choice }) }} spellCheck={false} aria-label="Tool definitions" />
            </details>
          </>}
        </section>
        <section className="pgroup" aria-label="Reply format">
          <h4>Reply format</h4>
          <Field label="Reply as" help={FORMAT_HELP[ws.output]}>
            <select className="sel sm" value={ws.output} onChange={(e) => set({ ...ws, output: e.target.value as Workshop['output'] })} aria-label="Reply format">
              <option value="text">text</option><option value="json">JSON</option><option value="schema">JSON matching a schema</option></select></Field>
          {ws.output === 'schema' && <label className="pfield"><span><b>JSON schema</b>{schemaError ? <span className="small bad"> not JSON: {schemaError}</span> : null}</span>
            <textarea className="input mono" rows={8} value={ws.schemaText} onChange={(e) => set({ ...ws, schemaText: e.target.value })} spellCheck={false} aria-label="JSON schema" /></label>}
          <details className="pmore">
            <summary className="small">How it is sent</summary>
            <p className="small muted" style={{ margin: 0 }}>As <code>tools</code>, <code>tool_choice</code> and <code>parallel_tool_calls</code> for the tools, <code>response_format</code> for JSON. An engine that refuses <code>response_format</code> is asked for JSON in the system prompt instead, and the reply is still checked.</p>
          </details>
        </section>
      </div>
    </div>
  )
}
