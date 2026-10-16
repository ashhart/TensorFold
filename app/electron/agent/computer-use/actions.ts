// The computer_use tool's input and output, kept pure so the main-process runner and tests share it.

export const COMPUTER_ACTIONS = ['observe', 'open_app', 'click', 'double_click', 'right_click', 'type', 'key', 'scroll', 'drag', 'wait'] as const
export type ComputerAction = (typeof COMPUTER_ACTIONS)[number]

export interface ComputerPoint { x: number; y: number }

export interface ComputerUseInput {
  action: ComputerAction
  // The app the action is for; the runner refuses unless it is the one in front (open_app brings it there).
  app: string
  element?: number
  x?: number
  y?: number
  to?: ComputerPoint
  text?: string
  keys?: string
  dx?: number
  dy?: number
  seconds?: number
  note?: string
}

export interface ComputerFrame { x: number; y: number; width: number; height: number }

export interface ComputerElement {
  id: number
  role: string
  subrole?: string
  title?: string
  value?: string
  description?: string
  frame: ComputerFrame
  enabled: boolean
  focused: boolean
  secure: boolean
}

export interface ComputerApp { name: string; bundleId: string | null; pid: number }

export interface ComputerObservation {
  app: ComputerApp
  window: { title: string; frame: ComputerFrame } | null
  screen?: { width: number; height: number; scale: number }
  elements: ComputerElement[]
}

const OWN_APP_NAMES = new Set(['tensorfold', 'tensorfold studio', 'electron'])
const MAX_TEXT = 4000
const POINT_ACTIONS = new Set<ComputerAction>(['click', 'double_click', 'right_click', 'scroll'])

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
const text = (value: unknown): string => typeof value === 'string' ? value.trim() : ''

function point(value: unknown, label: string): ComputerPoint {
  const raw = value as Record<string, unknown> | null
  if (!raw || !finite(raw.x) || !finite(raw.y)) throw new Error(`computer_use ${label} needs numeric x and y.`)
  return { x: Math.round(raw.x), y: Math.round(raw.y) }
}

/** Validates a model's computer_use call; every message says how to fix the call. */
export function normalizeComputerUseInput(args: unknown): ComputerUseInput {
  const raw = (args && typeof args === 'object' ? args : {}) as Record<string, unknown>
  const action = text(raw.action) as ComputerAction
  if (!COMPUTER_ACTIONS.includes(action)) throw new Error(`computer_use action must be one of: ${COMPUTER_ACTIONS.join(', ')}.`)
  const app = text(raw.app)
  if (!app) throw new Error('computer_use needs app: the name of the Mac app this action is for, for example "Safari".')
  if (OWN_APP_NAMES.has(app.toLowerCase())) throw new Error('computer_use cannot operate TensorFold itself.')
  const input: ComputerUseInput = { action, app }
  const note = text(raw.note)
  if (note) input.note = note.slice(0, 200)
  if (POINT_ACTIONS.has(action)) {
    if (finite(raw.element)) input.element = Math.round(raw.element)
    else if (finite(raw.x) && finite(raw.y)) Object.assign(input, point(raw, action))
    else if (action !== 'scroll') throw new Error(`computer_use ${action} needs element (an id from observe) or x and y.`)
  }
  if (action === 'scroll') {
    input.dx = finite(raw.dx) ? Math.max(-50, Math.min(50, Math.round(raw.dx))) : 0
    input.dy = finite(raw.dy) ? Math.max(-50, Math.min(50, Math.round(raw.dy))) : 0
    if (!input.dx && !input.dy) throw new Error('computer_use scroll needs dx or dy in lines (negative dy scrolls up).')
  }
  if (action === 'drag') {
    input.x = point(raw, 'drag start').x
    input.y = point(raw, 'drag start').y
    input.to = point(raw.to, 'drag to')
  }
  if (action === 'type') {
    if (typeof raw.text !== 'string' || !raw.text.length) throw new Error('computer_use type needs text.')
    if (raw.text.length > MAX_TEXT) throw new Error(`computer_use type takes at most ${MAX_TEXT} characters a call.`)
    input.text = raw.text
  }
  if (action === 'key') {
    const keys = text(raw.keys)
    if (!keys) throw new Error('computer_use key needs keys, for example "cmd+l" or "return".')
    input.keys = keys
  }
  if (action === 'wait') input.seconds = finite(raw.seconds) ? Math.max(0.1, Math.min(10, raw.seconds)) : 1
  return input
}

/** True when an app name or bundle id from the model names the given running app. */
export function matchesApp(wanted: string, app: ComputerApp): boolean {
  const target = wanted.trim().toLowerCase()
  return target === app.name.toLowerCase() || (!!app.bundleId && target === app.bundleId.toLowerCase())
}

const centre = (frame: ComputerFrame) => `${Math.round(frame.x + frame.width / 2)},${Math.round(frame.y + frame.height / 2)}`
const quote = (value: string) => JSON.stringify(value.length > 120 ? `${value.slice(0, 117)}...` : value)

/** The observation as compact lines: one per control, "[id] role "label" value=... at x,y". */
export function formatObservation(obs: ComputerObservation): string {
  const lines = [`App: ${obs.app.name}${obs.app.bundleId ? ` (${obs.app.bundleId})` : ''}`]
  lines.push(obs.window ? `Window: ${quote(obs.window.title || 'untitled')}` : 'Window: none')
  if (obs.screen) lines.push(`Screen: ${obs.screen.width}x${obs.screen.height} points`)
  if (!obs.elements.length) lines.push('No controls found in this window. Try open_app, wait, then observe again.')
  else lines.push(`Controls (${obs.elements.length}); act on them with element ids:`)
  for (const el of obs.elements) {
    const label = el.title || el.description || ''
    const parts = [`[${el.id}] ${el.subrole && el.subrole !== el.role ? `${el.role}/${el.subrole}` : el.role}`]
    if (label) parts.push(quote(label))
    if (el.value && !el.secure) parts.push(`value=${quote(el.value)}`)
    if (el.secure) parts.push('secure')
    if (el.focused) parts.push('focused')
    if (!el.enabled) parts.push('disabled')
    parts.push(`at ${centre(el.frame)}`)
    lines.push(parts.join(' '))
  }
  return lines.join('\n')
}

/** What the timeline and approval card call this action. */
export function describeComputerAction(input: ComputerUseInput): string {
  const where = input.element != null ? ` element ${input.element}` : input.x != null && input.y != null ? ` at ${input.x},${input.y}` : ''
  switch (input.action) {
    case 'observe': return `Read ${input.app}'s window`
    case 'open_app': return `Open ${input.app}`
    case 'type': return `Type ${input.text!.length} characters in ${input.app}`
    case 'key': return `Press ${input.keys} in ${input.app}`
    case 'scroll': return `Scroll ${input.app}${where}`
    case 'drag': return `Drag in ${input.app} from ${input.x},${input.y} to ${input.to!.x},${input.to!.y}`
    case 'wait': return `Wait ${input.seconds} s`
    default: return `${input.action.replace('_', ' ')} in ${input.app}${where}`
  }
}
