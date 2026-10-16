export const CONTROLLER_ACTIONS = ['activate_app', 'keystroke', 'menu_item'] as const

export type ControllerActionKind = typeof CONTROLLER_ACTIONS[number]
export type ControllerModifier = 'command' | 'shift' | 'option' | 'control'

export interface ControllerActionInput {
  action: ControllerActionKind
  appName: string
  keystroke?: string
  modifiers?: ControllerModifier[]
  menuPath?: string[]
  note?: string
}

export interface ControllerActionResult {
  ok: boolean
  action: ControllerActionKind
  appName: string
  target: string
  executedAt: number
  note?: string
}

const ACTIONS = new Set<string>(CONTROLLER_ACTIONS)
const MODIFIERS = new Set<string>(['command', 'shift', 'option', 'control'])

export function normalizeControllerActionInput(input: Record<string, unknown>, options: { ownAppName?: string } = {}): ControllerActionInput {
  const action = String(input.action ?? '').trim().toLowerCase() as ControllerActionKind
  if (!ACTIONS.has(action)) throw new Error(`controller_action action must be one of: ${CONTROLLER_ACTIONS.join(', ')}.`)

  const appName = cleanText(input.appName, 80)
  if (!appName) throw new Error('controller_action requires appName.')
  const ownAppName = cleanText(options.ownAppName, 80).toLowerCase()
  const ownNames = ['tensorfold', 'tensorfold studio'].includes(ownAppName)
    ? ['tensorfold', 'tensorfold studio'] : [ownAppName]
  if (ownAppName && ownNames.includes(appName.toLowerCase())) throw new Error(`controller_action cannot target ${options.ownAppName}.`)
  if (['vontra', 'electron'].includes(appName.toLowerCase())) throw new Error(`controller_action cannot target ${appName}.`)

  const normalized: ControllerActionInput = {
    action,
    appName,
    note: cleanOptional(input.note, 240),
  }

  if (action === 'keystroke') {
    normalized.keystroke = cleanText(input.keystroke, 16)
    if (!normalized.keystroke) throw new Error('controller_action keystroke requires keystroke.')
    normalized.modifiers = normalizeModifiers(input.modifiers)
  }

  if (action === 'menu_item') {
    normalized.menuPath = normalizeMenuPath(input.menuPath)
    if (normalized.menuPath.length < 2) throw new Error('controller_action menu_item requires menuPath with menu and item.')
  }

  return stripEmpty(normalized)
}

export function controllerActionTarget(input: ControllerActionInput): string {
  if (input.action === 'keystroke') {
    const prefix = input.modifiers?.length ? `${input.modifiers.map((m) => humanModifier(m)).join('+')}+` : ''
    return `${prefix}${input.keystroke ?? ''}`
  }
  if (input.action === 'menu_item') return input.menuPath?.join(' > ') ?? ''
  return input.appName
}

export function controllerActionScript(input: ControllerActionInput): string {
  const app = appleScriptString(input.appName)
  const activate = `tell application ${app} to activate`
  if (input.action === 'activate_app') return activate
  const body = input.action === 'keystroke'
    ? `keystroke ${appleScriptString(input.keystroke ?? '')}${input.modifiers?.length ? ` using {${input.modifiers.map((m) => `${m} down`).join(', ')}}` : ''}`
    : `click ${menuReference(input.menuPath ?? [])}`
  return [
    activate,
    'delay 0.2',
    'tell application "System Events"',
    `  tell process ${app}`,
    `    ${body}`,
    '  end tell',
    'end tell',
  ].join('\n')
}

export function formatControllerActionResult(result: ControllerActionResult): string {
  return [
    `Action: ${result.action}`,
    `App: ${result.appName}`,
    `Target: ${result.target}`,
    `Executed: ${new Date(result.executedAt).toLocaleString()}`,
    'Approval: controller_action tool approval was required before this Mac app action.',
    result.note ? `Note: ${result.note}` : '',
  ].filter(Boolean).join('\n')
}

function menuReference(path: string[]): string {
  if (path.length < 2) throw new Error('menuPath requires at least a menu and menu item.')
  let parent = `menu ${appleScriptString(path[0])} of menu bar 1`
  for (let index = 1; index < path.length; index++) {
    const item = `menu item ${appleScriptString(path[index])} of ${parent}`
    if (index === path.length - 1) return item
    parent = `menu ${appleScriptString(path[index])} of ${item}`
  }
  return parent
}

function normalizeModifiers(value: unknown): ControllerModifier[] {
  const raw = Array.isArray(value)
    ? value
    : typeof value === 'string'
      ? value.split(/[\s,+]+/)
      : []
  const out: ControllerModifier[] = []
  for (const item of raw) {
    const modifier = String(item ?? '').trim().toLowerCase()
    if (MODIFIERS.has(modifier) && !out.includes(modifier as ControllerModifier)) out.push(modifier as ControllerModifier)
  }
  return out
}

function normalizeMenuPath(value: unknown): string[] {
  const raw = Array.isArray(value)
    ? value
    : typeof value === 'string'
      ? value.split('>')
      : []
  return raw.map((item) => cleanText(item, 80)).filter(Boolean).slice(0, 4)
}

function cleanText(value: unknown, max: number): string {
  return String(value ?? '').replace(/\s+/g, ' ').trim().slice(0, max)
}

function cleanOptional(value: unknown, max: number): string | undefined {
  const text = cleanText(value, max)
  return text || undefined
}

function appleScriptString(value: string): string {
  return `"${String(value ?? '').replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`
}

function humanModifier(value: ControllerModifier): string {
  return value[0].toUpperCase() + value.slice(1)
}

function stripEmpty<T extends Record<string, any>>(value: T): T {
  const copy: Record<string, any> = {}
  for (const [key, item] of Object.entries(value)) {
    if (item !== undefined && item !== null && item !== '') copy[key] = item
  }
  return copy as T
}
