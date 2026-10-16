import { app } from 'electron'
import { promises as fs } from 'node:fs'
import { randomBytes } from 'node:crypto'
import path from 'node:path'

// VONTRA_HOME overrides the data dir (lets you relocate config, e.g. ~/.vontrastudio, and enables tests).
export const vontraHomeDir = () => process.env.VONTRA_HOME?.trim() || path.join(app.getPath('home'), '.vontra')
export const vontraPath = (...parts: string[]) => path.join(vontraHomeDir(), ...parts)
// No legacy Electron-userData migration when the data dir is relocated via VONTRA_HOME, or when the
// Electron app object isn't available (tests). Returns '' → readSettingsJson treats it as absent.
const legacyUserDataDir = () => {
  if (process.env.VONTRA_HOME?.trim()) return ''
  try { return app.getPath('userData') } catch { return '' }
}
const legacyPath = (name: string) => { const base = legacyUserDataDir(); return base ? path.join(base, name) : '' }
const statePath = (name: string) => vontraPath('state', name)
const settingsPath = () => vontraPath('settings.json')
const modelsPath = () => vontraPath('models.json')
const legacySettingsPath = () => legacyPath('settings.json')
const threadsPath = () => statePath('threads.json')
export const LEGACY_STATE_FILES = [
  'settings.json',
  'threads.json',
  'projects.json',
  'rooms.json',
  'tickets.json',
  'automations.json',
  'assistant-events.json',
  'missions.json',
  'jobs.json',
  'subagents.json',
  'skills.json',
  'commands.json',
  'project-memory.json',
  'mcp-servers.json',
  'webhooks.json',
  'incidents.json',
  'bot-control-sessions.json',
  'model-scorecards.json',
]

export function safeSegment(value: string): string {
  return String(value || 'item')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80) || 'item'
}

export function makeRemoteControlToken(): string {
  return randomBytes(32).toString('base64url')
}

async function readJson<T>(p: string, fallback: T): Promise<T> {
  try {
    const raw = await fs.readFile(p, 'utf8')
    return JSON.parse(raw) as T
  } catch {
    return fallback
  }
}

async function ensurePrivateDir(p: string): Promise<void> {
  await fs.mkdir(p, { recursive: true, mode: 0o700 })
  await fs.chmod(p, 0o700).catch(() => {})
}

export async function ensureTensorFoldStudioHome(): Promise<void> {
  await Promise.all([
    ensurePrivateDir(vontraHomeDir()),
    ensurePrivateDir(vontraPath('state')),
    ensurePrivateDir(vontraPath('archives', 'projects')),
    ensurePrivateDir(vontraPath('archives', 'threads')),
    ensurePrivateDir(vontraPath('archives', 'rooms')),
    ensurePrivateDir(vontraPath('archives', 'tickets')),
    ensurePrivateDir(vontraPath('rooms')),
  ])
}

async function fileExists(p: string): Promise<boolean> {
  try {
    const stat = await fs.stat(p)
    return stat.isFile()
  } catch {
    return false
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}

// Re-exported so store.ts (and other store submodules) can reuse the same helpers without
// redefining them. These are not part of the original public API but are shared internally.
export {
  legacyUserDataDir,
  legacyPath,
  statePath,
  settingsPath,
  modelsPath,
  legacySettingsPath,
  threadsPath,
  readJson,
  ensurePrivateDir,
  fileExists,
  isRecord,
}
