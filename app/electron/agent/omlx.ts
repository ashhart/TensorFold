import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import type { OmlxDiscovery } from './types'

function isRecord(value: unknown): value is Record<string, any> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}

async function readJson(file: string): Promise<Record<string, any> | null> {
  try {
    const parsed = JSON.parse(await fs.readFile(file, 'utf8'))
    return isRecord(parsed) ? parsed : null
  } catch {
    return null
  }
}

function localHost(host: unknown): string {
  const value = String(host || '').trim().toLowerCase()
  if (!value || value === '0.0.0.0' || value === '::' || value === '[::]') return '127.0.0.1'
  return value
}

// Wrap bare IPv6 literals in brackets so they interpolate into a valid URL
// authority. Without this, `http://${host}:${port}/v1` for an IPv6 host like
// '::1' yields 'http://::1:9000/v1', which throws 'Invalid URL' in new URL().
function urlHost(host: string): string {
  return host.includes(':') && !host.startsWith('[') ? `[${host}]` : host
}

function modelList(settings: Record<string, any> | null, stats: Record<string, any> | null): { models: string[]; defaultModel: string | null } {
  const configured = isRecord(settings?.models) ? settings.models : {}
  const names = new Set<string>()
  let defaultModel: string | null = null
  for (const [name, config] of Object.entries(configured)) {
    names.add(name)
    if (isRecord(config)) {
      if (typeof config.model_alias === 'string' && config.model_alias.trim()) names.add(config.model_alias.trim())
      if (config.is_default === true && !defaultModel) defaultModel = name
    }
  }
  if (isRecord(stats?.per_model)) {
    for (const name of Object.keys(stats.per_model)) names.add(name)
  }
  return { models: Array.from(names).sort(), defaultModel }
}

export async function discoverOmlx(root = path.join(os.homedir(), '.omlx')): Promise<OmlxDiscovery> {
  const configPath = path.join(root, 'settings.json')
  const settings = await readJson(configPath)
  if (!settings) {
    return {
      found: false,
      baseUrl: 'http://127.0.0.1:9000/v1',
      apiKey: '',
      models: [],
      defaultModel: null,
      configPath,
      message: 'No oMLX settings.json found.',
    }
  }

  const server = isRecord(settings.server) ? settings.server : {}
  const auth = isRecord(settings.auth) ? settings.auth : {}
  const port = Number(server.port || 9000)
  const host = localHost(server.host)
  const apiKey = typeof auth.api_key === 'string' && auth.api_key.trim()
    ? auth.api_key.trim()
    : typeof auth.secret_key === 'string' && auth.secret_key.trim()
      ? auth.secret_key.trim()
      : ''
  const models = modelList(
    await readJson(path.join(root, 'model_settings.json')),
    await readJson(path.join(root, 'stats.json')),
  )

  return {
    found: true,
    baseUrl: `http://${urlHost(host)}:${Number.isFinite(port) && port > 0 ? port : 9000}/v1`,
    apiKey,
    models: models.models,
    defaultModel: models.defaultModel,
    configPath,
    message: apiKey ? `Discovered oMLX on port ${Number.isFinite(port) && port > 0 ? port : 9000}.` : 'Discovered oMLX, but no API key was configured.',
  }
}
