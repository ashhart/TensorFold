// Static renderer serving for the remote control server (VST-035): the SAME built
// renderer bundle (dist/) is served from the token-gated loopback/LAN server so a paired
// browser or phone loads TensorFold straight from the host it drives.
//
// Security posture (explicit, per the DoD): the STATIC SHELL is public on the bound
// interface — it contains no user data, only the app code, exactly like any hosted web
// app. Every DATA endpoint (/v1/*) stays behind the owner/device-session token gate; a
// fresh browser renders the pairing screen until it redeems a credential.

import { promises as nodeFs } from 'node:fs'
import path from 'node:path'

export interface StaticFile {
  content: Buffer
  contentType: string
  // index.html must revalidate (it names the hashed assets); hashed assets are immutable.
  cacheControl: string
}

export interface StaticFs {
  readFile(filePath: string): Promise<Buffer>
  stat(filePath: string): Promise<{ isFile(): boolean }>
  lstat(filePath: string): Promise<{ isFile(): boolean }>
}

const CONTENT_TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.map': 'application/json',
  '.txt': 'text/plain; charset=utf-8',
  '.wasm': 'application/wasm',
}

export function staticContentType(filePath: string): string {
  return CONTENT_TYPES[path.extname(filePath).toLowerCase()] ?? 'application/octet-stream'
}

// Resolve a request path inside the static root, or null when it escapes/doesn't apply.
// Only GETs for non-/v1 paths reach this; anything outside the root (.. traversal,
// absolute smuggling, encoded separators) resolves to null, never to a file.
export function resolveStaticPath(root: string, requestPath: string): string | null {
  if (!root) return null
  let decoded: string
  try {
    decoded = decodeURIComponent(requestPath)
  } catch {
    return null
  }
  if (decoded.includes('\0')) return null
  const relative = decoded.replace(/^\/+/, '')
  const resolved = path.resolve(root, relative === '' ? 'index.html' : relative)
  const rootResolved = path.resolve(root)
  if (resolved !== rootResolved && !resolved.startsWith(rootResolved + path.sep)) return null
  return resolved
}

/**
 * Load a static file for the request path, with the SPA fallback: an unknown extensionless
 * path (a client-side route like /settings) serves index.html so a paired browser can deep
 * link. Returns null when serving is disabled (no root) or nothing matches.
 */
export async function loadStaticFile(root: string, requestPath: string, fs: StaticFs = nodeFs): Promise<StaticFile | null> {
  const resolved = resolveStaticPath(root, requestPath)
  if (!resolved) return null
  const tryRead = async (filePath: string): Promise<StaticFile | null> => {
    try {
      // lstat (no-follow): a symlink planted inside the served root must not read its
      // target — the shell is pre-auth by design, so the file jail has to be absolute.
      const stat = await fs.lstat(filePath)
      if (!stat.isFile()) return null
      const content = await fs.readFile(filePath)
      const isIndex = path.basename(filePath) === 'index.html'
      // Vite emits content-hashed asset filenames — safe to cache hard. index.html points
      // at them and must always revalidate so a host upgrade reaches paired clients.
      const hashed = /\.[0-9a-f]{8,}\./i.test(path.basename(filePath)) || filePath.includes(`${path.sep}assets${path.sep}`)
      return {
        content,
        contentType: staticContentType(filePath),
        cacheControl: isIndex ? 'no-cache' : hashed ? 'public, max-age=31536000, immutable' : 'no-cache',
      }
    } catch {
      return null
    }
  }
  const direct = await tryRead(resolved)
  if (direct) return direct
  // SPA fallback: only for paths that look like client routes, never for missed assets
  // (a missing .js must 404, not silently return HTML).
  if (!path.extname(resolved)) return tryRead(path.resolve(root, 'index.html'))
  return null
}
