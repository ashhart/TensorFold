// A cheap syntax gate for edits — the "lint-protected edit" pattern SWE-agent showed lifts
// solve rate: refuse to leave a code file syntactically broken and hand the parse error back
// so the model self-corrects, instead of letting one bad patch quietly derail a whole run.
//
// SYNTAX-only, not type-checking: we use esbuild's parser (already a dep via Vite) to answer
// "does this still parse?" — fast (~ms), and it deliberately ignores TS *type* errors (those
// aren't broken syntax and blocking them would be wrong). Fails OPEN on anything it can't
// confidently judge (unknown extension, oversized file, an unexpected internal error) so the
// gate can never wedge a legitimate write.

import { extname } from 'node:path'

export interface SyntaxCheckResult {
  // Did we actually run a check? false ⇒ skipped (non-code / oversized) ⇒ caller proceeds.
  checked: boolean
  // ok:false only when we checked AND found a genuine parse error.
  ok: boolean
  error?: string
}

// esbuild loader per extension. Only these are gated; everything else is left alone.
const LOADER: Record<string, 'ts' | 'tsx' | 'js' | 'jsx' | 'json'> = {
  '.ts': 'ts', '.mts': 'ts', '.cts': 'ts',
  '.tsx': 'tsx',
  '.js': 'js', '.mjs': 'js', '.cjs': 'js',
  '.jsx': 'jsx',
  '.json': 'json',
}

// Above this we skip parsing — a generated/minified blob shouldn't add latency to the edit
// path, and a megabyte-plus source file is not the case this gate is protecting.
const MAX_GATED_BYTES = 1_000_000

/** True when a path has an extension the syntax gate knows how to parse. */
export function isGatedCodePath(path: string): boolean {
  return extname(String(path || '').toLowerCase()) in LOADER
}

/**
 * Check whether `content` parses for the file's language. Never throws. Returns
 * {checked:false, ok:true} when it doesn't apply (non-code, oversized, empty) so the caller
 * can treat "not checked" and "checked-and-fine" the same for the proceed decision while
 * still being able to surface a real {checked:true, ok:false} parse failure.
 */
export async function checkEditedSyntax(path: string, content: string): Promise<SyntaxCheckResult> {
  const ext = extname(String(path || '').toLowerCase())
  const loader = LOADER[ext]
  if (!loader) return { checked: false, ok: true }
  const text = content ?? ''
  if (!text.trim()) return { checked: false, ok: true } // empty / clearing a file is fine
  if (text.length > MAX_GATED_BYTES) return { checked: false, ok: true }

  if (loader === 'json') {
    try { JSON.parse(text); return { checked: true, ok: true } }
    catch (e: any) { return { checked: true, ok: false, error: `JSON parse error: ${e?.message ?? e}` } }
  }

  try {
    // Lazy import so loading this module never forces esbuild in environments that don't edit.
    const esbuild = await import('esbuild')
    await esbuild.transform(text, { loader, logLevel: 'silent' })
    return { checked: true, ok: true }
  } catch (e: any) {
    // esbuild throws a TransformError carrying an `errors` array on a parse failure. Only treat
    // a real parse error as a block; if the shape is unexpected, fail OPEN (don't wedge writes).
    const errs = e?.errors
    if (Array.isArray(errs) && errs.length) {
      const first = errs[0]
      const loc = first?.location ? ` (line ${first.location.line})` : ''
      return { checked: true, ok: false, error: `Syntax error${loc}: ${first?.text ?? 'could not parse'}` }
    }
    return { checked: false, ok: true }
  }
}

/**
 * Format the standard tool-output line appended/returned when the gate trips, telling the
 * model exactly what to fix. Kept here (pure) so the message is consistent across the
 * write_file / edit_file / apply_patch call sites and unit-testable.
 */
export function syntaxGateMessage(path: string, error: string): string {
  return `Syntax check failed for ${path}: ${error}. The edit was NOT applied — fix the syntax and try again.`
}
