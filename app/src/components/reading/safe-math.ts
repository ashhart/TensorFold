import katex from 'katex'

export const MATH_LIMITS = { input: 4096, output: 250_000, expansions: 200, size: 20 } as const
export interface MathRenderer { renderToString(input: string, options: Record<string, unknown>): string }
export type MathResult = { ok: true; html: string } | { ok: false; reason: string }

export function renderBoundedMath(expression: string, displayMode = false, renderer: MathRenderer = katex): MathResult {
  if (!expression.trim()) return { ok: false, reason: 'Empty formula.' }
  if (expression.length > MATH_LIMITS.input) return { ok: false, reason: 'Formula is too large to render.' }
  // Reject extension/macro controls before invoking the renderer. Each formula
  // gets a fresh macro object, so one message cannot change another formula.
  if (/\\(?:html\w*|href|url|includegraphics|def|gdef|edef|xdef|newcommand|renewcommand|let|futurelet|catcode|input|include|require)\b/i.test(expression)) return { ok: false, reason: 'This formula uses an unsupported command.' }
  try {
    const html = renderer.renderToString(expression, { displayMode, throwOnError: true, trust: false, strict: 'error', maxExpand: MATH_LIMITS.expansions, maxSize: MATH_LIMITS.size, macros: {}, output: 'htmlAndMathml', globalGroup: false })
    if (html.length > MATH_LIMITS.output || /<(?:script|iframe|object|embed|img|a|foreignObject|video|audio|source|link|style)\b|\bon\w+\s*=|(?:src|href)\s*=/i.test(html)) return { ok: false, reason: 'Formula output exceeded the safe rendering limits.' }
    return { ok: true, html }
  } catch { return { ok: false, reason: 'Formula could not be rendered. The source is shown.' } }
}
