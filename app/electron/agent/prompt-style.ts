export const VONTRA_STYLE_PROMPT = [
  'TensorFold style contract:',
  '- Do not use em dashes in prose, summaries, commit messages, or code comments. Use commas, colons, parentheses, or short sentences instead.',
  '- Keep code comments rare, useful, and one line only.',
  '- If a comment cannot fit on one line, do not add it. Prefer clearer names or smaller code.',
  '- Do not add obvious, decorative, or bloated comments.',
].join('\n')

export function ensureTensorFoldStudioStylePrompt(prompt: string): string {
  return prompt.includes('TensorFold style contract:')
    ? prompt
    : `${prompt.trimEnd()}\n\n${VONTRA_STYLE_PROMPT}`
}
