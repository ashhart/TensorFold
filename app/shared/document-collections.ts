/** Absent is legacy enabled-collection behavior; an empty list grants no collection access. */
export function validateDocumentSourceIds(value: unknown): string[] | undefined {
  if (value === undefined) return undefined
  if (!Array.isArray(value) || value.length > 64 || value.some(id => typeof id !== 'string' || !id.length || id.length > 256 || /[\u0000-\u001f\u007f]/.test(id)) || new Set(value).size !== value.length) throw new Error('Choose up to 64 distinct document collections.')
  return [...value] as string[]
}
export function documentPolicySignature(sourceIds?: readonly string[]): string { return sourceIds === undefined ? 'legacy-enabled' : JSON.stringify([...sourceIds].sort()) }
