// Qualification may lower the model budget; no environment value can raise physical memory.
export function loweredEngineMemoryBudget(physicalBytes: number, requested: unknown): number {
  const value = typeof requested === 'string' && /^[1-9]\d{0,15}$/.test(requested) ? Number(requested) : requested
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0 ? Math.min(physicalBytes, value) : physicalBytes
}
