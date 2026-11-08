// TensorFold release numbers, compared as numbers ("1.0.9" comes before "1.0.11")


export function before(v: string | undefined | null, want: number[]): boolean {
  const got = (v ?? '').match(/\d+/g)?.slice(0, 3).map(Number) ?? []
  if (!got.length) return false
  for (let i = 0; i < want.length; i++) if ((got[i] ?? 0) !== want[i]) return (got[i] ?? 0) < want[i]
  return false
}



export function engineSkew(here: string | undefined | null, there: string | undefined | null): boolean {
  if (!here || !there || here.includes('Python') || there.includes('Python')) return false
  return here !== there
}

export function displayEngineVersion(machineVersion: string | null | undefined, flagsVersion: string | null | undefined): string {
  return machineVersion || flagsVersion || 'not installed'
}
