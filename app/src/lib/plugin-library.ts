import type { LocalScriptCapability, PackagedPluginCapability } from './types'

export function matchesPluginSearch(query: string, ...values: Array<string | null | undefined>): boolean {
  const haystack = values.filter(Boolean).join(' ').toLocaleLowerCase()
  return query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean).every((term) => haystack.includes(term))
}

/** Configuration is not a live runtime attestation. */
export function executableEvidence(item: Pick<LocalScriptCapability, 'checksumSha256' | 'signatureVerifiedAt'>): string {
  if (item.signatureVerifiedAt) return 'Signature recorded'
  if (item.checksumSha256) return 'Checksum recorded'
  return 'No integrity record'
}

export function pluginRegistrationEnabled(plugin: PackagedPluginCapability): boolean {
  return plugin.enabled === true && plugin.trusted === true
}

// The field is an argument list, never a shell command. Double-quote escaping is
// deliberately small and round-trips paths, whitespace and empty arguments.
export function formatMcpArguments(args: string[]): string {
  return args.map((arg) => !arg || /[\s"'\\]/.test(arg)
    ? `"${arg.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`
    : arg).join(' ')
}

export function parseMcpArguments(value: string): string[] {
  const args: string[] = []
  let current = ''
  let started = false
  let quote: '"' | "'" | null = null
  for (let i = 0; i < value.length; i++) {
    const char = value[i]
    if (char === '\\' && quote !== "'") {
      const next = value[i + 1]
      if (next === '\\' || next === '"' || (!quote && (next === "'" || /\s/.test(next ?? '')))) {
        current += next
        started = true
        i++
        continue
      }
    }
    if (quote) {
      if (char === quote) quote = null
      else current += char
      continue
    }
    if (char === '"' || char === "'") { quote = char; started = true; continue }
    if (/\s/.test(char)) {
      if (started) { args.push(current); current = ''; started = false }
      continue
    }
    current += char
    started = true
  }
  if (quote) throw new Error('Close the quote in Arguments before saving.')
  if (started) args.push(current)
  return args
}
