type TrustedPolicyFactory = {
  createPolicy(name: string, rules: {
    createHTML?(value: string): string
    createScript?(value: string): string
    createScriptURL?(value: string): string
  }): unknown
  getPolicyNames?(): string[]
}

// Fail closed at the Trusted Types boundary. The CSP enforces TT for script sinks
// and nothing in the app assigns HTML through them, so the default policy must not
// "sanitize": a regex blocklist is bypassable (e.g. `java\tscript:` slips past URL
// parsing) and would silently launder any future sink. Throwing here also squats
// the `default` name so nothing else can register a laundering policy later.
export function installDefaultTrustedTypesPolicy(scope: any = globalThis): boolean {
  const factory = scope.trustedTypes as TrustedPolicyFactory | undefined
  if (!factory?.createPolicy) return false
  if (factory.getPolicyNames?.().includes('default')) return true
  factory.createPolicy('default', {
    createHTML: () => {
      throw new TypeError('Blocked HTML sink: this app registers no Trusted Types HTML policy.')
    },
    createScript: (value) => {
      if (value) throw new TypeError('Inline script text is not allowed.')
      return value
    },
    createScriptURL: (value) => {
      const base = String(scope.location?.href ?? 'file:///')
      const url = new URL(value, base)
      const current = new URL(base)
      if (url.protocol === 'blob:' || url.origin === current.origin || (url.protocol === 'file:' && current.protocol === 'file:')) return url.href
      throw new TypeError('Cross-origin script URLs are not allowed.')
    },
  })
  return true
}

installDefaultTrustedTypesPolicy()
