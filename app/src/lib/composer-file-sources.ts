// Selected File handles live only in the renderer session, scoped like persistent drafts.
const sources = new Map<string, Map<string, File>>()
export function composerFileSources(scope: string): Map<string, File> {
  let files = sources.get(scope)
  if (!files) { files = new Map(); sources.set(scope, files) }
  return files
}
export function forgetComposerFileSources(scope?: string): void {
  if (scope === undefined) {
    for (const files of sources.values()) files.clear()
    sources.clear()
  } else {
    sources.get(scope)?.clear()
    sources.delete(scope)
  }
}

/** Transfer originals to the private session while invalidating old callback owners. */
export function makeComposerFileSourcesIncognito(normalKey: string, privateKey: string): void {
  if (normalKey === privateKey) return
  const normal = sources.get(normalKey)
  if (!normal) return
  const previousPrivate = sources.get(privateKey)
  const moved = new Map([...normal, ...(previousPrivate ?? [])])
  normal.clear()
  previousPrivate?.clear()
  sources.delete(normalKey)
  sources.set(privateKey, moved)
}
