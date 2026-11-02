import type { QuickAskPreferences } from '../../../shared/quick-ask'
interface ShortcutRegistry { register(key: string, callback: () => void): boolean; unregister(key: string): void }
export function createAskShortcut(registry: ShortcutRegistry, toggle: () => void) {
  let registered: string | null = null
  return {
    apply(preferences: Pick<QuickAskPreferences, 'enabled' | 'shortcut'>): void {
      const next = preferences.enabled ? preferences.shortcut : null
      if (next === registered) return
      if (next && !registry.register(next, toggle)) throw new Error('This shortcut is already in use or the desktop denied it. Choose another shortcut. Ask TensorFold is still available from the app menu.')
      if (registered) registry.unregister(registered)
      registered = next
    },
    dispose(): void { if (registered) registry.unregister(registered); registered = null },
  }
}
