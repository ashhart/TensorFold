import { app, Notification } from 'electron'
import { createRunAttention } from '../host/run-attention'
import { store } from '../state'
import { sendToRenderer } from '../host/renderer-events'

export function createApplicationAttention() {
  const liveNotifications = new Set<Notification>()
  const runAttention = createRunAttention({
    isIncognito: (id) => {
      const thread = store.threads.find((thread) => thread.id === id)

      return !thread || thread.incognito === true || thread.phoneLocalOnly === true || thread.googleLocalOnly === true
    },
    enabled: () => store.settings?.notifyRuns !== false,
    focused: () => { try { return !!store.win && !store.win.isDestroyed() && store.win.isFocused() } catch { return false } },

    supported: () => !!store.win && Notification.isSupported(),
    threadTitle: (id) => store.threads.find((t) => t.id === id)?.title,
    notify: (notice) => {
      const n = new Notification({ title: notice.title, body: notice.body })
      liveNotifications.add(n)
      n.on('click', () => {
        liveNotifications.delete(n)
        const win = store.win
        try { if (win && !win.isDestroyed()) { win.show(); win.focus() } } catch {}
        sendToRenderer('app:open-thread', notice.threadId)
      })
      n.on('close', () => { liveNotifications.delete(n) })
      n.show()
    },
    setBadge: (text) => { app.dock?.setBadge(text) },
  })

  return runAttention
}
