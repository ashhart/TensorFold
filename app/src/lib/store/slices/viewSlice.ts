import type { Slice, View } from '../types'

export const viewSlice: Slice = (set, get) => ({
  view: { name: 'home' } as View,

  setView(v) { set({ view: v }) },
  openThread(id) { set({ view: { name: 'thread', id } }) },
  // New task: the home composer, where a folder makes it a task and no folder makes it a chat.
  newTask() {
    set({ view: { name: 'home' } })
    requestAnimationFrame(() => document.querySelector<HTMLElement>('[data-vontra-composer="true"]')?.focus())
  },
  openProject(id) { set({ view: { name: 'project', id } }) },
  openRoom(id) { set({ view: { name: 'room', id } }) },
  openThreadDock(threadId, dock) {
    set({ view: { name: 'thread', id: threadId }, threadDockRequest: { threadId, dock, nonce: Date.now() } })
  },
})
