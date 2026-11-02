import type { IpcRenderer } from 'electron'
import type { QuickAskApi, QuickAskEvent, QuickAskLauncher } from '../../shared/quick-ask'
type AskIpc = Pick<IpcRenderer, 'invoke' | 'on' | 'removeListener'>
export function quickAskBridge(ipc: AskIpc): QuickAskApi {
  return {
    resize: height => ipc.invoke('quick-ask:resize', height),
    pickWindow: () => ipc.invoke('quick-ask:pick-window'),
    useWindow: token => ipc.invoke('quick-ask:use-window', token),
    snapshot: options => ipc.invoke('quick-ask:snapshot', options),
    saveDraft: draft => ipc.invoke('quick-ask:draft', draft),
    configure: preferences => ipc.invoke('quick-ask:configure', preferences),
    recordShortcut: active => ipc.invoke('quick-ask:record-shortcut', active),
    send: input => ipc.invoke('quick-ask:send', input),
    cancel: () => ipc.invoke('quick-ask:cancel'),
    reset: () => ipc.invoke('quick-ask:reset'),
    clipboard: () => ipc.invoke('quick-ask:clipboard'),
    displays: () => ipc.invoke('quick-ask:displays'),
    capture: displayId => ipc.invoke('quick-ask:capture', displayId),
    loadModel: routeId => ipc.invoke('quick-ask:load', routeId),
    handoff: () => ipc.invoke('quick-ask:handoff'),
    copy: text => ipc.invoke('quick-ask:copy', text),
    openLink: url => ipc.invoke('quick-ask:link', url),
    hide: () => ipc.invoke('quick-ask:hide'),
    acknowledgeFlush: input => ipc.invoke('quick-ask:flushed', input),
    speak: text => ipc.invoke('quick-ask:speak', text),
    onEvent: callback => { const handler = (_: unknown, event: QuickAskEvent) => callback(event); ipc.on('quick-ask:event', handler); return () => ipc.removeListener('quick-ask:event', handler) },
  }
}
export function quickAskLauncher(ipc: AskIpc): QuickAskLauncher {
  return {
    open: () => ipc.invoke('quick-ask:open'),
    consumeHandoff: () => ipc.invoke('quick-ask:consume-handoff'),
    onHandoff: callback => { const handler = () => callback(); ipc.on('quick-ask:navigate', handler); return () => ipc.removeListener('quick-ask:navigate', handler) },
  }
}
