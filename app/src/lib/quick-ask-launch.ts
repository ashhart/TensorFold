export function openAskTensorFold(): void {
  void window.tensorfoldQuickAskLauncher?.open().catch(error => {
    window.dispatchEvent(new CustomEvent('vontra:ipc-error', { detail: String(error instanceof Error ? error.message : error) }))
  })
}
