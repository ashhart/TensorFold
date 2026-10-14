export interface NativeConfirmationRequest {
  title: string
  message: string
  detail: string
  confirmLabel: string
}

export type ConfirmSensitiveAction = (input: NativeConfirmationRequest) => Promise<boolean>

/** Main-process confirmation gate. Missing confirmation wiring fails closed. */
export async function requireNativeConfirmation(
  confirm: ConfirmSensitiveAction | undefined,
  input: NativeConfirmationRequest,
): Promise<void> {
  if (!confirm) throw new Error('Sensitive action cancelled because native confirmation is unavailable.')
  if (!await confirm(input)) throw new Error('Sensitive action cancelled by the local user.')
}
