import type { TensorFoldStudioApi } from '../../shared/studio-api'
import { pairedToken } from './remote-pairing'

let demo: TensorFoldStudioApi | null = null
let preparing: Promise<void> | null = null

/** Called before importing App or ipc, and only after pairing has decided the mode. */
export async function prepareBrowserDemo(
  load: () => Promise<{ createMockApi(): TensorFoldStudioApi }> = () => import('./mock'),
): Promise<void> {
  if (window.tensorfoldStudio || window.vontra || pairedToken() || demo) return
  if (!preparing) preparing = load().then(({ createMockApi }) => { demo = createMockApi() }).catch(error => {
    preparing = null
    throw error
  })
  await preparing
}

export function preparedBrowserDemo(): TensorFoldStudioApi {
  if (!demo) throw new Error('Browser demo has not been prepared before importing the app.')
  return demo
}
