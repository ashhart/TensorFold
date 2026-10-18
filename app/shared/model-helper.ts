import type { ModelHelperRole, ModelQualificationRoute } from './model-fit'
export interface ModelHelperRequest { requestId: string; role: ModelHelperRole; threadId: string; text: string }
export interface ModelHelperResult { route: ModelQualificationRoute; text: string; complete: boolean; warning?: string; sources: Array<{ documentId: string; passageId: string; revision: string }> }
export interface ModelHelperApi {
  modelHelperRun(input: ModelHelperRequest): Promise<ModelHelperResult>
  modelHelperCancel(requestId: string): Promise<void>
}
