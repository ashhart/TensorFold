import type { ComposerDraftScope } from './composer-drafts'

/** The host resolves the conversation's exact workspace and verifies the selected account. */
export interface InputHistoryScope {
  draft: ComposerDraftScope
  projectId: string | null
  cwd: string
  providerId: string | null
  accountId: string | null
}
export interface InputHistoryEntry { id: string; text: string; createdAt: number }
export interface InputHistorySnapshot { revision: number; entries: InputHistoryEntry[]; partial: boolean }
export interface InputHistoryApi {
  getInputHistory(scope: InputHistoryScope): Promise<InputHistorySnapshot>
  recallInputHistory(input: { scope: InputHistoryScope; id: string; expectedRevision: number }): Promise<string>
  clearInputHistory(input: { scope: InputHistoryScope; expectedRevision: number }): Promise<InputHistorySnapshot>
}
