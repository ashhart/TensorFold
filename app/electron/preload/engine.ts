import type { EngineModelComparisonInput, EngineModelComparisonSnapshot, EngineModelPreviewResult, SaveEngineModelProfilesInput, EngineConnection, EngineControlVectorSnapshot, EngineEnvOverrides, EngineEvent, EngineLearningSnapshot, EngineInstalledModel, EngineSnapshot, EngineStatus, HubModelSummary, ModelPreflight, PullProgress, Settings } from '../agent/types'
import type { EngineModelLibrarySnapshot, EngineLibraryScanOptions } from '../agent/types'

export type Invoke = (channel: string, ...args: unknown[]) => Promise<any>
export type Subscribe = <T>(channel: string, listener: (event: unknown, payload: T) => void) => () => void

// Built-in TensorFold engine: status, installed models, Hugging Face search and downloads.
export function enginePreload(invoke: Invoke, subscribe: Subscribe) {
  return {
    engineSnapshot: (): Promise<EngineSnapshot> => invoke('engine:snapshot'),
    engineConnection: (): Promise<EngineConnection | null> => invoke('engine:connection'),
    engineRefresh: (): Promise<EngineSnapshot> => invoke('engine:refresh'),
    engineLibrary: (): Promise<EngineModelLibrarySnapshot> => invoke('engine:library'),
    engineScanLibrary: (options?: EngineLibraryScanOptions): Promise<EngineModelLibrarySnapshot> => invoke('engine:scanLibrary', options),
    engineAddLibraryFolder: (): Promise<EngineModelLibrarySnapshot | null> => invoke('engine:addLibraryFolder'),
    engineRemoveLibraryFolder: (id: string): Promise<EngineModelLibrarySnapshot> => invoke('engine:removeLibraryFolder', id),
    engineCancelLibraryScan: (): Promise<EngineModelLibrarySnapshot> => invoke('engine:cancelLibraryScan'),
    engineSearch: (query: string): Promise<HubModelSummary[]> => invoke('engine:search', query),
    enginePreflight: (repo: string, revision?: string | null): Promise<ModelPreflight> => invoke('engine:preflight', { repo, revision }),
    enginePull: (repo: string, revision?: string | null): Promise<PullProgress> => invoke('engine:pull', { repo, revision }),
    engineCancelPull: (repo: string): Promise<{ cancelled: boolean }> => invoke('engine:cancelPull', repo),
    engineCompareModels: (input: EngineModelComparisonInput): Promise<EngineModelComparisonSnapshot> => invoke('engine:compareModels', input),
    engineComparisonStatus: (id: string): Promise<EngineModelComparisonSnapshot | null> => invoke('engine:comparisonStatus', id),
    engineCancelComparison: (id: string): Promise<boolean> => invoke('engine:cancelComparison', id),
    engineTryModel: (model: string): Promise<EngineModelPreviewResult> => invoke('engine:tryModel', model),
    engineCancelTryModel: (): Promise<boolean> => invoke('engine:cancelTryModel'),
    engineSaveModelProfiles: (input: SaveEngineModelProfilesInput): Promise<Settings> => invoke('engine:profilesSave', input),
    engineLoad: (repo: string): Promise<EngineStatus> => invoke('engine:load', repo),
    engineUnload: (): Promise<EngineStatus> => invoke('engine:unload'),
    engineSetCortheon: (enabled: boolean): Promise<EngineStatus> => invoke('engine:setCortheon', enabled),
    engineControlVector: (): Promise<EngineControlVectorSnapshot> => invoke('engine:controlVector'),
    engineChooseControlVector: (): Promise<EngineControlVectorSnapshot> => invoke('engine:chooseControlVector'),
    engineSetControlVector: (input: { enabled?: boolean; alpha?: number }): Promise<EngineControlVectorSnapshot> => invoke('engine:setControlVector', input),
    engineClearControlVector: (): Promise<EngineControlVectorSnapshot> => invoke('engine:clearControlVector'),
    engineLearning: (): Promise<EngineLearningSnapshot> => invoke('engine:learning'),
    engineSetLearning: (enabled: boolean, model?: string): Promise<EngineLearningSnapshot> => invoke('engine:setLearning', { enabled, model }),
    engineEnvOverrides: (): Promise<EngineEnvOverrides> => invoke('engine:envOverrides'),
    engineSetEnvOverrides: (overrides: EngineEnvOverrides): Promise<EngineEnvOverrides> => invoke('engine:setEnvOverrides', overrides),
    engineTeach: (text: string): Promise<EngineLearningSnapshot> => invoke('engine:teach', text),
    engineCancelTeach: (): Promise<EngineLearningSnapshot> => invoke('engine:cancelTeach'),
    engineDelete: (repo: string): Promise<EngineInstalledModel[]> => invoke('engine:delete', repo),
    onEngineEvent: (cb: (event: EngineEvent) => void) => subscribe<EngineEvent>('engine:event', (_e, event) => cb(event)),
  }
}
