import type { EngineRuntimeSnapshot } from '../agent/types/engine-runtime'
import type { Settings } from '../agent/types'
import type { EngineModelComparisonInput, EngineModelComparisonSnapshot, EngineModelPreviewResult, SaveEngineModelProfilesInput, EngineConnection, EngineEnvOverrides, EngineEvent, EngineLearningSnapshot, EngineInstalledModel, EngineSnapshot, EngineStatus, HubModelSummary, ModelPreflight, PullProgress } from '../agent/types'
import type { EngineModelLibrarySnapshot, EngineLibraryScanOptions } from '../agent/types'
import type { IpcRenderer } from 'electron'
import type { TensorFoldStudioApi } from '../../shared/studio-api'

export function enginePreload(ipcRenderer: Pick<IpcRenderer, 'invoke' | 'on' | 'removeListener'>): Pick<TensorFoldStudioApi, 'engineRuntimeStatus' | 'engineRuntimeCheck' | 'engineRuntimeInstall' | 'engineRuntimeUse' | 'engineRuntimeCancel' | 'engineSnapshot' | 'engineConnection' | 'engineRefresh' | 'engineLibrary' | 'engineScanLibrary' | 'engineAddLibraryFolder' | 'engineRemoveLibraryFolder' | 'engineCancelLibraryScan' | 'engineSearch' | 'enginePreflight' | 'enginePull' | 'engineCancelPull' | 'engineCompareModels' | 'engineComparisonStatus' | 'engineCancelComparison' | 'engineTryModel' | 'engineCancelTryModel' | 'engineSaveModelProfiles' | 'engineLoad' | 'engineUnload' | 'engineSetCortheon' | 'engineLearning' | 'engineSetLearning' | 'engineEnvOverrides' | 'engineSetEnvOverrides' | 'engineTeach' | 'engineCancelTeach' | 'engineDelete' | 'onEngineEvent'> {
  return {
    // Built-in TensorFold engine: status, installed models, Hugging Face search and downloads.
    engineRuntimeStatus: (): Promise<EngineRuntimeSnapshot> => ipcRenderer.invoke('engine:runtimeStatus'),
    engineRuntimeCheck: (): Promise<EngineRuntimeSnapshot> => ipcRenderer.invoke('engine:runtimeCheck'),
    engineRuntimeInstall: (id: string): Promise<EngineRuntimeSnapshot> => ipcRenderer.invoke('engine:runtimeInstall', id),
    engineRuntimeUse: (id: string | null): Promise<EngineRuntimeSnapshot> => ipcRenderer.invoke('engine:runtimeUse', id),
    engineRuntimeCancel: (): Promise<boolean> => ipcRenderer.invoke('engine:runtimeCancel'),
    engineSnapshot: (): Promise<EngineSnapshot> => ipcRenderer.invoke('engine:snapshot'),
    engineConnection: (): Promise<EngineConnection | null> => ipcRenderer.invoke('engine:connection'),
    engineRefresh: (): Promise<EngineSnapshot> => ipcRenderer.invoke('engine:refresh'),
    engineLibrary: (): Promise<EngineModelLibrarySnapshot> => ipcRenderer.invoke('engine:library'),
    engineScanLibrary: (options?: EngineLibraryScanOptions): Promise<EngineModelLibrarySnapshot> => ipcRenderer.invoke('engine:scanLibrary', options),
    engineAddLibraryFolder: (): Promise<EngineModelLibrarySnapshot | null> => ipcRenderer.invoke('engine:addLibraryFolder'),
    engineRemoveLibraryFolder: (id: string): Promise<EngineModelLibrarySnapshot> => ipcRenderer.invoke('engine:removeLibraryFolder', id),
    engineCancelLibraryScan: (): Promise<EngineModelLibrarySnapshot> => ipcRenderer.invoke('engine:cancelLibraryScan'),
    engineSearch: (query: string): Promise<HubModelSummary[]> => ipcRenderer.invoke('engine:search', query),
    enginePreflight: (repo: string, revision?: string | null): Promise<ModelPreflight> => ipcRenderer.invoke('engine:preflight', { repo, revision }),
    enginePull: (repo: string, revision?: string | null): Promise<PullProgress> => ipcRenderer.invoke('engine:pull', { repo, revision }),
    engineCancelPull: (repo: string): Promise<{ cancelled: boolean }> => ipcRenderer.invoke('engine:cancelPull', repo),
    engineCompareModels: (input: EngineModelComparisonInput): Promise<EngineModelComparisonSnapshot> => ipcRenderer.invoke('engine:compareModels', input),
    engineComparisonStatus: (id: string): Promise<EngineModelComparisonSnapshot | null> => ipcRenderer.invoke('engine:comparisonStatus', id),
    engineCancelComparison: (id: string): Promise<boolean> => ipcRenderer.invoke('engine:cancelComparison', id),
    engineTryModel: (model: string): Promise<EngineModelPreviewResult> => ipcRenderer.invoke('engine:tryModel', model),
    engineCancelTryModel: (): Promise<boolean> => ipcRenderer.invoke('engine:cancelTryModel'),
    engineSaveModelProfiles: (input: SaveEngineModelProfilesInput): Promise<Settings> => ipcRenderer.invoke('engine:profilesSave', input),
    engineLoad: (repo: string): Promise<EngineStatus> => ipcRenderer.invoke('engine:load', repo),
    engineUnload: (): Promise<EngineStatus> => ipcRenderer.invoke('engine:unload'),
    engineSetCortheon: (enabled: boolean): Promise<EngineStatus> => ipcRenderer.invoke('engine:setCortheon', enabled),
    engineLearning: (): Promise<EngineLearningSnapshot> => ipcRenderer.invoke('engine:learning'),
    engineSetLearning: (enabled: boolean, model?: string): Promise<EngineLearningSnapshot> => ipcRenderer.invoke('engine:setLearning', { enabled, model }),
    engineEnvOverrides: (): Promise<EngineEnvOverrides> => ipcRenderer.invoke('engine:envOverrides'),
    engineSetEnvOverrides: (overrides: EngineEnvOverrides): Promise<EngineEnvOverrides> => ipcRenderer.invoke('engine:setEnvOverrides', overrides),
    engineTeach: (text: string): Promise<EngineLearningSnapshot> => ipcRenderer.invoke('engine:teach', text),
    engineCancelTeach: (): Promise<EngineLearningSnapshot> => ipcRenderer.invoke('engine:cancelTeach'),
    engineDelete: (repo: string): Promise<EngineInstalledModel[]> => ipcRenderer.invoke('engine:delete', repo),
    onEngineEvent: (cb: (event: EngineEvent) => void) => { const h = (_e: unknown, event: EngineEvent) => cb(event); ipcRenderer.on('engine:event', h); return () => ipcRenderer.removeListener('engine:event', h) },
  }
}
