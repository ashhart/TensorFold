import type { ClusterWorkspaceApi } from '../../shared/cluster-workspace'
export function clusterPreload(invoke: (channel: string, ...args: unknown[]) => Promise<any>): ClusterWorkspaceApi {
  return {
    clusterWorkspace: () => invoke('cluster:workspace'), clusterSaveHost: input => invoke('cluster:saveHost', input), clusterRemoveHost: input => invoke('cluster:removeHost', input),
    clusterDiscover: () => invoke('cluster:discover'), clusterInspectHost: input => invoke('cluster:inspectHost', input), clusterPlan: input => invoke('cluster:plan', input),
    clusterEnsureModel: input => invoke('cluster:ensureModel', input), clusterActivate: input => invoke('cluster:activate', input), clusterStop: input => invoke('cluster:stop', input),
    clusterCancelJob: input => invoke('cluster:cancelJob', input), clusterReconcile: input => invoke('cluster:reconcile', input), clusterEndpointCredentials: input => invoke('cluster:endpointCredentials', input),
    clusterUseEndpoint: input => invoke('cluster:useEndpoint', input), clusterSshKey: input => invoke('cluster:sshKey', input), clusterBootstrapPreview: input => invoke('cluster:bootstrapPreview', input), clusterBootstrapApply: input => invoke('cluster:bootstrapApply', input),
  }
}
