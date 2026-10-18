import type { ClusterWorkspaceApi } from '../../../shared/cluster-workspace'
/** Browser previews never open sockets, inspect keys or pretend hardware has passed a check. */
export function createMockClusterApi(): ClusterWorkspaceApi {
  const unavailable = async (): Promise<never> => { throw new Error('Compute management requires the desktop app; no machine was contacted.') }
  return { clusterWorkspace: async () => ({ revision: 0, hosts: [], plans: [], jobs: [], deployments: [] }), clusterDiscover: async () => [], clusterSaveHost: unavailable, clusterRemoveHost: unavailable, clusterInspectHost: unavailable, clusterPlan: unavailable, clusterEnsureModel: unavailable, clusterActivate: unavailable, clusterStop: unavailable, clusterCancelJob: unavailable, clusterReconcile: unavailable, clusterEndpointCredentials: unavailable, clusterUseEndpoint: unavailable, clusterSshKey: unavailable, clusterBootstrapPreview: unavailable, clusterBootstrapApply: unavailable }
}
