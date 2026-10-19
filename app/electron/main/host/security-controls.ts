import { canUseMacSandbox } from '../../agent/sandbox'
import { canUseLinuxSandbox } from '../../agent/linux-sandbox'
import type { SecurityControlSnapshot } from '../../agent/types'
import { store } from '../state'
import { auditActor, currentSecurityRole, envEnabled, getLastAuditError } from './audit'
import { signedUpdatesRequired, updateSignaturePublicKey } from './updates'
import { VONTRA_UPDATE_TRUST_ROOT_PRODUCTION_READY } from '../../agent/update-trust'
import { lastPersistFailure } from './persistence'
import { localSecretStorageStatus } from '../../store'

export function localMemoryCounts(): Record<string, number> {
  return {
    artifacts: store.artifacts.length,
    threads: store.threads.length,
    projects: store.projects.length,
    rooms: store.rooms.length,
    tickets: store.tickets.length,
    automations: store.automations.length,
    connectorEvents: store.connectorEvents.length,
    missions: store.missions.length,
    jobs: store.jobs.length,
    subAgents: store.subAgents.length,
    botSessions: store.botSessions.length,
    projectMemory: store.projectMemory.length,
    projectMemoryCandidates: store.projectMemoryCandidates.length,
    incidents: store.incidents.length,
    contacts: Object.keys(store.contacts.records).length,
    cliSessions: store.cliSessions.length,
  }
}

export function securityControlSnapshot(): SecurityControlSnapshot {
  const localSecrets = localSecretStorageStatus()
  return {
    actor: auditActor(),
    role: currentSecurityRole(),
    commandSandboxAvailable: canUseMacSandbox() || canUseLinuxSandbox(),
    mcpSandboxAvailable: canUseMacSandbox(),
    signedUpdatesRequired: signedUpdatesRequired(),
    updatePublicKeyConfigured: !!updateSignaturePublicKey(),
    updateTrustRootProductionReady: VONTRA_UPDATE_TRUST_ROOT_PRODUCTION_READY,
    signedMcpRequired: envEnabled('VONTRA_REQUIRE_SIGNED_MCP'),
    trustedMcpExplicitlyAllowed: envEnabled('VONTRA_ALLOW_TRUSTED_MCP'),
    localStateKeyMode: localSecrets.keyMode,
    externalSecretKeyRequired: localSecrets.externalKeyRequired,
    copiedProfileDecryptable: localSecrets.copiedProfileDecryptable,
    legacySafeStorageSecrets: localSecrets.legacySafeStorageSecrets,
    legacySafeStorageBackupPath: localSecrets.legacySafeStorageBackupPath,
    quarantinedProtectedFiles: localSecrets.quarantinedProtectedFiles,
    quarantinedProtectedPaths: localSecrets.quarantinedProtectedPaths,
    quarantineBackupPaths: localSecrets.quarantineBackupPaths,
    // There is no plaintext-store bypass. Keep the compatibility field honest
    // even if a stale deployment still exports the former unused variable.
    unencryptedLocalStoreAllowed: false,
    strictAuditEnabled: envEnabled('VONTRA_STRICT_AUDIT'),
    auditHealthy: !getLastAuditError(),
    lastAuditError: getLastAuditError(),
    lastPersistenceFailure: lastPersistFailure(),
    localMemoryCounts: localMemoryCounts(),
  }
}
