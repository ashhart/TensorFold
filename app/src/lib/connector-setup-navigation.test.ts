import assert from 'node:assert/strict'
import { requestConnectorSetup, consumeConnectorSetup, requestSettingsSection, consumeSettingsSection } from './store/settings-section'
import { CONNECTOR_SETUP_IDS, CONNECTOR_CATALOG, isConnectorSetupId } from '../../shared/connector-catalog'

for (const id of CONNECTOR_SETUP_IDS) {
  requestConnectorSetup(id)
  assert.equal(consumeSettingsSection(), 'connectors')
  assert.equal(consumeConnectorSetup(), id, 'The chosen service must survive the Settings mount')
  assert.equal(consumeConnectorSetup(), null, 'A consumed service cannot reopen on a later visit')
  assert.ok(CONNECTOR_CATALOG[id].name)
}
requestConnectorSetup('whatsapp')
requestSettingsSection('voice')
assert.equal(consumeConnectorSetup(), null, 'A later unrelated settings request cancels the stale service target')
assert.equal(isConnectorSetupId('unsupported-service'), false)
assert.throws(() => requestConnectorSetup('unsupported-service' as any), /supported connection/)
