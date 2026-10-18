import assert from 'node:assert/strict'
import { aggregateHubFailures, decodeHubFailure, hubError, hubFailureCopy, type HubFailureCategory } from '../../shared/hub-failure'

for (const category of ['policy', 'transport', 'offline', 'http', 'check'] as HubFailureCategory[]) {
  const message = category === 'policy'
    ? 'Hub host huggingface.co refused address 10.0.0.1. Use fake-ip-filter or HF_ENDPOINT.' : `${category} detail`
  const mainError = hubError(category, message)
  // Electron drops main-process class identity and custom properties, preserving only the message.
  const serialized = JSON.parse(JSON.stringify({ message: `Error invoking remote method 'engine:preflight': Error: ${mainError.message}` }))
  const failure = decodeHubFailure(serialized)
  assert.equal(failure.category, category)
  assert.equal(failure.message, message)
  assert.equal(failure.coded, true)
  assert.ok(hubFailureCopy(category).title)
}
assert.equal(decodeHubFailure(new Error('unexpected validation failure')).category, 'check')
assert.equal(decodeHubFailure('[TF_HUB_OTHER] unknown code').category, 'check')
assert.equal(aggregateHubFailures(['transport', 'http', 'policy']), 'policy')
assert.equal(aggregateHubFailures(['check', 'offline']), 'offline')
assert.equal(aggregateHubFailures(['http', 'transport']), 'transport')
assert.equal(aggregateHubFailures([]), null)
const policy = hubFailureCopy('policy')
assert.equal(policy.title, 'Model checks blocked by network policy')
assert.equal(policy.detail, 'The Hub address was refused. Check your DNS or proxy settings, then retry.')
assert.doesNotMatch(`${policy.title} ${policy.detail}`, /reconnect|credentials/i)
assert.equal(hubFailureCopy('transport').title, 'Could not reach the Hub')
assert.equal(hubFailureCopy('offline').title, 'Model checks are offline')
assert.notEqual(hubFailureCopy('http').title, hubFailureCopy('transport').title)
