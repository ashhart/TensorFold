import assert from 'node:assert/strict'
import { agentBuilderModelSourceCount } from './agent-builder'
import type { Provider } from './types'

const providers: Provider[] = [
  {
    id: 'cloud',
    name: 'Cloud',
    kind: 'cloud',
    baseUrl: 'https://example.com/v1',
    apiKey: '',
    models: ['model-a'],
  },
  {
    id: 'cli',
    name: 'CLI',
    kind: 'cloud',
    transport: 'codex-cli',
    baseUrl: '',
    apiKey: '',
    models: ['model-b'],
    accounts: [
      { id: 'personal', name: 'Personal', kind: 'subscription', enabled: true, models: ['model-b'], createdAt: 1, updatedAt: 1 },
      { id: 'work', name: 'Work', kind: 'subscription', enabled: true, models: ['model-c'], createdAt: 1, updatedAt: 1 },
    ],
  },
  {
    id: 'empty',
    name: 'No models',
    kind: 'local',
    baseUrl: 'http://127.0.0.1:9999/v1',
    apiKey: '',
    models: [],
  },
]

assert.equal(
  agentBuilderModelSourceCount(providers),
  3,
  'Agent Builder counts usable provider/account model groups even before an agent preset is saved',
)

console.log('Agent Builder model-source count follows available model groups')
