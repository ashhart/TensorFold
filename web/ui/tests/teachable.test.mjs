import assert from 'node:assert/strict'
import { test } from 'node:test'
import { whyNotTeach } from '../src/teachable.ts'

const running = { status: 'running', managed: true, owned: true, model: 'TensorFold/Qwen3.8-27B-MLX-4bit', model_repo: 'TensorFold/Qwen3.8-27B-MLX-4bit', model_name: 'Qwen3.8-27B' }
const view = { teachable: null, served: null, copies: [], bases: ['TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit'] }

test('nothing running, or nothing known yet: the tile stays as it is', () => {
  assert.equal(whyNotTeach(null, view), null)
  assert.equal(whyNotTeach({ ...running, status: 'stopped' }, view), null)
  assert.equal(whyNotTeach(running, null), null)
})

test('a model of a family TensorFold does not learn into says so', () => {
  assert.match(whyNotTeach(running, view), /Qwen3.8-27B is not a model TensorFold learns into yet/)
  const nemotron = { ...running, model: view.bases[0], model_repo: view.bases[0], model_name: 'Nemotron 3.5 Lightning' }
  assert.equal(whyNotTeach(nemotron, view), null)
})

test('a served copy follows its own learnable flag', () => {
  const copyOf = (learnable) => ({ ...view, served: 'eden', copies: [{ name: 'eden', learnable, family: learnable ? 'nemotron_h' : 'qwen3' }] })
  const served = { ...running, model: 'living:eden', model_repo: null, model_name: 'eden' }
  assert.equal(whyNotTeach(served, copyOf(true)), null)
  assert.match(whyNotTeach(served, copyOf(false)), /eden is a copy of qwen3/)
})

test('a machine or an engine that cannot learn, and an engine run elsewhere, come first', () => {
  assert.match(whyNotTeach(running, { ...view, teachable: 'this engine (1.0.5) does not learn' }), /cannot run on this machine: this engine/)
  assert.match(whyNotTeach({ ...running, managed: false, owned: false }, view), /runs on another engine/)
})
