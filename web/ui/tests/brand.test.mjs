import assert from 'node:assert/strict'
import { test } from 'node:test'
import { brandOf, builtIn, hardwareOf, setCustomBrands } from '../src/brand.ts'

test('a model is known by any of its names', () => {
  assert.equal(brandOf('Nemotron 3.5 Lightning 30B-A3B')?.id, 'nvidia')
  assert.equal(brandOf('TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit')?.id, 'nvidia')
  assert.equal(brandOf('nemotron-living')?.id, 'nvidia')                    // a teachable copy
  assert.equal(brandOf('Qwen3.8-27B-MLX-4bit')?.id, 'qwen')
  assert.equal(brandOf('Ternary Bonsai 2 27B', 'prism_hadamard_qwen35')?.id, 'prism')   // not Qwen
  assert.equal(brandOf('mlx-community/gemma-4-26b-a4b-it-4bit')?.id, 'google')
  assert.equal(brandOf('GLM-5.3-Flash')?.id, 'zai')
  assert.equal(brandOf('DeepSeek-V4-Flash')?.id, 'deepseek')
  assert.equal(brandOf(null, undefined, 'my-own-model'), null)
  assert.equal(brandOf(null, 'Qwen3.6-35B-A3B')?.id, 'qwen')                // the first name that says
})

test('a machine shows its hardware maker', () => {
  assert.equal(hardwareOf('Apple M6')?.id, 'apple')
  assert.equal(hardwareOf(null, 'mlx')?.id, 'apple')
  assert.equal(hardwareOf('NVIDIA GB10 (GX10)')?.id, 'nvidia')
  assert.equal(hardwareOf('unknown', 'cuda')?.id, 'nvidia')
  assert.equal(hardwareOf('AMD Ryzen AI Max+ 395')?.id, 'amd')
  assert.equal(hardwareOf('SiFive P870', 'none'), null)
})

test('the families a new machine is likely to serve have a maker', () => {
  assert.equal(brandOf('Llama-3.3-70B-Instruct')?.id, 'meta')
  assert.equal(brandOf('Devstral-Small-2507')?.id, 'mistralai')
  assert.equal(brandOf('Kimi-K2-Instruct')?.id, 'kimi')
  assert.equal(brandOf('gpt-oss-120b')?.id, 'openai')
  assert.equal(brandOf('Phi-4-mini')?.id, 'microsoft')
  assert.equal(brandOf('granite-4.0-h-small')?.id, 'ibm')
  assert.equal(hardwareOf('Intel Core Ultra 9 285H')?.id, 'intel')
  assert.equal(hardwareOf('Snapdragon X Elite')?.id, 'qualcomm')
})

test('a maker added in Settings comes first, and renames a known one of the same id', () => {
  setCustomBrands([
    { id: 'falcon', label: 'TII Falcon', kind: 'model', words: ['falcon'], logo: null },
    { id: 'qwen', label: 'Qwen (Alibaba)', kind: 'model', words: [], logo: 'data:image/png;base64,AA==' },
    { id: 'sifive', label: 'SiFive', kind: 'hardware', words: ['sifive'], logo: null },
  ])
  try {
    assert.deepEqual(brandOf('Falcon-H1-34B'), { id: 'falcon', label: 'TII Falcon' })
    assert.deepEqual(brandOf('Qwen3.8-27B'), { id: 'qwen', label: 'Qwen (Alibaba)' })
    assert.equal(hardwareOf('SiFive P870')?.id, 'sifive')
    assert.equal(hardwareOf('Falcon'), null)                     // a model maker's words say nothing of a chip
    assert.ok(builtIn('hardware').some((b) => b.id === 'apple'))
  } finally { setCustomBrands([]) }
})

test('the known makers read as plain words in Settings', () => {
  const ms = builtIn('model').find((b) => b.id === 'microsoft')
  assert.equal(ms?.words, 'phi-…, microsoft')
  assert.equal(builtIn('model').find((b) => b.id === 'zai')?.words, 'glm, z.ai, zhipu')
})
