import assert from 'node:assert/strict'
import { describeFormat, formatName, parseCapabilities, parseInfoOutput, parseModelsOutput, parseVersion, runEngineCli } from './cli'
import { makeHarness, seedModel } from './test-fixtures/harness'

async function main() {
  // Text exactly as 1.0.7 prints it (zig/src/cli/{models,info}.zig tests).
  assert.equal(parseVersion('tensorfold-native 1.0.7\n'), '1.0.7')
  assert.equal(parseVersion('something else'), null)

  assert.deepEqual(parseModelsOutput([
    'Org/Flash: Qwen3.8 Flash Next (qwen4_exp)',
    'TensorFold/Qwen3.8-27B-MLX-4bit: Qwen3.5-2B or Qwen3.8-27B (qwen3_5)',
    'mlx-community/Qwen3.5-2B-MLX-4bit: Qwen3.5-2B or Qwen3.8-27B (qwen3_5)',
  ].join('\n')), [
    { repo: 'Org/Flash', family: 'Qwen3.8 Flash Next', modelType: 'qwen4_exp' },
    { repo: 'TensorFold/Qwen3.8-27B-MLX-4bit', family: 'Qwen3.5-2B or Qwen3.8-27B', modelType: 'qwen3_5' },
    { repo: 'mlx-community/Qwen3.5-2B-MLX-4bit', family: 'Qwen3.5-2B or Qwen3.8-27B', modelType: 'qwen3_5' },
  ])
  assert.deepEqual(parseModelsOutput('no checkpoints on disk that a registered Zig family can serve; run: tensorfold pull OWNER/NAME\n'), [])

  const info = parseInfoOutput([
    'family: Qwen3.8 Flash Next (qwen4_exp)',
    'format: 6-bit, group 32',
    'context: 65536',
    'weights: 12.00 GiB (12884901888 bytes)',
    'memory floor: 14.00 GiB (weights plus a 2.00 GiB runtime margin)',
  ].join('\n'))
  assert.deepEqual(info, { family: 'Qwen3.8 Flash Next', modelType: 'qwen4_exp', format: '6-bit, group 32', contextTokens: 65536, weightsBytes: 12884901888, memoryFloorBytes: 12884901888 + 2 * 1024 ** 3 })
  assert.throws(() => parseInfoOutput('Org/X is not in the Hugging Face cache; run: tensorfold pull Org/X\n'), /not in the Hugging Face cache/)

  const caps = parseCapabilities('{"schema": 1, "engine": "zig", "version": "1.0.7", "chip": null, "backends": ["auto", "mlx"], "families": {"nemotron_h": ["mlx-q4g64"], "qwen3_5": ["mlx-q4g64"]}, "serve": {"flags": {"--host": {}, "--drafter": {}}}, "env": ["TENSORFOLD_API_KEY"]}\n')
  assert.deepEqual(caps.families, { nemotron_h: ['mlx-q4g64'], qwen3_5: ['mlx-q4g64'] })
  assert.deepEqual(caps.flags, ['--host', '--drafter'])
  assert.throws(() => parseCapabilities('not json'), /did not print JSON/)
  assert.throws(() => parseCapabilities('{"engine": "python"}'), /Unexpected capabilities/)

  assert.equal(formatName({ quantization: { bits: 4, group_size: 64 } }), 'mlx-q4g64')
  assert.equal(formatName({ quantization: { bits: 6 } }), 'mlx-q6g32', 'a missing group size is 32, as hub.zig reads it')
  assert.equal(formatName({}), null)
  assert.equal(describeFormat('mlx-q4g64'), '4-bit, group 64')
  assert.equal(describeFormat(null), 'full precision')

  // The wrappers against the fake engine: exit codes come back, nothing throws on a refusal.
  const h = await makeHarness()
  try {
    assert.equal(parseVersion((await runEngineCli(h.binary, ['--version'], h.env)).stdout), '1.0.7')
    assert.deepEqual(parseModelsOutput((await runEngineCli(h.binary, ['models'], h.env)).stdout), [])
    await seedModel(h.hub, 'mlx-community/Qwen3.5-2B-MLX-4bit', { bytes: 4096 })
    await seedModel(h.hub, 'Org/Gemma', { modelType: 'gemma4' })
    assert.deepEqual(parseModelsOutput((await runEngineCli(h.binary, ['models'], h.env)).stdout).map((m) => m.repo), ['mlx-community/Qwen3.5-2B-MLX-4bit'])
    const ok = await runEngineCli(h.binary, ['info', 'mlx-community/Qwen3.5-2B-MLX-4bit'], h.env)
    assert.equal(ok.code, 0)
    assert.equal(parseInfoOutput(ok.stdout).format, '4-bit, group 64')
    const refused = await runEngineCli(h.binary, ['info', 'Org/Gemma'], h.env)
    assert.equal(refused.code, 1)
    assert.match(refused.stdout, /no registered Zig family serves model_type gemma4/)
    await assert.rejects(() => runEngineCli({ command: '/nonexistent/tensorfold-native', args: [] }, ['--version'], h.env), /Could not run tensorfold-native/)
  } finally {
    await h.cleanup()
  }
  console.log('engine cli tests passed')
}

main().catch((error) => { console.error(error); process.exit(1) })
