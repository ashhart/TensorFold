// Deliberately narrow desktop-only exclusions. No native binding or ONNX web entrypoint is removed.
export const VOICE_RUNTIME_EXCLUSIONS = [
  'node_modules/kokoro-js/dist/*.map',
  'node_modules/@huggingface/transformers/dist/*.map',
  'node_modules/onnxruntime-web/dist/*.map',
  'node_modules/onnxruntime-node/dist/*.map',
  'node_modules/@met4citizen/headtts/dictionaries/fi.txt',
  'node_modules/@met4citizen/headtts/modules/language-fi.mjs',
  'node_modules/@met4citizen/headtts/modules/headtts.mjs',
  'node_modules/@met4citizen/headtts/modules/headtts-node.mjs',
  'node_modules/@met4citizen/headtts/modules/worker-tts.mjs',
]

export const VOICE_REQUIRED_FILES = {
  'kokoro-js': ['package.json', 'dist/kokoro.cjs', 'LICENSE'],
  '@huggingface/transformers': ['package.json', 'dist/transformers.node.cjs', 'dist/transformers.node.mjs', 'LICENSE'],
  'onnxruntime-web': ['package.json', 'dist/ort.node.min.js', 'dist/ort.node.min.mjs'],
  'onnxruntime-node': ['package.json', 'dist/index.js', 'bin/napi-v3/darwin/arm64/onnxruntime_binding.node', 'bin/napi-v3/darwin/arm64/libonnxruntime.1.21.0.dylib'],
  '@met4citizen/headtts': ['package.json', 'modules/language-en-us.mjs', 'modules/language.mjs', 'modules/utils.mjs', 'dictionaries/en-us.txt', 'LICENSE'],
  phonemizer: ['package.json', 'index.cjs', 'LICENSE'],
}
