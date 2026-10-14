import type { KokoroVoice } from '../agent/types/kokoro'

export const KOKORO_REVISION = '1939ad2a8e416c0acfeecc08a694d14ef25f2231'
export const KOKORO_MODEL_ID = 'onnx-community/Kokoro-82M-v1.0-ONNX'
export const KOKORO_DEFAULT_VOICE = 'af_heart'
export const KOKORO_VOICES: KokoroVoice[] = [
  ['af_heart', 'Heart', 'en-US'], ['af_bella', 'Bella', 'en-US'], ['af_nicole', 'Nicole', 'en-US'],
  ['af_nova', 'Nova', 'en-US'], ['af_sarah', 'Sarah', 'en-US'], ['af_alloy', 'Alloy', 'en-US'],
  ['am_michael', 'Michael', 'en-US'], ['am_fenrir', 'Fenrir', 'en-US'], ['am_puck', 'Puck', 'en-US'],
].map(([id, name, language]) => ({ id, name, language: language as KokoroVoice['language'] }))

export interface KokoroAsset { path: string; bytes: number; sha256: string; url: string }
const cacheBase = `https://huggingface.co/api/resolve-cache/models/${KOKORO_MODEL_ID}/${KOKORO_REVISION}`
export const KOKORO_ASSETS: KokoroAsset[] = [
  { path: 'config.json', bytes: 44, sha256: 'df34b4f930b23447cd4dc410fabfb42eb3f24e803e6c3f97d618fb359380a36f', url: `${cacheBase}/config.json` },
  { path: 'tokenizer.json', bytes: 3497, sha256: '77a02c8e164413299b4b4c403b14f8e0e1c1b727db4d46a09d6327b861060a34', url: `${cacheBase}/tokenizer.json` },
  { path: 'tokenizer_config.json', bytes: 113, sha256: 'be1cb066d6ef6b074b3f15e6a6dd21ac88ff3cdaedf325f0aaed686c70f75d20', url: `${cacheBase}/tokenizer_config.json` },
  { path: 'onnx/model_quantized.onnx', bytes: 92361116, sha256: 'fbae9257e1e05ffc727e951ef9b9c98418e6d79f1c9b6b13bd59f5c9028a1478', url: `https://huggingface.co/${KOKORO_MODEL_ID}/resolve/${KOKORO_REVISION}/onnx/model_quantized.onnx` },
]
export const KOKORO_DOWNLOAD_BYTES = KOKORO_ASSETS.reduce((sum, asset) => sum + asset.bytes, 0)
