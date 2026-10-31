// Which maker a model comes from, read from any name it goes by: the catalog's ("Nemotron 3.5 Lightning"), its
// repository ("TensorFold/NVIDIA-Nemotron-..."), a teachable copy's ("nemotron-living") or another machine's.
// The model's own family logo when there is one (Qwen), else its maker's (NVIDIA for Nemotron, Google for Gemma).
// A maker with no logo file shows its initial (brandLogo.tsx), so a new family only needs its line here.

export type Brand = { id: string; label: string }
export type BrandKind = 'model' | 'hardware'
// a maker added or completed from Settings (tensorfold_web/brands.py): checked before the lists below, its logo first
export type CustomBrand = Brand & { kind: BrandKind; words: string[]; logo: string | null }

let custom: CustomBrand[] = []
export function setCustomBrands(list: CustomBrand[]) { custom = list }
export function customBrand(id: string, kind: BrandKind): CustomBrand | undefined {
  return custom.find((c) => c.id === id && c.kind === kind)
}

// a Settings entry whose words a name holds, else the list's rule; a Settings entry of the same id renames it
function find(kind: BrandKind, rules: [RegExp, Brand][], names: (string | null | undefined)[]): Brand | null {
  for (const n of names) {
    if (!n) continue
    const low = n.toLowerCase()
    const own = custom.find((c) => c.kind === kind && c.words.some((w) => low.includes(w)))
    if (own) return { id: own.id, label: own.label }
    const hit = rules.find(([re]) => re.test(n))
    if (hit) return { id: hit[1].id, label: customBrand(hit[1].id, kind)?.label ?? hit[1].label }
  }
  return null
}

// a rule's pattern as the words a person reads ("phi-…, microsoft" for /\bphi-?\d|microsoft/i)
const readable = (re: RegExp) => re.source.split('|')
  .map((w) => w.replace(/\\b|\^|\$/g, '').replace(/\\d\+?/g, '…').replace(/-\?/g, '-').replace(/\\(.)/g, '$1')).join(', ')

/** The makers TensorFold Web knows on its own, each once with the words that recognise it, for Settings to show. */
export function builtIn(kind: BrandKind): (Brand & { words: string })[] {
  const out = new Map<string, Brand & { words: string }>()
  for (const [re, b] of kind === 'model' ? RULES : HARDWARE) {
    const had = out.get(b.id)
    out.set(b.id, { ...b, words: had ? `${had.words}, ${readable(re)}` : readable(re) })
  }
  return [...out.values()]
}

const RULES: [RegExp, Brand][] = [
  [/bonsai|prism/i, { id: 'prism', label: 'Prism ML' }],          // before Qwen: Bonsai is built on a Qwen
  [/nemotron|nvidia/i, { id: 'nvidia', label: 'NVIDIA' }],
  [/qwen/i, { id: 'qwen', label: 'Qwen' }],
  [/gemma/i, { id: 'google', label: 'Google' }],
  [/\bglm|z\.ai|zhipu/i, { id: 'zai', label: 'Z.ai' }],
  [/deepseek/i, { id: 'deepseek', label: 'DeepSeek' }],
  [/llama/i, { id: 'meta', label: 'Meta' }],
  [/mistral|mixtral|ministral|devstral|magistral|codestral|pixtral|voxtral/i, { id: 'mistralai', label: 'Mistral AI' }],
  [/kimi|moonshot/i, { id: 'kimi', label: 'Moonshot AI' }],
  [/minimax/i, { id: 'minimax', label: 'MiniMax' }],
  [/smollm|huggingface/i, { id: 'huggingface', label: 'Hugging Face' }],
  [/ernie|baidu/i, { id: 'baidu', label: 'Baidu' }],
  [/seed-oss|bytedance/i, { id: 'bytedance', label: 'ByteDance' }],
  [/mimo|xiaomi/i, { id: 'xiaomi', label: 'Xiaomi' }],
  [/\bphi-?\d|microsoft/i, { id: 'microsoft', label: 'Microsoft' }],
  [/gpt-oss|openai/i, { id: 'openai', label: 'OpenAI' }],
  [/granite|\bibm\b/i, { id: 'ibm', label: 'IBM' }],
  [/olmo|allenai/i, { id: 'ai2', label: 'Ai2' }],
  [/command-r|\baya\b|cohere/i, { id: 'cohere', label: 'Cohere' }],
  [/hermes|nousresearch/i, { id: 'nous', label: 'Nous Research' }],
]

export function brandOf(...names: (string | null | undefined)[]): Brand | null {
  return find('model', RULES, names)
}

// Which maker built a machine, from its chip and backend: Apple for an M chip, NVIDIA for a GB10 or any CUDA GPU, AMD,
// Intel, Qualcomm
const HARDWARE: [RegExp, Brand][] = [
  [/apple|\bM\d+\b|^mlx$/i, { id: 'apple', label: 'Apple' }],
  [/nvidia|\bgb\d+|geforce|\brtx\b|^cuda$/i, { id: 'nvidia', label: 'NVIDIA' }],
  [/\bamd\b|ryzen|radeon|epyc|instinct|^rocm$/i, { id: 'amd', label: 'AMD' }],
  [/intel|xeon|core ultra|\barc\b|gaudi|^xpu$/i, { id: 'intel', label: 'Intel' }],
  [/qualcomm|snapdragon/i, { id: 'qualcomm', label: 'Qualcomm' }],
]

export function hardwareOf(...names: (string | null | undefined)[]): Brand | null {
  return find('hardware', HARDWARE, names)
}
