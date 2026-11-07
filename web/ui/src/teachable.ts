// Explain when a served model cannot learn; learning requires an owned engine and a supported learning copy.

import type { EngineInfo, LivingView } from './api'

export function whyNotTeach(engine: EngineInfo | null, living: LivingView | null): string | null {
  if (!engine || engine.status !== 'running' || !living) return null
  const name = engine.model_name ?? engine.model ?? 'The AI'
  if (engine.managed === false || !engine.owned) return `${name} runs on another engine: teaching needs the AI started by this TensorFold Web.`
  if (living.teachable) return `Learning cannot run on this machine: ${living.teachable}.`
  const copy = living.served ? living.copies.find((c) => c.name === living.served) : null
  if (copy) return copy.learnable ? null : `${copy.name} is a copy of ${copy.family ?? 'a model'}, a family TensorFold does not learn into yet: Nemotron only, today.`
  const repo = engine.model_repo ?? engine.model
  if (repo && !living.bases.includes(repo)) return `${name} is not a model TensorFold learns into yet: Nemotron 3.5 Lightning only, today.`
  return null
}
