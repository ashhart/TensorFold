// Per-channel ephemeral system-prompt selection. A connector source can carry a
// `channelPrompt`; when an agent run is handling a message from that source, the prompt is
// injected as a SEPARATE, ephemeral system message (never persisted, never part of the
// system-prompt snapshot). This module is the pure selection + rendering logic so it
// unit-tests without a store or a model.

import { parseThreadAddress } from './thread-address'
import type { ApiMessage } from './types/base'
import type {
  DiscordConnectorSettings,
  EmailConnectorSettings,
  SlackConnectorSettings,
} from './types/connectors'

export interface ChannelPromptSources {
  slack?: Pick<SlackConnectorSettings, 'sources'>
  discord?: Pick<DiscordConnectorSettings, 'sources'>
  email?: Pick<EmailConnectorSettings, 'sources'>
}

/**
 * Resolve the channelPrompt override for a conversation, given the connector source
 * configs. The threadKey's account segment is the sourceId (slack:<src>:<chan>:<ts>,
 * discord:<src>:<chan>, email:<root>). Returns the trimmed prompt, or null when the
 * platform/source has none. Pure + tested.
 */
export function selectChannelPrompt(threadKey: string | null | undefined, sources: ChannelPromptSources): string | null {
  const address = parseThreadAddress(threadKey)
  if (!address) return null
  const sourceId = address.account
  const list =
    address.platform === 'slack' ? sources.slack?.sources :
    address.platform === 'discord' ? sources.discord?.sources :
    address.platform === 'email' ? sources.email?.sources :
    undefined
  if (!list) return null
  // Email keys carry no account segment (email:<root>); fall back to the first enabled
  // source that defines a channelPrompt so a single-account inbox still applies its prompt.
  const match = sourceId
    ? list.find((s) => s.id === sourceId)
    : list.find((s) => (s as { enabled?: boolean }).enabled !== false && hasPrompt(s))
  const prompt = match?.channelPrompt?.trim()
  return prompt ? prompt : null
}

function hasPrompt(source: { channelPrompt?: string }): boolean {
  return Boolean(source.channelPrompt && source.channelPrompt.trim())
}

/**
 * Wrap a channelPrompt as a labelled ephemeral system block. Keeps the override clearly
 * scoped ("for the current channel only") so the model treats it as channel etiquette, not
 * a durable system instruction. Returns '' for an empty prompt so callers can skip it.
 */
export function renderChannelPrompt(prompt: string | null | undefined): string {
  const text = String(prompt ?? '').trim()
  if (!text) return ''
  return `CHANNEL GUIDANCE (applies only to the current channel/conversation; not a durable instruction):\n${text}`
}

/**
 * Produce the message list to send to the model with the per-channel prompt injected as a
 * SECOND, ephemeral system message right after the first system message (the snapshot). The
 * returned array is a NEW array — `base` is never mutated — so the ephemeral block never lands
 * in thread.apiMessages nor the persisted snapshot (switching channels mid-thread can't leak
 * one channel's instructions into another's history). An empty/blank prompt returns `base`
 * unchanged (same reference). Pure + tested.
 */
export function withEphemeralChannelPrompt(base: ApiMessage[], channelPrompt: string | null | undefined): ApiMessage[] {
  const block = renderChannelPrompt(channelPrompt)
  if (!block) return base
  const ephemeral: ApiMessage = { role: 'system', content: block }
  const systemIndex = base.findIndex((m) => m.role === 'system')
  if (systemIndex < 0) return [ephemeral, ...base]
  return [...base.slice(0, systemIndex + 1), ephemeral, ...base.slice(systemIndex + 1)]
}
