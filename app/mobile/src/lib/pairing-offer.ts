import { parsePairingInput } from './client'

/** What a link opened from outside the app asks for: a host to pair with, or the reason the link cannot be used. */
export type PairingOffer = { link: string; baseUrl: string } | { problem: string }

/** Reads a link another app or page opened. A pairing link is only an offer; the user accepts the host it names first. */
export function outsidePairingLink(url: string | null, allowAndroidEmulator = false): PairingOffer | null {
  if (!url) return null
  let link: string | null
  try {
    const parsed = new URL(url)
    link = parsed.protocol === 'tensorfold-mobile:' ? parsed.searchParams.get('link') : url
  } catch { return null }
  if (!link || !/[#?&]pair=/.test(link)) return null
  if (!/^https?:\/\//i.test(link)) return { problem: 'This pairing link is invalid.' }
  try { return { link, baseUrl: parsePairingInput(link, undefined, allowAndroidEmulator).baseUrl } }
  catch (failure) { return { problem: (failure as Error).message || 'This pairing link is invalid.' } }
}
