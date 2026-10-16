import { createConnectorFetch } from '../../agent/secure-fetch'
import { store } from '../state'

/**
 * WhatsApp Cloud API reply helper. Standalone — only closes over the central store's
 * WhatsApp settings, which are read at call time so a freshly-persisted token takes effect
 * immediately. Kept on its own module (extracted verbatim from main-app.ts) so the webhook
 * server and main-app can both reference it without pulling the rest of the boot wiring.
 */
export async function sendWhatsAppCloudReply(to: string, text: string): Promise<void> {
  const settings = store.settings.botControl.whatsapp
  const endpoint = `https://graph.facebook.com/v23.0/${encodeURIComponent(settings.phoneNumberId)}/messages`
  const response = await createConnectorFetch(endpoint, 'WhatsApp Cloud API')(endpoint, {
    method: 'POST',
    headers: { authorization: `Bearer ${settings.accessToken}`, 'content-type': 'application/json' },
    body: JSON.stringify({ messaging_product: 'whatsapp', recipient_type: 'individual', to, type: 'text', text: { body: text.slice(0, 4096), preview_url: false } }),
    redirect: 'error',
    signal: AbortSignal.timeout(12_000),
  })
  if (!response.ok) throw new Error(`WhatsApp Cloud reply failed (${response.status}).`)
}
