// Embedded release-manifest trust root.
//
// This is a PUBLIC Ed25519 SPKI key, not a credential. Production update checks always use this
// compiled-in root: environment variables cannot replace it or disable signature verification.
// Key rotation therefore requires a reviewed application release that embeds the next public key.
// `build/release-trust.json` is the one effective source consumed by both this runtime module and
// the release preflight. Keeping the values in strict JSON prevents comments or string literals in
// TypeScript source from impersonating the constants that the packaged application actually uses.

import releaseTrust from '../../build/release-trust.json'

const updateTrust: {
  id: string
  productionReady: boolean
  ed25519PublicKey: string
} = releaseTrust.update

export const VONTRA_UPDATE_ED25519_PUBLIC_KEY = updateTrust.ed25519PublicKey
export const VONTRA_UPDATE_TRUST_ROOT_ID = updateTrust.id
export const VONTRA_UPDATE_TRUST_ROOT_PRODUCTION_READY = updateTrust.productionReady

export function updateTrustRootPublicKey(): string {
  return VONTRA_UPDATE_ED25519_PUBLIC_KEY
}
