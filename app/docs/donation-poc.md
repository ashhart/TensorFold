# TensorFold Community (POC)

Lend a machine's inference for a window, and use a machine somebody else lends. No payment and no
proof of computation: a card names the host, the token, the model, the window and the caps, and both
sides take the rest on trust.

## What a card is

`schema: tensorfold-donation/1` (`app/shared/donation.ts`) with:

| Field | Meaning |
|---|---|
| `endpoint` | Origin only, no path or credentials. A `tls` card must be https; a `poc-insecure-lan` card is plain http and the app reaches it through a loopback forward. |
| `token` | Bearer token the host enforces on every inference route. |
| `model`, `revision` | The served model id and optional immutable revision. |
| `window` | `from`/`until` epoch milliseconds. Outside it the host answers 403. |
| `caps` | `maxConcurrent`, `maxTokensPerRequest`, `maxRequestBytes`. Over a cap the host answers 429/400/413. |
| `spec` | What the host reports: hardware, memory, parameters, quantization, context, measured tokens/s and first-token time, engine version, notes. Shown, not trusted. |

Tokens never leave the host's profile: the renderer sees a card without its token, and the donor's
`state/donation.json` is written mode 0600.

## Lend

1. Models → **Lend or borrow** → **Lend this machine**. Fill in the address renters reach, the local
   engine, the model, the window and the caps; **Create card**.
2. Copy the donor config and run it next to the engine:

   ```bash
   node donation-node.js --config donation.json
   ```

   The node is a small authenticated proxy (`app/electron/donation-node-main.ts`, bundled to
   `dist-electron/donation-node.js`). It forwards `/v1/models`, `/v1/chat/completions` and
   `/v1/responses` to the local engine, enforces the card, and prints one JSON receipt per request
   with usage and `token_sha`. It sends `identity` encoding upstream, drops hop-by-hop framing,
   reserves a concurrency slot before the body is read, slows the engine when the renter stops
   reading, and re-checks the window at the moment a request would reach the engine. An omitted
   output limit becomes the card's cap; an explicit one must be a whole number within it; only
   the card's model is served and one completion per request. `upstreamTimeoutMs` (1000–600000)
   sets the idle timeout.
3. Hand the card to whoever should use it.

A non-loopback listener requires a `poc-insecure-lan` card. A `tls` card needs
`tls: {certFile, keyFile}` in the donor config.

## Borrow

1. Paste the card under Models → **Lend or borrow** → **Borrow a machine**. Import applies the same
   endpoint policy as any remote provider (https, or plain HTTP only to loopback, never link-local or
   metadata), pins each probe to the address it resolved to, and reads at most 256 KiB of the answer.
   It probes `/health` and `/v1/models` with the card's token and model before saving anything; a card
   whose window has not opened yet is accepted and verified by its first in-window request. The card
   and its provider row are written as one serialized change, rolled back together on failure.
2. The borrowed model appears under **TensorFold Community** on the Models page and in onboarding
   step 2, with the host's reported specs. **Use** opens a chat on that host.
3. The provider is registered as `donation:<card id>` with `remoteCompute: true`, so it is classified
   as external: agent runs plan with it and keep tools local.

A borrowed endpoint must be https or a loopback forward. Over a LAN, open the forward yourself:

```bash
ssh -N -L 8020:127.0.0.1:8020 you@host
```

and give the card `"endpoint": "http://127.0.0.1:8020"`.

## Out of scope in the POC

- No money, credits or ledger.
- No proof that the host ran the model it claims, and no spot-check replay.
- No directory: cards travel out of band.
- No multi-machine model splits; one host serves whole requests.
- The host's own engine restarts are visible as 502/500 responses; the app reports them rather than
  falling back to another model.
