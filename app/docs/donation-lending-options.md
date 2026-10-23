# Lending: how a renter reaches a host, and what exactly is being lent

Status: options for the donation POC. Two questions are open — how a renter reaches a donor, and
what exactly is being lent. This records the choices, what is built, and what each next slice costs.

---

## 1. Addressing: how does a renter reach a donor?

Today a card is a URL + token, and the in-app donor binds `127.0.0.1` only — so in practice a
renter needs an SSH tunnel or a third-party tunnel. Four ways to give it a public address:

| Option | Donor runs | Renter types | Cost | Weakness |
|---|---|---|---|---|
| **A. Relay, card id in the path** | nothing extra — the app dials out | `https://relay/<card-id>/v1` | one Worker + DO | relay terminates TLS and can read prompts |
| B. Wildcard subdomain per card | nothing extra (same relay underneath) | `https://<card-id>.lend.example` | wildcard cert + per-id DNS | purely cosmetic; more moving parts than A |
| C. User-run tunnel (cloudflared, Tailscale, SSH) | a second service | whatever the tunnel gives | free for the project | terminal required; dies with the tunnel; donor must keep it alive |
| D. WebRTC + signalling + TURN | nothing extra | nothing — card carries signalling info | signalling Worker + TURN VM | TURN bandwidth, NAT edge cases, and the peer must live in a renderer because Electron main has no WebRTC |

**Recommendation: A**, with C kept as the "I already have a tunnel" escape hatch. B can be added
later as a cosmetic layer over A if a prettier link matters. D is the right answer for donors who
refuse to route through anything, but it is a second transport to build and maintain, and it does
not remove a server — signalling and TURN remain.

**Built so far:** the relay, the donor dialer and the card-path allowance.
`relay/` is a Worker + one Durable Object per card id (deployable to any Cloudflare account, or self-hosted behind the same protocol); `app/electron/donation-relay-client.ts`
attaches outbound and replays frames against the local donation node; `app/shared/donation.ts`
now accepts a card endpoint with exactly one path segment. Verified through `wrangler dev`
(real workerd) with a real model: models list, streamed chat (first token 4.42 s), 401 from the
donor passed through, 503 with no donor, 413 on an oversized body, and a second donor key refused.

**Not built:** the donate flow does not mint a relay card yet (no donor-key storage, no
auto-attach, no reconnect surfaced in the UI), there is no relay configuration UI, no payload
encryption, and no abuse handling. Those are the remainder of slice A, listed in
`app/docs/donation-relay.md`.

---

## 2. Lending hardware without a model

Today's card is model-bound: `model` is required, the renter's provider gets exactly that one
model, and the donor only answers `/v1/models`, `/v1/chat/completions`, `/v1/responses`.

| Option | What the renter gets | Reuses | Missing |
|---|---|---|---|
| **A. Renter picks from the host's models** | inference on a model the donor can stage | card, donor proxy, `cluster-node` staging (`pull repo@rev`), admission checks, journal | card v2 (`models[]` or a model request), donor node gains ensure/start, spec measured after load, picker + provider row |
| B. Renter supplies a container image | raw GPU inside a digest-pinned image | nothing beyond the design work already written | donor agent that owns Docker, image pinning/allowlist, GPU UUID assignment, read-only model volumes, egress policy, drain/cleanup, receipts |
| C. Time-sliced bare GPU via a job API | a job slot, not an OpenAI endpoint | almost nothing | a second protocol, sandboxing, and every trust question at once |

**Recommendation: A first.** It keeps the trust question as "which model" rather than "which
code", it reuses the cluster node's immutable-revision staging and memory admission, and it fits
the card shape with one new field plus a donor route. B is a different product: it turns the free
POC into a compute marketplace and the abuse surface (mining, driver exploits, thermal, egress)
arrives all at once.

Honest hard part in A: the card's `spec` is hand-typed today. If the renter chooses the model, the
donor must measure parameters, quantisation, context and speed *after* loading and publish those
numbers, or borrowed cards advertise specs belonging to a different model.

---

## 3. Slice order

1. **Relay (done, uncommitted):** Worker + DO, dialer, card path allowance, docs, tests.
2. **Donate flow on the relay:** donor key, attach on start, reconnect, relay URL in the card,
   stop on window end; a relay field in settings or one built-in relay.
3. **Card v2, model choice:** `models[]` (or `modelRequest`), donor `ensure`/`start`, measured
   spec, picker support.
4. **Payload encryption** for the relay (card-token-derived key), if the relay is not self-hosted.
5. **Donor agent + containers (B)** only after 1–4 are in use.

## 4. Risks to settle before any of this is public

- The relay sees prompts unless payloads are encrypted. Decide before inviting strangers.
- Without per-IP limits and revocation, a public relay is an open proxy to every lent card.
- A relay card depends on whoever operates the relay; donors should see that dependency in the UI.
- Model-choice (slice 3) lets a renter make the donor download weights and thrash VRAM; needs a
  donor-side allowlist, disk quota and a swap budget.
- Container lending (B) needs someone accountable for what runs on the donor's GPU. That is a
  policy question, not a code one.
