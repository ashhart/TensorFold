# Single sign-on

Starlight signs people in through the organisation's own identity provider (IdP). It is an OpenID Connect relying party
and a SAML 2.0 service provider, so it works with every mainstream workforce IdP, and with any standards-compliant one
through the generic presets. All SSO features are part of the open-source build; there is no seat limit.

## Choose a guide

| Identity provider | OIDC | SAML | Groups from | Guide |
| --- | --- | --- | --- | --- |
| Okta | Yes | Yes | `groups` claim or attribute, SCIM | [okta.md](okta.md) |
| Microsoft Entra ID (Azure AD) | Yes | Yes | `groups` claim, Microsoft Graph on overage, SCIM | [entra.md](entra.md) |
| Google Workspace | Yes | Yes | Admin SDK Directory API, SAML group membership | [google.md](google.md) |
| Auth0 | Yes | | Custom claim from an Action | [auth0.md](auth0.md) |
| OneLogin | Yes | Yes | Groups parameter | [onelogin.md](onelogin.md) |
| PingOne | Yes | Yes | Attribute mapping | [pingone.md](pingone.md) |
| PingFederate | Yes | Yes | Attribute contract | [pingfederate.md](pingfederate.md) |
| JumpCloud | Yes | Yes | Group attribute | [jumpcloud.md](jumpcloud.md) |
| Keycloak | Yes | Yes | Group Membership mapper | [keycloak.md](keycloak.md) |
| authentik | Yes | Yes | `groups` claim | [authentik.md](authentik.md) |
| ZITADEL | Yes | | Project roles | [zitadel.md](zitadel.md) |
| AWS IAM Identity Center | | Yes | Attribute mapping | [aws_identity_center.md](aws_identity_center.md) |
| AD FS | Yes | Yes | Token-Groups claim rule | [adfs.md](adfs.md) |
| Shibboleth IdP | | Yes | `isMemberOf` | [shibboleth.md](shibboleth.md) |
| Duo Single Sign-On | Yes | Yes | Group attribute | [duo.md](duo.md) |
| GitLab | Yes | | `groups_direct` claim | [gitlab.md](gitlab.md) |
| Any OIDC provider | Yes | | Any claim | [generic_oidc.md](generic_oidc.md) |
| Any SAML 2.0 provider | | Yes | Any attribute | [generic_saml.md](generic_saml.md) |

## URLs to register at the IdP

Every connection has a slug, a short lowercase name chosen when it is created. With Starlight at `https://starlight.example.com`
and the slug `corp`:

| Purpose | URL |
| --- | --- |
| OIDC redirect (sign-in callback) | `https://starlight.example.com/auth/corp/callback` |
| OIDC back-channel logout | `https://starlight.example.com/auth/corp/backchannel-logout` |
| SAML Assertion Consumer Service | `https://starlight.example.com/auth/corp/saml/acs` |
| SAML entity ID and SP metadata | `https://starlight.example.com/auth/corp/saml/metadata` |
| SAML single logout | `https://starlight.example.com/auth/corp/saml/slo` |
| Sign-in page | `https://starlight.example.com/auth/login` |

Starlight must be reachable on its public URL over HTTPS. The SP metadata document lists Starlight's signing and encryption
certificate and its single logout endpoint for both bindings; IdPs that import metadata pick them up.

## First owner and break-glass access

`starlight init` prints a one-time setup link. Opening it lets an administrator create the first connection in a short
wizard and then sign in with it; the first account to finish becomes the owner, and every setup link stops working,
including spare ones issued with `starlight admin setup-code`. Setup links are refused once any owner exists.

If the IdP is down or misconfigured, an operator with access to the Starlight host can issue a recovery link for an existing
owner. It lasts at most fifteen minutes, works once, opens a confirmation page before it signs anyone in, and is
recorded in the audit log as `auth.recovery_login`. There are no standing local passwords to steal.

## What Starlight checks

OIDC sign-ins use the authorization code flow with PKCE (S256), a random `state` and `nonce`, and a server-side
request record that is used once and bound to the browser that started it. The ID token must be signed with an
asymmetric algorithm (RS, PS, ES or EdDSA; `none` and HMAC are refused), name Starlight as audience and authorized party,
come from the configured issuer, be unexpired, and carry the nonce. A requested `max_age` is enforced against
`auth_time`. Client secrets, signing keys and service-account keys are sealed at rest.

SAML sign-ins send signed HTTP-Redirect AuthnRequests and accept only the HTTP-POST binding. A response must be signed
by the IdP's metadata certificate, answer the request Starlight sent, name Starlight's ACS as recipient and Starlight's entity ID as
audience, carry a bearer subject confirmation and be within its validity window. Each assertion ID is accepted once.
IdP-initiated sign-in is off unless a connection turns it on. Encrypted assertions are supported. Before any signature
work, a response is refused if it nests deeper than 32 levels, gives an element more than 32 attributes, holds more than
20,000 elements or more than one assertion, or has a signature anywhere but directly under the response or its
assertion. Signatures and digests must use SHA-256 or stronger. The ACS and single logout endpoints are rate limited
per client network.

SAML logout messages must be signed by a current certificate from the IdP metadata: in the query string for the
HTTP-Redirect binding (RSA with SHA-256 or stronger), or for HTTP-POST with one enveloped XML signature directly under
the message's root, also SHA-256 or stronger. Unsigned messages and messages over 32 KiB are refused. Each must name the IdP as issuer and Starlight's single logout URL as destination, and be no more
than ten minutes old. A LogoutResponse must answer a LogoutRequest Starlight sent and is accepted once; each LogoutRequest
ID from the IdP is accepted once.

## How accounts are linked

A person is identified by the connection's immutable subject: `sub` for OIDC, the NameID or a configured attribute for
SAML, and the tenant and object IDs for Microsoft Entra ID. Email is never the key.

- Each email domain belongs to at most one connection, the one that lists it under **Domains**. A connection can never
  sign anyone in with an email on another connection's domain, link to an account another connection provisioned, or
  create one there, so a partner or contractor IdP cannot claim an employee's account by asserting their address.
- A known subject signs in to its existing account, unless that account is suspended or deactivated.
- A new subject links to an existing account with the same email only when the email is on one of the connection's own
  domains (for Google, also its Workspace domains) and the IdP proves it (`email_verified`, or Entra's `xms_edov`), or
  when SCIM provisioned that account through the same connection.
- Otherwise, with **Create users on first sign-in** on, Starlight creates the account if the email is proven and not on
  another connection's domain. With it off, only provisioned people can sign in. SCIM follows the same domain rule.

SCIM-provisioned profiles are owned by SCIM; sign-in does not overwrite their email or name. See
[SCIM provisioning](../scim.md).

## Groups

Groups drive project membership and roles. Each connection uses one strategy:

| Strategy | Source |
| --- | --- |
| `claim` | A claim or SAML attribute named in the connection, synced at every sign-in |
| `entra_graph` | The `groups` claim, and Microsoft Graph when Entra omits groups because a user has too many |
| `google_directory` | The Admin SDK Directory API through a delegated service account, optionally including nested groups |
| `scim` | SCIM only; sign-in never changes memberships |
| `none` | No groups |

Claim-sourced memberships are replaced on each sign-in, so removing someone from an IdP group removes it in Starlight the
next time they sign in. SCIM keeps groups current between sign-ins. **Required groups** limits a connection to members
of at least one listed group.

## Email routing and several connections

A connection can claim email domains. The sign-in page then asks for a work email and sends each person to the
connection that owns their domain; buttons for every enabled connection stay available. A domain belongs to at most one
connection.

## Logout

Signing out always ends the Starlight session first. With **Sign out of the IdP too** on, Starlight then sends the browser to
the IdP as well:

- OIDC: to the end-session endpoint (RP-initiated logout).
- SAML: with a signed LogoutRequest that repeats the NameID and SessionIndex from sign-in. Starlight uses the IdP's
  HTTP-Redirect single logout endpoint, or HTTP-POST when that is the only one. The IdP's LogoutResponse returns to
  Starlight's single logout URL. When it passes the checks and reports success, the browser lands on the signed-out page.
  If the IdP metadata lists no single logout endpoint, sign-out ends the Starlight session only.

The IdP can end Starlight sessions too, whether or not that setting is on:

- OIDC Back-Channel Logout: Starlight verifies the logout token (signature, issuer, audience, the logout event, no nonce, a
  fresh `iat` and a single-use `jti`) and ends the matching sessions.
- SAML single logout: Starlight accepts a signed LogoutRequest at its single logout URL over either binding. It ends every
  session for that NameID on the connection, or only the named sessions when the request lists SessionIndex values,
  and answers with a signed LogoutResponse.

Disabling or deleting a connection signs out everyone with a session through it, spends the sign-in approvals those
sessions gave, and revokes the Connect and CLI tokens of everyone who signs in through it. API keys and service
accounts are unaffected.

Starlight signs SAML logout messages with the connection's SP key even when it does not sign AuthnRequests, so its metadata
always lists the signing certificate. The audit log records `auth.logout` when someone signs out of Starlight,
`auth.logout_response` for the IdP's answer, and `auth.idp_logout` for logouts the IdP starts. A refused message is
recorded with the stage and reason.

## Client authentication

An OIDC connection proves its identity at the IdP's token endpoint with a client secret or with a certificate.

| Method | What Starlight sends |
| --- | --- |
| `client_secret_basic` | The client secret in an HTTP Basic header; the default |
| `client_secret_post` | The client secret in the request body |
| `private_key_jwt` | A client assertion: a JWT signed with the connection's private key |

With `private_key_jwt`, every token request carries a new assertion. Its `iss` and `sub` are the client ID, `aud` is the
token endpoint, `jti` is random, and it expires three minutes after its `iat` and `nbf`. RSA keys sign with RS256 and EC
P-256 keys with ES256. The header names the key by `kid` and, when the connection has a certificate, by `x5t` (the
certificate's SHA-1 thumbprint, which Microsoft Entra ID requires) and `x5t#S256`. Calls to Microsoft Graph for Entra
group overage use the same certificate. The client secret is never sent.

To set it up, choose `private_key_jwt`, then either tick **Generate a key and certificate** or paste the PEM private key
into its field and the certificate alone into its own. Starlight seals the key at rest, as it does client secrets; the
certificate field is not sealed, so one that also holds a key, a chain or any other PEM block is refused. The connection page shows the certificate,
its SHA-1 thumbprint and expiry, a download link, and the public key as a JWK for providers that register keys that
way. Generated certificates last two years. **Replace with a new key and certificate** takes effect at once, so upload
the new certificate straight away. The `kid` is the key's RFC 7638 thumbprint unless you set **Key ID**, for example to
an ID the IdP assigned. The setup wizard accepts a pasted key and certificate; generating a pair needs the console.

## IdP access tokens for scripts and harnesses

An OIDC connection can accept the IdP's own JWT access tokens as bearer credentials on Starlight's API, for automation that
already holds them. Turn on **Accept IdP access tokens on the API** (`accept_bearer`) and list the audience your IdP
puts in access tokens for Starlight (`bearer_audiences`).
Starlight checks the signature, issuer, expiry and audience, refuses ID tokens (an access token must carry `scp`, `scope` or
`roles`), and maps the token to an account that already exists; it never creates accounts from bearer tokens.

## Troubleshooting

| Message | Meaning |
| --- | --- |
| Sign-in expired | The attempt took longer than ten minutes, was replayed, or came back in a different browser |
| Account not set up | No account exists and the connection does not create users, or provisioning has not run yet |
| Account conflict | Someone with this email exists and the IdP did not prove the email; provision the account through SCIM from this connection, or have the IdP verify emails |
| Wrong organisation | Entra tenant or Google Workspace domain does not match the connection |
| Email not verified | The IdP did not verify the email; mark the connection trusted only if your IdP always verifies |
| Access not granted | The account is not in any of the connection's required groups |
| Incomplete sign-in | The IdP sent no stable subject or no email; check the claim and attribute names |
| Groups unavailable | Microsoft Graph or the Directory API refused Starlight's request; check the credentials and consent |
| Sign-in failed | A signature, audience, issuer or time check failed; the audit log entry for `auth.login` has the reason |
| Sign-out not confirmed | Starlight ended its session, but the IdP's LogoutResponse failed a check; see `auth.logout_response` |
| Sign-out incomplete | The IdP answered the LogoutRequest with an error status, so its own session may still be open |
| Sign-out request refused | A LogoutRequest was unsigned, forged, stale or replayed; see `auth.idp_logout` |

Every refusal is written to the audit log with the connection and stage, and the page shows a reference that matches
the request ID in Starlight's logs.
