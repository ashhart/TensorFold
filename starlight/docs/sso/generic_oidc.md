# Other OpenID Connect providers

Any provider that implements OpenID Connect discovery and the authorization code flow. The examples use Starlight at
`https://starlight.example.com` and the slug `corp`.

## Requirements

- Discovery at `<issuer>/.well-known/openid-configuration`, with an `issuer` that matches exactly (a trailing slash
  difference is tolerated).
- Authorization code flow with PKCE support, a confidential client, and HTTPS endpoints.
- ID tokens signed with RS, PS, ES or EdDSA algorithms.

Register the redirect URI `https://starlight.example.com/auth/corp/callback`, and the back-channel logout URI
`https://starlight.example.com/auth/corp/backchannel-logout` if the provider supports it.

## Starlight fields

| Field | Use |
| --- | --- |
| Issuer | The provider's issuer URL |
| Client ID, client secret | Starlight's registration |
| Client authentication | `client_secret_basic`, `client_secret_post` or `private_key_jwt` with a key and certificate |
| Generate a key and certificate | Creates the `private_key_jwt` pair; register the certificate or the JWK the console shows |
| Key ID | Only when the provider assigned the key an ID; otherwise Starlight sends the key's thumbprint |
| Scopes | `openid` is always sent; add what your provider needs |
| Stable subject claim | `sub` unless the provider recommends another immutable claim |
| Email, name, groups claims | Claim names when they differ from `email`, `name` and `groups` |
| Use userinfo | Fills claims missing from the ID token; the userinfo `sub` must match |
| Trust email | Only for providers that never release unverified emails and do not send `email_verified` |

## Check it works

Sign in from `https://starlight.example.com/auth/login`. The audit entry for a refused sign-in names the stage and reason.
See [client authentication](README.md#client-authentication) for what a `private_key_jwt` assertion contains.
