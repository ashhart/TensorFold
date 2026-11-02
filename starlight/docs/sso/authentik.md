# authentik

authentik over OpenID Connect or SAML 2.0. The examples use Starlight at `https://starlight.example.com`, authentik at
`https://auth.example.com`, the application slug `starlight` and the connection slug `authentik`.

## OpenID Connect

1. Create an **OAuth2/OpenID Provider** with client type Confidential and the strict redirect URI
   `https://starlight.example.com/auth/authentik/callback`.
2. Choose a **signing key**. Without one, authentik signs tokens with HMAC, which Starlight refuses.
3. Create an application with the slug `starlight` and attach the provider.
4. Copy the client ID and client secret.

| Starlight field | Value |
| --- | --- |
| authentik URL | `https://auth.example.com` |
| Application slug | `starlight` |
| Client ID, client secret | From step 4 |

Starlight derives the issuer `https://auth.example.com/application/o/starlight/`.

### Groups

authentik's default profile scope mapping includes a `groups` claim with the user's group names; Starlight reads it.

## SAML 2.0

1. Create a **SAML Provider** with the ACS URL `https://starlight.example.com/auth/authentik/saml/acs`, the audience
   `https://starlight.example.com/auth/authentik/saml/metadata`, POST binding and a signing certificate.
2. Attach it to an application and copy the provider's metadata URL into Starlight.

## Check it works

Sign in from `https://starlight.example.com/auth/login`.
