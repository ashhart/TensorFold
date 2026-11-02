# PingOne

PingOne (Ping Identity's cloud service) over OpenID Connect or SAML 2.0, in any region. The examples use Starlight at
`https://starlight.example.com` and the slug `pingone`.

## OpenID Connect

1. In the PingOne admin console, add an application of type **OIDC Web App**.
2. Set the redirect URI to `https://starlight.example.com/auth/pingone/callback`, keep the Authorization Code grant, and
   use client secret basic authentication.
3. Add an attribute mapping named `groups` from the user's group names if Starlight should receive groups.
4. Turn the application on and grant access to the right groups.
5. Copy the client ID, client secret and environment ID.

| Starlight field | Value |
| --- | --- |
| Region domain | `auth.pingone.com`, `auth.pingone.eu`, `auth.pingone.ca`, `auth.pingone.asia` or `auth.pingone.com.au` |
| Environment ID | From the environment's settings |
| Client ID, client secret | From step 5 |
| Groups claim | `groups`, if you added the mapping |

Starlight derives the issuer `https://<region domain>/<environment ID>/as`, which matches the application's configuration
page.

## SAML 2.0

1. Add a SAML application. Set the ACS URL to `https://starlight.example.com/auth/pingone/saml/acs` and the entity ID to
   `https://starlight.example.com/auth/pingone/saml/metadata`.
2. Map `email`, `name` and optionally `groups` as SAML attributes.
3. Copy the IdP metadata URL into Starlight.

## Check it works

Sign in from `https://starlight.example.com/auth/login`.
