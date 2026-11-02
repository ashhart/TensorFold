# Duo Single Sign-On

Duo Single Sign-On as an OpenID Connect provider or SAML 2.0 IdP. The examples use Starlight at `https://starlight.example.com`
and the slug `duo`.

## OpenID Connect

1. In the Duo Admin Panel, protect a new application of type **Generic OIDC Relying Party**.
2. Add the sign-in redirect URL `https://starlight.example.com/auth/duo/callback` and allow the `openid`, `email` and
   `profile` scopes.
3. Copy the client ID, client secret and issuer that Duo shows for the application.

| Starlight field | Value |
| --- | --- |
| Issuer | The issuer from step 3 |
| Client ID, client secret | From step 3 |
| Groups claim | The claim name, if your Duo configuration sends group membership |

## SAML 2.0

1. Protect a **Generic SAML Service Provider** with the entity ID `https://starlight.example.com/auth/duo/saml/metadata`
   and the ACS URL `https://starlight.example.com/auth/duo/saml/acs`.
2. Map `email`, `name` and optionally `groups`, then give Starlight Duo's metadata URL or XML.

## Check it works

Sign in from `https://starlight.example.com/auth/login`.
