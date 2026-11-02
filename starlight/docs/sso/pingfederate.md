# PingFederate

A self-managed PingFederate server over OpenID Connect or SAML 2.0. The examples use Starlight at
`https://starlight.example.com` and the slug `pingfed`.

## OpenID Connect

1. Create an OAuth client with the Authorization Code grant, a client secret and the redirect URI
   `https://starlight.example.com/auth/pingfed/callback`.
2. Attach an OpenID Connect policy that issues an ID token with `email`, `email_verified` where available, `name` and,
   if Starlight should receive groups, a groups attribute.
3. Note PingFederate's issuer: the base URL shown in `https://pf.example.com/.well-known/openid-configuration`.

| Starlight field | Value |
| --- | --- |
| Issuer | The PingFederate base URL, for example `https://pf.example.com` |
| Client ID, client secret | From step 1 |
| Groups claim | The attribute name from step 2 |

If PingFederate does not release `email_verified` and its email attribute comes from your directory, an administrator
can mark the connection's email as trusted.

## SAML 2.0

1. Create an SP connection by importing Starlight's metadata from `https://starlight.example.com/auth/pingfed/saml/metadata`.
2. Enable browser SSO with the POST binding for responses and an attribute contract with `email`, `name` and `groups`.
3. Give Starlight the IdP metadata PingFederate publishes for the connection, by URL or as XML.
4. For single logout, enable the SP-initiated and IdP-initiated SLO profiles in the connection's browser SSO settings;
   the logout endpoints come from Starlight's metadata. Turn on **Sign out of the IdP too** in Starlight.

## Check it works

Sign in from `https://starlight.example.com/auth/login`.
