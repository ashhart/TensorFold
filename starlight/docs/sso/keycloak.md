# Keycloak

Keycloak, or the Red Hat build of Keycloak, over OpenID Connect or SAML 2.0. The examples use Starlight at `https://starlight.example.com`, Keycloak
at `https://sso.example.com`, the realm `example` and the slug `keycloak`.

## OpenID Connect

1. In the realm, create a client of type OpenID Connect with the client ID `starlight`.
2. Turn **Client authentication** on and keep the standard flow.
3. Set the valid redirect URI to `https://starlight.example.com/auth/keycloak/callback`.
4. Set the back-channel logout URL to `https://starlight.example.com/auth/keycloak/backchannel-logout` and require the
   back-channel logout session, so Starlight can end the exact session Keycloak logs out.
5. Copy the client secret from the **Credentials** tab.

| Starlight field | Value |
| --- | --- |
| Keycloak URL | `https://sso.example.com` |
| Realm | `example` |
| Client ID, client secret | `starlight` and the secret from step 5 |

Starlight derives the issuer `https://sso.example.com/realms/example`.

### Groups

Add a **Group Membership** mapper to the client's dedicated scope with the token claim name `groups`, full group path
off, and inclusion in the ID token on.

## SAML 2.0

1. Create a SAML client whose client ID is `https://starlight.example.com/auth/keycloak/saml/metadata`, or import Starlight's
   SP metadata from that URL.
2. Set the assertion consumer URL for POST binding to `https://starlight.example.com/auth/keycloak/saml/acs`.
3. Add mappers for `email`, `name` and a group list named `groups`.
4. Starlight's metadata URL is `https://sso.example.com/realms/example/protocol/saml/descriptor`; enter the Keycloak URL
   and realm and Starlight fills it in.
5. For single logout, keep **Front channel logout** on and set the logout service URLs for both bindings to
   `https://starlight.example.com/auth/keycloak/saml/slo`; importing Starlight's metadata fills them in. Keep **Sign documents**
   on, because Starlight accepts only signed logout messages. Turn on **Sign out of the IdP too** in Starlight.

## Check it works

Sign in, then sign out from Keycloak's account console. Over OIDC, Starlight's session ends through back-channel logout;
over SAML, through single logout.
