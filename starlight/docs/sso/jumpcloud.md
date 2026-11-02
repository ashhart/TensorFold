# JumpCloud

JumpCloud over OpenID Connect or SAML 2.0. The examples use Starlight at `https://starlight.example.com` and the slug
`jumpcloud`.

## OpenID Connect

1. In the JumpCloud admin portal, add a **Custom OIDC App** under SSO applications.
2. Set the redirect URI to `https://starlight.example.com/auth/jumpcloud/callback` and the login URL to
   `https://starlight.example.com/auth/login`. Use client secret basic authentication.
3. Release the standard `email` and `profile` scopes. To send groups, enable the group attribute and note its name.
4. Bind the user groups who may use Starlight to the application.
5. Copy the client ID and client secret.

| Starlight field | Value |
| --- | --- |
| Client ID, client secret | From step 5 |
| Group attribute name | The name from step 3 |

Starlight uses JumpCloud's issuer `https://oauth.id.jumpcloud.com/`.

## SAML 2.0

1. Add a **Custom SAML App**. Set the SP entity ID to `https://starlight.example.com/auth/jumpcloud/saml/metadata` and the
   ACS URL to `https://starlight.example.com/auth/jumpcloud/saml/acs`.
2. Add attributes `email` and `name`, and include the group attribute if Starlight should receive groups.
3. Export the IdP metadata and paste it into Starlight, or give its URL.

## Check it works

Sign in from `https://starlight.example.com/auth/login`.
