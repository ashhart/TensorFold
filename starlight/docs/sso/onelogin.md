# OneLogin

OneLogin over OpenID Connect or SAML 2.0. The examples use Starlight at `https://starlight.example.com` and the slug
`onelogin`.

## OpenID Connect

1. In the OneLogin admin portal, add an application from the **OpenId Connect (OIDC)** connector.
2. In its configuration, add the redirect URI `https://starlight.example.com/auth/onelogin/callback`.
3. In the SSO tab, copy the client ID and client secret, and choose basic or POST token endpoint authentication.
4. Assign the application to the users or roles who may use Starlight.

| Starlight field | Value |
| --- | --- |
| OneLogin subdomain | The part before `.onelogin.com`, for example `example` |
| Client ID, client secret | From step 3 |
| Groups claim | The claim name you configure for group membership |

Starlight derives the issuer `https://<subdomain>.onelogin.com/oidc/2`.

### Groups

Add a parameter for group or role membership in the application's parameters and include it in the token, then enter
its name in Starlight's groups claim field. If OneLogin requires a scope to release it, add that scope in Starlight.

## SAML 2.0

1. Add an application from the **SAML Custom Connector (Advanced)**.
2. Set the audience (entity ID) to `https://starlight.example.com/auth/onelogin/saml/metadata` and the recipient and ACS
   URL to `https://starlight.example.com/auth/onelogin/saml/acs`.
3. Add parameters for `email`, `name` and, if needed, `groups`.
4. Copy the issuer (metadata) URL from the SSO tab into Starlight's metadata URL.
5. For single logout, set the single logout URL to `https://starlight.example.com/auth/onelogin/saml/slo`, have OneLogin
   sign its SLO responses, and turn on **Sign out of the IdP too** in Starlight.

## Check it works

Sign in from `https://starlight.example.com/auth/login` and confirm the console shows your account.
