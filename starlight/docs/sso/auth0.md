# Auth0

An Auth0 tenant, or a custom domain in front of it, over OpenID Connect. The examples use Starlight at
`https://starlight.example.com` and the slug `auth0`.

## Set up the application

1. In the Auth0 Dashboard, create an application of type **Regular Web Application**.
2. In its settings, set the allowed callback URL to `https://starlight.example.com/auth/auth0/callback`. If you turn on
   RP-initiated logout in Starlight, allow the logout URL `https://starlight.example.com/auth/login?signed_out=1`.
3. Copy the domain, client ID and client secret.

| Starlight field | Value |
| --- | --- |
| Auth0 domain | The tenant domain, for example `example.us.auth0.com`, or your custom domain |
| Client ID, client secret | From step 3 |
| Groups claim | The custom claim your Action sets, if any |

Starlight derives the issuer `https://<domain>/` with the trailing slash Auth0 uses.

## Groups

Auth0 does not put groups or roles in ID tokens by default. Add a post-login Action that sets a custom claim, for
example from the user's roles:

```js
exports.onExecutePostLogin = async (event, api) => {
  api.idToken.setCustomClaim("https://starlight.example.com/groups", event.authorization?.roles ?? []);
};
```

Enter the same claim name in Starlight's groups claim field. Auth0 recommends namespaced claim names such as this URL.

## Logout

If your Auth0 tenant supports OpenID Connect back-channel logout, set the application's back-channel logout URI to
`https://starlight.example.com/auth/auth0/backchannel-logout`.

## Check it works

Sign in from `https://starlight.example.com/auth/login`. Auth0 sends `email_verified`; unverified emails cannot create or
link Starlight accounts.
