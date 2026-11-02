# Okta

Okta Workforce Identity over OpenID Connect (recommended) or SAML 2.0. The examples use Starlight at
`https://starlight.example.com` and the connection slug `okta`.

## OpenID Connect

1. In the Okta Admin Console, open **Applications**, choose **Create App Integration**, select **OIDC - OpenID Connect**
   and **Web Application**.
2. Set the sign-in redirect URI to `https://starlight.example.com/auth/okta/callback`. If you turn on RP-initiated logout in
   Starlight, add `https://starlight.example.com/auth/login?signed_out=1` as a sign-out redirect URI.
3. Keep the Authorization Code grant. Use a client secret, or **Public key / Private key** client authentication if
   Starlight should authenticate with `private_key_jwt`, as described below.
4. Assign the people or groups who may use Starlight.
5. Copy the client ID and client secret.

| Starlight field | Value |
| --- | --- |
| Okta domain | Your Okta domain, for example `example.okta.com` |
| Authorization server ID | Empty for the org authorization server, or the ID of a custom server such as `default` |
| Client ID, client secret | From step 5 |
| Email domains | The domains your people sign in with |

### Public key instead of a client secret

1. In Starlight, set **Client authentication** to `private_key_jwt` and tick **Generate a key and certificate**.
2. Copy the public key that the connection's page in the console shows as a JWK.
3. In Okta, set the app's client authentication to **Public key / Private key** and add that key.

Okta finds the key by its `kid`, which the JWK carries. If Okta generated the key pair instead, paste its private key
into Starlight and enter Okta's key ID in **Key ID**.

### Groups

Okta adds groups to tokens only when asked:

- **Org authorization server:** in the app's **Sign On** tab, under the OpenID Connect ID token settings, set a groups
  claim named `groups` with a filter, for example a regex that matches the groups Starlight should see.
- **Custom authorization server:** under **Security → API → Authorization Servers**, open the server, add a claim
  named `groups` to the ID token with the Groups value type and a filter.

Starlight syncs the claim at every sign-in. Use [SCIM](../scim.md) to keep groups and deactivations current between
sign-ins.

## SAML 2.0

1. Choose **Create App Integration** and **SAML 2.0**.
2. Set the single sign-on URL to `https://starlight.example.com/auth/okta/saml/acs` and use it for the recipient and
   destination. Set the audience URI to `https://starlight.example.com/auth/okta/saml/metadata`.
3. Choose a stable Name ID format, such as Persistent, or keep Unspecified with a username that never changes.
4. Add an attribute statement `email` for the user's email and `name` for the display name, and a group attribute
   statement `groups` with a filter.
5. Copy the IdP metadata URL from the app's **Sign On** tab into Starlight's metadata URL.
6. For single logout, open **Show Advanced Settings**, tick **Enable Single Logout**, and set the single logout URL to
   `https://starlight.example.com/auth/okta/saml/slo` and the SP issuer to
   `https://starlight.example.com/auth/okta/saml/metadata`. Upload the signing certificate shown on the connection's page,
   then turn on **Sign out of the IdP too** in Starlight.

## Check it works

Open `https://starlight.example.com/auth/login`, choose the Okta button, and confirm that the console shows your name.
The audit log records `auth.login` with the connection slug and how many groups arrived.
