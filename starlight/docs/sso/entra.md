# Microsoft Entra ID

Microsoft Entra ID (formerly Azure AD) over OpenID Connect or SAML 2.0. Starlight treats a connection as single-tenant: it
checks the tenant ID in every token and identifies people by tenant and object ID, never by email. The examples use
Starlight at `https://starlight.example.com` and the slug `entra`.

## OpenID Connect

1. In the Microsoft Entra admin center, open **App registrations** and create a registration for Starlight. Choose
   accounts in this organizational directory only.
2. Add a **Web** redirect URI `https://starlight.example.com/auth/entra/callback`.
3. Under **Certificates & secrets**, create a client secret and copy its value. If your tenant forbids client
   secrets, use a certificate instead, as described below.
4. Under **Token configuration**:
   - add the optional ID token claims `email` and `xms_edov`;
   - add a groups claim (security groups, or only the groups assigned to the application) emitted as group IDs.
5. Under **API permissions**, add the Microsoft Graph **application** permission `GroupMember.Read.All` and grant admin
   consent. Starlight uses it only when a user is in too many groups for the token.
6. In **Enterprise applications**, open the matching application, require assignment, and assign users or groups.

| Starlight field | Value |
| --- | --- |
| Directory (tenant) ID | The tenant GUID from the registration's overview |
| Client ID | The application (client) ID |
| Client secret | From step 3; empty with a certificate |
| Email domains | Your verified domains |

For access tokens from Entra on Starlight's API, set the API's `accessTokenAcceptedVersion` to 2 in the app manifest so
its tokens use the same v2.0 issuer as sign-in.

### Certificate instead of a client secret

With a certificate, Starlight authenticates with `private_key_jwt`: it signs each token request with a private key whose
certificate is registered on the app.

1. In Starlight, set **Client authentication** to `private_key_jwt` and tick **Generate a key and certificate**. To use your
   own pair instead, paste the PEM private key and certificate.
2. On the connection's page in the console, choose **Download certificate**.
3. In the app registration, open **Certificates & secrets**, choose the **Certificates** tab and upload the file.
4. Check that the thumbprint Entra shows matches the SHA-1 thumbprint on Starlight's connection page.

Starlight names the certificate in each assertion's `x5t` header, which Entra uses to find it. Calls to Microsoft Graph for
group overage use the same certificate, so no Graph secret is needed; if a separate Graph client ID is set, upload the
certificate to that app as well. A generated certificate lasts two years. Before it expires, choose **Replace with a new
key and certificate** and upload the new certificate straight away, because Starlight uses the new key at once.

### Email trust

Entra does not verify the `email` claim, so Starlight trusts it only when `xms_edov` is true or the email equals the
user principal name on one of the connection's email domains. That closes the account-linking hole known as nOAuth.

### Groups and overage

Entra sends group object IDs. When a user belongs to more groups than fit in a token, Entra leaves the groups claim
out and marks the token instead; Starlight then reads the user's transitive group membership from Microsoft Graph with the
connection's client credentials. Assigning only the groups that matter to the application keeps tokens small.

In Starlight, map projects and roles to the group object IDs.

## SAML 2.0

1. In **Enterprise applications**, create your own application and choose the non-gallery option.
2. Under **Single sign-on**, choose SAML. Set the identifier (entity ID) to
   `https://starlight.example.com/auth/entra/saml/metadata` and the reply URL (ACS) to
   `https://starlight.example.com/auth/entra/saml/acs`.
3. Under **Attributes & Claims**, keep the default claims and add a group claim with the group ID as its source.
4. Copy the App Federation Metadata URL into Starlight's metadata URL.
5. To sign people out of Entra when they sign out of Starlight, set the logout URL under **Basic SAML Configuration** to
   `https://starlight.example.com/auth/entra/saml/slo` and turn on **Sign out of the IdP too** in Starlight.
6. For Graph lookups on overage, create an app registration with the `GroupMember.Read.All` application permission and
   a client secret, and enter its client ID and secret in Starlight's Graph fields.

Entra always sends `objectidentifier` and `tenantid` claims; Starlight uses them as the stable subject, exactly as for
OIDC.

## Logout and provisioning

Over OIDC, Entra uses front-channel logout rather than OIDC back-channel logout, so Starlight ends sessions when people sign
out of Starlight and when provisioning deactivates them. **Sign out of the IdP too** also sends the browser to Entra's
end-session endpoint.

Over SAML, Starlight and Entra exchange single logout messages at the logout URL from step 5. Starlight accepts only signed
logout messages. If a reply from Entra fails Starlight's checks, Starlight has still ended its own session and says that it
could not confirm the Entra sign-out.

Set up [SCIM](../scim.md) from the same enterprise application.

## Check it works

Sign in from `https://starlight.example.com/auth/login`. A user in another tenant is refused with "Wrong organisation".
