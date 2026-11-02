# AD FS

Active Directory Federation Services over SAML 2.0 (recommended) or OpenID Connect. The examples use Starlight at
`https://starlight.example.com`, AD FS at `https://fs.example.com` and the slug `adfs`.

## SAML 2.0

1. In AD FS Management, add a claims-aware relying party trust and import its data from
   `https://starlight.example.com/auth/adfs/saml/metadata`.
2. Choose an access control policy, for example permitting only a Starlight users group.
3. In the claim issuance policy, add a **Send LDAP Attributes as Claims** rule from Active Directory:
   - E-Mail-Addresses to E-Mail Address;
   - Display-Name to Name;
   - Token-Groups - Unqualified Names to Group.
4. Add a rule that issues a **Name ID** in the persistent format from an attribute that never changes for a person.
5. Give Starlight the federation metadata URL `https://fs.example.com/FederationMetadata/2007-06/FederationMetadata.xml`.

Starlight's preset reads the standard claim types for email, name and group. Because Starlight publishes an encryption
certificate, AD FS can encrypt assertions to it.

The imported metadata also gives AD FS Starlight's SAML logout endpoints and the certificate Starlight signs logout messages
with. Turn on **Sign out of the IdP too** in Starlight to end the AD FS session when people sign out of Starlight; a sign-out
that AD FS starts ends the person's Starlight sessions too.

## OpenID Connect

AD FS can also act as an OpenID Connect provider through an application group with a server application, a shared
secret and the redirect URI `https://starlight.example.com/auth/adfs/callback`. Its issuer is `https://fs.example.com/adfs`.
The ID token must carry `email` and, for groups, a `group` claim through issuance transform rules.

| Starlight field | Value |
| --- | --- |
| AD FS host | `fs.example.com` (OIDC) |
| IdP metadata URL | The federation metadata URL (SAML) |

## Check it works

Sign in from `https://starlight.example.com/auth/login` with a domain account.
