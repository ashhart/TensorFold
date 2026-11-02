# Google Workspace

Google Workspace over OpenID Connect (recommended) or SAML 2.0. Starlight accepts only accounts on the Workspace domains
you list: it checks the `hd` claim and `email_verified` on every sign-in, because a Google OIDC client otherwise
accepts any Google account. The examples use Starlight at `https://starlight.example.com` and the slug `google`.

## OpenID Connect

1. In the Google Cloud console, pick a project owned by your Workspace organisation and configure the OAuth consent
   screen with the **Internal** audience.
2. Create an OAuth client ID of type **Web application** with the authorized redirect URI
   `https://starlight.example.com/auth/google/callback`.
3. Copy the client ID and client secret.

| Starlight field | Value |
| --- | --- |
| Workspace domains | Every domain your people sign in with, for example `example.com` |
| Client ID, client secret | From step 3 |
| Email domains | Usually the same domains, for email routing on the sign-in page |

### Groups

Google ID tokens carry no groups. To sync them, give Starlight a service account with domain-wide delegation:

1. In the same project, enable the **Admin SDK API**, create a service account and create a JSON key for it.
2. In the Google Admin console, open **Security → Access and data control → API controls → Domain-wide delegation**
   and authorise the service account's client ID for the single scope
   `https://www.googleapis.com/auth/admin.directory.group.readonly`.
3. In Starlight, paste the JSON key and name an admin account the service account acts as; that account needs permission
   to read groups.

Starlight lists the groups each person belongs to at sign-in, identified by group email address, and can follow
group-in-group membership a few levels deep. The key is sealed at rest and used only with the read-only groups scope.

## SAML 2.0

1. In the Admin console, open **Apps → Web and mobile apps**, add a custom SAML app, and download the IdP metadata.
2. Set the ACS URL to `https://starlight.example.com/auth/google/saml/acs`, the entity ID to
   `https://starlight.example.com/auth/google/saml/metadata`, and the Name ID format to EMAIL with the primary email.
3. Map the primary email to an app attribute named `email`. To send groups, add group membership mapped to an
   attribute named `groups` and choose the groups to include.
4. Turn the app on for the organisational units or groups that should use Starlight.
5. Paste the downloaded metadata XML into Starlight and list your Workspace domains.

## Provisioning

Google Workspace provisions users only to applications from its own catalogue. Leave **Create users on first
sign-in** on, and Starlight creates accounts for verified Workspace users the first time they sign in.

## Check it works

Sign in with a Workspace account; a personal Google account or another domain is refused with "Wrong organisation".
