# ZITADEL

ZITADEL Cloud or a self-hosted instance over OpenID Connect. Project roles become Starlight groups. The examples use Starlight
at `https://starlight.example.com`, the instance `https://example-abc123.zitadel.cloud` and the slug `zitadel`.

## Set up the application

1. In a project, create a **Web** application with an authentication method that uses a client secret.
2. Add the redirect URI `https://starlight.example.com/auth/zitadel/callback`.
3. In the application's token settings, include user roles and user info inside the ID token.
4. In the project settings, assert roles on authentication, and grant roles to the people who may use Starlight.
5. Copy the client ID and client secret.

| Starlight field | Value |
| --- | --- |
| Instance URL | `https://example-abc123.zitadel.cloud` or your own domain |
| Client ID, client secret | From step 5 |

## Groups

ZITADEL sends project roles in the `urn:zitadel:iam:org:project:roles` claim as an object keyed by role. Starlight reads
the keys as group names, so a role `starlight-admin` maps like a group of that name.

## Check it works

Sign in from `https://starlight.example.com/auth/login` and confirm your roles appear as groups on your profile.
