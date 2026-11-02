# GitLab

GitLab.com or self-managed GitLab over OpenID Connect. GitLab.com is open to anyone, so a GitLab.com connection must
require membership of at least one of your groups. The examples use Starlight at `https://starlight.example.com` and the slug
`gitlab`.

## Set up the application

1. Create an OAuth application: on GitLab.com under a group you own, or on self-managed GitLab instance-wide in the
   Admin Area or under a group.
2. Set the redirect URI to `https://starlight.example.com/auth/gitlab/callback`, keep it confidential, and select the
   `openid`, `profile` and `email` scopes.
3. Copy the application ID and secret.

| Starlight field | Value |
| --- | --- |
| GitLab URL | Empty for `https://gitlab.com`, or your instance URL |
| Client ID, client secret | From step 3 |
| Required groups | Group paths such as `example-org` or `example-org/platform` |

## Groups

GitLab's ID token lists the user's direct groups by path in `groups_direct`; Starlight syncs them and enforces the
required groups.

## Check it works

Sign in with an account in a required group; an account outside them is refused with "Access not granted".
