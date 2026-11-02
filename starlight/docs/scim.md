# SCIM provisioning

Starlight serves SCIM 2.0 ([RFC 7643](https://www.rfc-editor.org/rfc/rfc7643), [RFC 7644](https://www.rfc-editor.org/rfc/rfc7644))
so an identity provider can create, update, suspend and remove Starlight users and groups as people join, move and leave.
SSO decides who can sign in; SCIM decides who exists and keeps groups current between sign-ins.

## Endpoint and tokens

The base URL is Starlight's public URL followed by `/scim/v2`, for example `https://starlight.example.com/scim/v2`.

Every SCIM token belongs to one SSO connection. Issue it from that connection's provisioning settings. The token
(`sls_...`) is shown once; Starlight stores only its SHA-256. A token can carry an expiry and is revoked from the same place.
Revocation takes effect on the next request, and disabling the connection stops all of its tokens.

A token sees and changes only its own connection's resources:

- users it provisioned, and users who signed in through the same connection before provisioning was switched on;
- groups it provisioned.

Another connection's users and groups answer 404, and a group cannot take another connection's user as a member.

## Okta

1. In the Starlight app integration's **General** tab, enable SCIM provisioning.
2. In **Provisioning → Integration**, set the SCIM connector base URL to the endpoint above, the unique identifier
   field for users to `userName`, and the authentication mode to HTTP Header with the Starlight token as the bearer token.
   Choose Push New Users, Push Profile Updates and Push Groups, then **Test Connector Configuration**.
3. In **Provisioning → To App**, enable Create Users, Update User Attributes and Deactivate Users.
4. Assign people and groups to the app, and push groups from the **Push Groups** tab.

Okta's deactivation (`active: false`) suspends the Starlight user. Removing a person from the app follows the same path.

## Microsoft Entra ID

1. Open the Starlight enterprise application, select **Provisioning**, and set the mode to Automatic.
2. Under Admin Credentials, set the Tenant URL to the endpoint above and the Secret Token to the Starlight token, then
   select **Test Connection**.
3. Keep the default user and group mappings: `userPrincipalName` to `userName`, `mail` to
   `emails[type eq "work"].value`, `objectId` to the group's `externalId`, and `members`.
4. Scope provisioning to assigned users and groups, turn provisioning on and save. Use **Provision on demand** to
   check one user immediately; Entra otherwise works in periodic cycles.

Starlight accepts both Entra's legacy PATCH shapes (`"Replace"`, `"False"` strings, path-less objects with dotted keys)
and the compliant ones, so the `aadOptscim062020` tenant URL flag is optional.

## Attributes

| SCIM attribute | Starlight |
| --- | --- |
| `userName` | Unique per connection, case-insensitive; the sign-in name the IdP matches on |
| `emails` | Stored as sent; the primary (else work, else first) email becomes the account email, unique among live users |
| `displayName`, `name` | Stored as sent; the account's display name is `displayName`, else `name.formatted`, else given and family names |
| `externalId` | Stored as sent, case-exact |
| `active` | `true` is an active account, `false` a suspended one |
| Enterprise extension | `employeeNumber`, `costCenter`, `organization`, `division`, `department`, `manager` |
| Other attributes | Stored and returned as sent, up to 64 KiB per user |
| `password` | Never stored or returned |
| `groups` | Read-only; lists the connection's SCIM groups the user belongs to |

Groups have `displayName` (unique per connection, case-insensitive), `externalId` and `members`. Members are users;
nested groups are refused with `invalidValue`. Starlight role bindings and project membership can name a SCIM group.

## What deactivation does

Setting `active` to `false` suspends the user and, in the same transaction, revokes every console session, refresh and
access token, API key and agent task the user owns, pending device sign-ins and unused authorisation codes. Setting
`active` back to `true` lets the person sign in again; revoked keys and tasks stay revoked.

`DELETE` deactivates the user permanently: the same revocations, plus removal from the connection's groups and from the
connection's identity links. The resource then answers 404, and a later `POST` with the same `userName` creates a new
user. Deleting a group also deletes the role bindings that named it.

## The last owner

Starlight will not suspend or delete its last active owner over SCIM, because that would lock everyone out of the console
until someone used a break-glass link. A `PATCH` or `PUT` with `active` set to `false`, a `POST` that would adopt such
a person, who signed in before provisioning, as suspended, and a `DELETE` then answer 409 with an error whose `detail`
asks the administrator to make someone else an owner first. Nothing changes, and the refused attempt is audited. Owners count whether they hold the role
directly or through a group, so suspending the last member of an owner group is refused too.

Group changes are never refused. Removing a person from a group in the IdP removes the roles that the group grants, even
from the last owner, after which Starlight has no active owner left to protect. Keep at least one owner who holds the role
directly rather than only through a group: group sync cannot take it away, and `starlight admin recovery` issues
break-glass links only to direct owners.

## Protocol support

| Feature | Support |
| --- | --- |
| Resources | `/Users`, `/Groups`, `/ServiceProviderConfig`, `/ResourceTypes`, `/Schemas` |
| Methods | `GET`, `POST`, `PUT`, `PATCH`, `DELETE` |
| Filters | `eq ne co sw ew gt ge lt le pr`, `and`, `or`, `not`, parentheses and value filters such as `emails[type eq "work"]` |
| Filterable user attributes | `id`, `userName`, `externalId`, `displayName`, `active`, `emails`, `phoneNumbers`, `addresses`, `name.*`, `title`, `userType`, `nickName`, `preferredLanguage`, `locale`, `meta.created`, `meta.lastModified`, enterprise `department`, `employeeNumber`, `costCenter`, `organization`, `division`, `manager.value` |
| Filterable group attributes | `id`, `displayName`, `externalId`, `members`, `meta.created`, `meta.lastModified` |
| Pagination | `startIndex` (1-based) and `count` (default 100, at most 500) |
| Projection | `attributes` and `excludedAttributes`, including `excludedAttributes=members` |
| Versions | Weak ETags in `meta.version` and `ETag`; `If-Match` on `PUT`, `PATCH` and `DELETE` |
| Not supported | Bulk, sorting, `/Me`, `.search`, password changes, nested groups |

A group `PATCH` answers 204 unless the request names `attributes` or `excludedAttributes`; a user `PATCH` returns the
user.

## Limits and errors

- 25 requests a second per token with a burst of 200, counted across all Starlight instances together; beyond that, 429
  with `Retry-After`.
- Request bodies up to 4 MiB, 1,000 PATCH operations, 20,000 members per group, 100 values per multi-valued attribute.
- Filters up to 4,096 characters, 64 terms and 16 levels of nesting.

Errors use the RFC 7644 envelope with `status` and, where it applies, `scimType`: `uniqueness` (409) for a taken
`userName`, email or group name; `invalidFilter`, `invalidPath`, `invalidValue`, `invalidSyntax`, `noTarget` and
`mutability` (400); 409 without a `scimType` for removing the last owner; 412 for a stale `If-Match`; 401 for a
missing, revoked or expired token.

## Audit

Every change is in Starlight's audit chain with the SCIM token as the actor: `scim.user.create`, `scim.user.adopt`,
`scim.user.update`, `scim.user.suspend`, `scim.user.reactivate`, `scim.user.delete`, `scim.group.create`,
`scim.group.update`, `scim.group.members` (with the ids added and removed) and `scim.group.delete`. A refused attempt
to remove the last owner is recorded as `scim.user.suspend` or `scim.user.delete` with the outcome `denied` and the
reason `last_owner`.
