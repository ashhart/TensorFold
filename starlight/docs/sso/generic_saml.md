# Other SAML 2.0 providers

Any SAML 2.0 identity provider that supports Web Browser SSO. The examples use Starlight at `https://starlight.example.com` and
the slug `corp`.

## Requirements

- IdP metadata with an HTTP-Redirect single sign-on endpoint and a signing certificate.
- Responses over the HTTP-POST binding, with a signed response or signed assertions.
- An audience restriction naming Starlight's entity ID and a bearer subject confirmation for Starlight's ACS URL.
- For single logout, a SingleLogoutService endpoint in the metadata with the HTTP-Redirect or HTTP-POST binding, and
  signed logout messages.

Register Starlight at the IdP with the entity ID `https://starlight.example.com/auth/corp/saml/metadata`, the ACS URL
`https://starlight.example.com/auth/corp/saml/acs` and the single logout URL `https://starlight.example.com/auth/corp/saml/slo`,
or import the SP metadata from the entity ID URL. The metadata also carries the certificate Starlight signs logout messages
with.

## Starlight fields

| Field | Use |
| --- | --- |
| IdP metadata URL or XML | Exactly one; a URL is refreshed at most daily, and a recent cached copy covers outages |
| NameID format | `persistent` is best; `emailAddress` and `unspecified` work; `transient` needs a subject attribute |
| Stable subject attribute | Use an attribute instead of the NameID when the NameID can change |
| Email, name, groups attributes | Attribute names as the IdP releases them; friendly names also match |
| Allow IdP-initiated sign-in | Off by default; when on, each assertion is still accepted only once |
| Sign out of the IdP too | Sends a signed LogoutRequest at sign-out when the IdP metadata lists a single logout endpoint |

## Check it works

Sign in from `https://starlight.example.com/auth/login`. Replaying the same response is refused. With single logout on,
signing out of Starlight also ends the IdP session and returns to the signed-out page; a LogoutRequest from the IdP ends
that person's Starlight sessions.
