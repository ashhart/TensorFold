package presets

import (
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
)

// eduPerson and X.500 attribute names that Shibboleth and other SAML IdPs release.
const (
	OIDMail        = "urn:oid:0.9.2342.19200300.100.1.3"
	OIDDisplayName = "urn:oid:2.16.840.1.113730.3.1.241"
	OIDPrincipal   = "urn:oid:1.3.6.1.4.1.5923.1.1.1.6"
	OIDIsMemberOf  = "urn:oid:1.3.6.1.4.1.5923.1.5.1.1"
)

func init() {
	register(Preset{
		Name: "keycloak", Label: "Keycloak", Protocols: both(),
		Summary: "Keycloak or Red Hat build of Keycloak realm.",
		Fields: []Field{
			{Name: "keycloak_url", Label: "Keycloak URL", Kind: KindURL, Required: true, Placeholder: "https://sso.example.com"},
			{Name: "keycloak_realm", Label: "Realm", Kind: KindText, Required: true, Placeholder: "example"},
		},
		issuer: func(v Values) (string, error) {
			base, err := v.base("keycloak_url", "Keycloak URL")
			if err != nil {
				return "", err
			}
			realm, err := v.need("keycloak_realm", "Realm")
			return base + "/realms/" + realm, err
		},
		oidc: groupsClaim("groups"),
		saml: func(in *connections.Input, v Values) {
			in.Config.Claims.Groups = "groups"
			if in.Config.SAML.MetadataURL == "" && in.Config.SAML.MetadataXML == "" && v["keycloak_url"] != "" && v["keycloak_realm"] != "" {
				if base, err := v.base("keycloak_url", "Keycloak URL"); err == nil {
					in.Config.SAML.MetadataURL = base + "/realms/" + v["keycloak_realm"] + "/protocol/saml/descriptor"
				}
			}
		},
	})
	register(Preset{
		Name: "authentik", Label: "authentik", Protocols: both(),
		Summary: "authentik OAuth2/OpenID or SAML provider.",
		Fields: []Field{
			{Name: "authentik_url", Label: "authentik URL", Kind: KindURL, Required: true, Protocol: connections.OIDC, Placeholder: "https://auth.example.com"},
			{Name: "authentik_slug", Label: "Application slug", Kind: KindText, Required: true, Protocol: connections.OIDC, Placeholder: "starlight"},
		},
		issuer: func(v Values) (string, error) {
			base, err := v.base("authentik_url", "authentik URL")
			if err != nil {
				return "", err
			}
			slug, err := v.need("authentik_slug", "Application slug")
			return base + "/application/o/" + slug + "/", err
		},
		oidc: groupsClaim("groups"),
		saml: groupsClaim("http://schemas.xmlsoap.org/claims/Group"),
	})
	register(Preset{
		Name: "pingfederate", Label: "PingFederate", Protocols: both(),
		Summary: "Self-managed PingFederate OAuth client or SP connection.",
		Fields: []Field{
			{Name: "issuer", Label: "Issuer (base URL)", Kind: KindURL, Required: true, Protocol: connections.OIDC, Placeholder: "https://pf.example.com"},
			{Name: "groups_claim", Label: "Groups claim or attribute", Kind: KindText},
			{Name: "trust_email", Label: "Trust email without email_verified", Kind: KindBool, Protocol: connections.OIDC},
		},
	})
	register(Preset{
		Name: "adfs", Label: "AD FS", Protocols: both(),
		Summary: "Active Directory Federation Services relying party trust or application group.",
		Fields: []Field{
			{Name: "adfs_host", Label: "AD FS host", Kind: KindText, Required: true, Protocol: connections.OIDC, Placeholder: "fs.example.com"},
		},
		issuer: func(v Values) (string, error) {
			host, err := v.host("adfs_host", "AD FS host")
			return "https://" + host + "/adfs", err
		},
		oidc: func(in *connections.Input, _ Values) {
			in.Config.Claims = connections.ClaimMap{Email: "email", Name: "unique_name", Username: "upn", Groups: "group"}
			in.Config.Policy.TrustEmail = true
		},
		saml: func(in *connections.Input, _ Values) {
			in.Config.Claims = connections.ClaimMap{Email: ClaimEmail, Name: ClaimName, Groups: ClaimGroup}
			if in.Config.SAML.NameIDFormat == "" {
				in.Config.SAML.NameIDFormat = "persistent"
			}
		},
	})
	register(Preset{
		Name: "shibboleth", Label: "Shibboleth IdP", Protocols: samlOnly(),
		Summary: "Shibboleth Identity Provider releasing eduPerson attributes.",
		saml: func(in *connections.Input, _ Values) {
			in.Config.Claims = connections.ClaimMap{Email: OIDMail, Name: OIDDisplayName, Username: OIDPrincipal, Groups: OIDIsMemberOf}
			if in.Config.SAML.NameIDFormat == "" {
				in.Config.SAML.NameIDFormat = "persistent"
			}
		},
	})
	register(Preset{
		Name: "generic_oidc", Label: "Other OpenID Connect provider", Protocols: oidcOnly(),
		Summary: "Any standards-compliant OpenID Connect provider with discovery.",
		Fields: join([]Field{{Name: "issuer", Label: "Issuer", Kind: KindURL, Required: true}}, clientAuth(true), []Field{
			{Name: "scopes", Label: "Scopes", Kind: KindList, Placeholder: "openid, email, profile"},
			{Name: "subject_claim", Label: "Stable subject claim", Kind: KindText, Placeholder: "sub"},
			{Name: "email_claim", Label: "Email claim", Kind: KindText, Placeholder: "email"},
			{Name: "name_claim", Label: "Name claim", Kind: KindText, Placeholder: "name"},
			{Name: "groups_claim", Label: "Groups claim", Kind: KindText, Placeholder: "groups"},
			{Name: "use_userinfo", Label: "Read the userinfo endpoint", Kind: KindBool},
			{Name: "trust_email", Label: "Trust email without email_verified", Kind: KindBool,
				Help: "Only for IdPs that never release unverified emails."},
		}),
	})
	register(Preset{
		Name: "generic_saml", Label: "Other SAML 2.0 provider", Protocols: samlOnly(),
		Summary: "Any SAML 2.0 identity provider with metadata.",
		Fields: []Field{
			{Name: "name_id_format", Label: "NameID format", Kind: KindSelect, Options: []string{"persistent", "emailAddress", "unspecified", "transient"}},
			{Name: "subject_claim", Label: "Stable subject attribute", Kind: KindText, Help: "Leave empty to use the NameID."},
			{Name: "email_claim", Label: "Email attribute", Kind: KindText, Placeholder: "email"},
			{Name: "name_claim", Label: "Name attribute", Kind: KindText, Placeholder: "name"},
			{Name: "groups_claim", Label: "Groups attribute", Kind: KindText, Placeholder: "groups"},
		},
	})
}
