package presets

import (
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
)

// Entra SAML claim URIs, as Microsoft Entra ID issues them.
const (
	EntraObjectID    = "http://schemas.microsoft.com/identity/claims/objectidentifier"
	EntraTenantID    = "http://schemas.microsoft.com/identity/claims/tenantid"
	EntraDisplayName = "http://schemas.microsoft.com/identity/claims/displayname"
	EntraGroups      = "http://schemas.microsoft.com/ws/2008/06/identity/claims/groups"
	EntraGroupsLink  = "http://schemas.microsoft.com/claims/groups.link"
	ClaimEmail       = "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress"
	ClaimName        = "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name"
	ClaimGroup       = "http://schemas.xmlsoap.org/claims/Group"
)

func init() {
	register(Preset{
		Name: "okta", Label: "Okta", Protocols: both(),
		Summary: "Okta Workforce Identity with the org or a custom authorization server.",
		Fields: join([]Field{
			{Name: "okta_domain", Label: "Okta domain", Kind: KindText, Required: true, Protocol: connections.OIDC, Placeholder: "example.okta.com"},
			{Name: "okta_server", Label: "Authorization server ID", Kind: KindText, Protocol: connections.OIDC,
				Help: "Leave empty for the org authorization server; use default for the default custom server."},
		}, clientAuth(true)),
		issuer: func(v Values) (string, error) {
			host, err := v.host("okta_domain", "Okta domain")
			if err != nil {
				return "", err
			}
			if s := v["okta_server"]; s != "" {
				return "https://" + host + "/oauth2/" + s, nil
			}
			return "https://" + host, nil
		},
		oidc: groupsClaim("groups"),
		saml: groupsClaim("groups"),
	})
	register(Preset{
		Name: "entra", Label: "Microsoft Entra ID", Protocols: both(),
		Summary: "Microsoft Entra ID (Azure AD), single tenant, with Graph lookups when a user has too many groups.",
		Fields: join([]Field{
			{Name: "entra_tenant_id", Label: "Directory (tenant) ID", Kind: KindText, Required: true, Placeholder: "00000000-0000-0000-0000-000000000000"},
			{Name: "graph_client_id", Label: "Graph client ID", Kind: KindText, Protocol: connections.SAML, Help: "App registration with GroupMember.Read.All."},
			{Name: "graph_client_secret", Label: "Graph client secret", Kind: KindSecret, Protocol: connections.SAML},
		}, clientAuth(false)),
		issuer: func(v Values) (string, error) {
			t, err := v.need("entra_tenant_id", "Tenant ID")
			if err != nil {
				return "", err
			}
			return "https://login.microsoftonline.com/" + t + "/v2.0", nil
		},
		oidc: func(in *connections.Input, v Values) {
			in.Config.Claims = connections.ClaimMap{Subject: "oid", Tenant: "tid", Email: "email", EmailVerified: "xms_edov", Name: "name",
				Username: "preferred_username", Groups: "groups"}
			in.Config.Groups.Strategy = connections.GroupsEntraGraph
			in.Config.Provider.EntraTenantID = v["entra_tenant_id"]
			in.Config.Provider.GraphClientID = v["client_id"]
		},
		saml: func(in *connections.Input, v Values) {
			in.Config.Claims = connections.ClaimMap{Subject: EntraObjectID, Tenant: EntraTenantID, Email: ClaimEmail, Name: EntraDisplayName,
				Username: ClaimName, Groups: EntraGroups}
			in.Config.Policy.TrustEmail = false
			in.Config.Provider.EntraTenantID = v["entra_tenant_id"]
			in.Config.Groups.Strategy = connections.GroupsClaim
			if v["graph_client_id"] != "" {
				in.Config.Groups.Strategy = connections.GroupsEntraGraph
				in.Config.Provider.GraphClientID = v["graph_client_id"]
				in.Secrets.GraphClientSecret = v["graph_client_secret"]
			}
		},
	})
	register(Preset{
		Name: "google", Label: "Google Workspace", Protocols: both(),
		Summary: "Google Workspace accounts on your domains, with groups from the Admin SDK Directory API.",
		Fields: []Field{
			{Name: "google_domains", Label: "Workspace domains", Kind: KindList, Required: true, Placeholder: "example.com"},
			{Name: "google_service_account", Label: "Service account key (JSON)", Kind: KindTextarea, Protocol: connections.OIDC, WriteOnly: true,
				Help: "Optional; enables group sync through domain-wide delegation."},
			{Name: "google_admin_email", Label: "Admin to act as", Kind: KindText, Protocol: connections.OIDC, Placeholder: "directory-reader@example.com"},
		},
		issuer: fixed("https://accounts.google.com"),
		oidc: func(in *connections.Input, v Values) {
			in.Config.Provider.GoogleDomains = v.List("google_domains")
			in.Config.Claims.Username = ""
			if d := in.Config.Provider.GoogleDomains; len(d) == 1 {
				in.Config.OIDC.AuthParams = map[string]string{"hd": d[0]}
			}
			if v["google_service_account"] != "" || v["google_admin_email"] != "" {
				in.Config.Groups.Strategy = connections.GroupsGoogleDirectory
				in.Config.Provider.GoogleAdminEmail = v["google_admin_email"]
				in.Secrets.GoogleServiceAccount = v["google_service_account"]
			}
		},
		saml: func(in *connections.Input, v Values) {
			in.Config.Provider.GoogleDomains = v.List("google_domains")
			in.Config.Policy.AllowedDomains = v.List("google_domains")
			if in.Config.SAML.NameIDFormat == "" {
				in.Config.SAML.NameIDFormat = "emailAddress"
			}
			in.Config.Claims.Groups = "groups"
		},
	})
	register(Preset{
		Name: "auth0", Label: "Auth0", Protocols: oidcOnly(),
		Summary: "Auth0 tenant or custom domain; groups from a custom claim set by an Action.",
		Fields: []Field{
			{Name: "auth0_domain", Label: "Auth0 domain", Kind: KindText, Required: true, Placeholder: "example.us.auth0.com"},
			{Name: "groups_claim", Label: "Groups claim", Kind: KindText, Placeholder: "https://starlight.example.com/groups"},
		},
		issuer: func(v Values) (string, error) {
			host, err := v.host("auth0_domain", "Auth0 domain")
			return "https://" + host + "/", err
		},
	})
	register(Preset{
		Name: "onelogin", Label: "OneLogin", Protocols: both(),
		Summary: "OneLogin OpenID Connect or SAML custom connector.",
		Fields: []Field{
			{Name: "onelogin_subdomain", Label: "OneLogin subdomain", Kind: KindText, Required: true, Protocol: connections.OIDC, Placeholder: "example"},
			{Name: "groups_claim", Label: "Groups claim or attribute", Kind: KindText, Placeholder: "groups"},
		},
		issuer: func(v Values) (string, error) {
			s, err := v.host("onelogin_subdomain", "OneLogin subdomain")
			return "https://" + s + ".onelogin.com/oidc/2", err
		},
	})
	register(Preset{
		Name: "pingone", Label: "PingOne", Protocols: both(),
		Summary: "PingOne for Enterprise environments in any region.",
		Fields: []Field{
			{Name: "pingone_region", Label: "Region domain", Kind: KindSelect, Protocol: connections.OIDC, Required: true,
				Options: []string{"auth.pingone.com", "auth.pingone.eu", "auth.pingone.ca", "auth.pingone.asia", "auth.pingone.com.au"}},
			{Name: "pingone_environment", Label: "Environment ID", Kind: KindText, Required: true, Protocol: connections.OIDC},
			{Name: "groups_claim", Label: "Groups claim or attribute", Kind: KindText, Placeholder: "groups"},
		},
		issuer: func(v Values) (string, error) {
			region, err := v.host("pingone_region", "Region domain")
			if err != nil {
				return "", err
			}
			env, err := v.need("pingone_environment", "Environment ID")
			return "https://" + region + "/" + env + "/as", err
		},
	})
	register(Preset{
		Name: "jumpcloud", Label: "JumpCloud", Protocols: both(),
		Summary: "JumpCloud custom OIDC or SAML application.",
		Fields:  []Field{{Name: "groups_claim", Label: "Group attribute name", Kind: KindText, Help: "The name set for the group attribute in JumpCloud."}},
		issuer:  fixed("https://oauth.id.jumpcloud.com/"),
	})
	register(Preset{
		Name: "duo", Label: "Duo Single Sign-On", Protocols: both(),
		Summary: "Duo SSO as a generic OIDC relying party or SAML service provider.",
		Fields: []Field{
			{Name: "issuer", Label: "Issuer", Kind: KindURL, Required: true, Protocol: connections.OIDC, Help: "Copy the issuer Duo shows for the application."},
			{Name: "groups_claim", Label: "Groups claim or attribute", Kind: KindText},
		},
	})
	register(Preset{
		Name: "gitlab", Label: "GitLab", Protocols: oidcOnly(),
		Summary: "GitLab.com or self-managed GitLab; access restricted to your groups.",
		Fields: []Field{
			{Name: "gitlab_url", Label: "GitLab URL", Kind: KindURL, Placeholder: "https://gitlab.com", Help: "Your self-managed URL, or gitlab.com."},
		},
		issuer: func(v Values) (string, error) {
			if v["gitlab_url"] == "" {
				return "https://gitlab.com", nil
			}
			return v.base("gitlab_url", "GitLab URL")
		},
		oidc: func(in *connections.Input, _ Values) {
			in.Config.Claims.Username = "nickname"
			in.Config.Claims.Groups = "groups_direct"
		},
	})
	register(Preset{
		Name: "aws_identity_center", Label: "AWS IAM Identity Center", Protocols: samlOnly(),
		Summary: "IAM Identity Center customer-managed SAML 2.0 application.",
		saml: func(in *connections.Input, _ Values) {
			if in.Config.SAML.NameIDFormat == "" {
				in.Config.SAML.NameIDFormat = "persistent"
			}
			in.Config.Claims.Groups = "groups"
		},
	})
	register(Preset{
		Name: "zitadel", Label: "ZITADEL", Protocols: oidcOnly(),
		Summary: "ZITADEL Cloud or self-hosted; project roles become groups.",
		Fields:  []Field{{Name: "issuer", Label: "Instance URL", Kind: KindURL, Required: true, Placeholder: "https://example-abc123.zitadel.cloud"}},
		issuer: func(v Values) (string, error) {
			return v.base("issuer", "Instance URL")
		},
		oidc: func(in *connections.Input, _ Values) {
			in.Config.Claims.Groups = "urn:zitadel:iam:org:project:roles"
			in.Config.Claims.GroupsAreKeys = true
		},
	})
}
