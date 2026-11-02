package identitypages

import (
	"sort"
	"strings"

	"github.com/ashhart/TensorFold/starlight/internal/console"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
)

// editForm is a connection's stored settings as form values; secrets are never filled in.
type editForm struct {
	Name, Slug, Domains, RequiredGroups, AllowedDomains                                         string
	JIT, TrustEmail                                                                             bool
	SubjectClaim, EmailClaim, EmailVerifiedClaim, NameClaim, UsernameClaim, GroupsClaim, Tenant string
	GroupsStrategy                                                                              string
	NestedDepth                                                                                 int
	Issuer, ClientID, AuthMethod, Scopes, Prompt, SigningKeyID, AuthParams, BearerAudiences     string
	SigningCertificate                                                                          string
	UseUserinfo, RPLogout, AcceptBearer                                                         bool
	MetadataURL, MetadataXML, NameIDFormat                                                      string
	AllowIDPInitiated, SignRequests                                                             bool
	EntraTenantID, GraphClientID, GoogleDomains, GoogleAdminEmail                               string
}

var (
	strategies   = []string{connections.GroupsNone, connections.GroupsClaim, connections.GroupsEntraGraph, connections.GroupsGoogleDirectory, connections.GroupsSCIM}
	authMethods  = []string{connections.AuthBasic, connections.AuthPost, connections.AuthPrivateJWT}
	nameIDFormat = []string{"", "persistent", "emailAddress", "unspecified", "transient"}
)

func lines(list []string) string { return strings.Join(list, "\n") }

func formFrom(cn connections.Connection) editForm {
	cfg := cn.Config
	f := editForm{Name: cn.Name, Slug: cn.Slug, Domains: lines(cn.Domains), JIT: cn.JIT, RequiredGroups: lines(cfg.Policy.RequiredGroups),
		AllowedDomains: lines(cfg.Policy.AllowedDomains), TrustEmail: cfg.Policy.TrustEmail, SubjectClaim: cfg.Claims.Subject,
		EmailClaim: cfg.Claims.Email, EmailVerifiedClaim: cfg.Claims.EmailVerified, NameClaim: cfg.Claims.Name, UsernameClaim: cfg.Claims.Username,
		GroupsClaim: cfg.Claims.Groups, Tenant: cfg.Claims.Tenant, GroupsStrategy: cfg.Groups.Strategy, NestedDepth: cfg.Groups.NestedDepth,
		EntraTenantID: cfg.Provider.EntraTenantID, GraphClientID: cfg.Provider.GraphClientID, GoogleDomains: lines(cfg.Provider.GoogleDomains),
		GoogleAdminEmail: cfg.Provider.GoogleAdminEmail}
	if o := cfg.OIDC; o != nil {
		f.Issuer, f.ClientID, f.AuthMethod, f.Scopes, f.Prompt, f.SigningKeyID = o.Issuer, o.ClientID, o.AuthMethod, lines(o.Scopes), o.Prompt, o.SigningKeyID
		f.SigningCertificate = o.SigningCertificate
		f.UseUserinfo, f.RPLogout, f.AcceptBearer, f.BearerAudiences = o.UseUserinfo, o.RPLogout, o.AcceptBearer, lines(o.BearerAudiences)
		keys := make([]string, 0, len(o.AuthParams))
		for k := range o.AuthParams {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		var params []string
		for _, k := range keys {
			params = append(params, k+"="+o.AuthParams[k])
		}
		f.AuthParams = lines(params)
	}
	if s := cfg.SAML; s != nil {
		f.MetadataURL, f.MetadataXML, f.NameIDFormat, f.AllowIDPInitiated, f.SignRequests = s.MetadataURL, s.MetadataXML, s.NameIDFormat,
			s.AllowIDPInitiated, s.SignRequests
		f.RPLogout = s.RPLogout
	}
	return f
}

// list splits a textarea on newlines and commas.
func list(q *console.Request, k string) []string {
	var out []string
	for _, part := range strings.FieldsFunc(q.R.PostFormValue(k), func(r rune) bool { return r == ',' || r == '\n' || r == '\r' }) {
		if s := strings.TrimSpace(part); s != "" {
			out = append(out, s)
		}
	}
	return out
}

// inputFrom applies the edit form to the stored connection; empty secret fields keep their sealed values.
func inputFrom(cn connections.Connection, q *console.Request) (connections.Input, error) {
	cfg := cn.Config
	in := connections.Input{Slug: q.Form("slug"), Name: q.Form("name"), Protocol: cn.Protocol, Preset: cn.Preset, Domains: list(q, "domains"),
		Enabled: cn.Enabled, JIT: q.Bool("jit"), Config: cfg}
	c := &in.Config
	c.Policy = connections.Policy{RequiredGroups: list(q, "required_groups"), AllowedDomains: list(q, "allowed_domains"), TrustEmail: q.Bool("trust_email")}
	c.Claims = connections.ClaimMap{Subject: q.Form("subject_claim"), Email: q.Form("email_claim"), EmailVerified: q.Form("email_verified_claim"),
		Name: q.Form("name_claim"), Username: q.Form("username_claim"), Groups: q.Form("groups_claim"), Tenant: q.Form("tenant_claim"),
		GroupsAreKeys: cfg.Claims.GroupsAreKeys}
	depth, err := q.Int("nested_depth", 0)
	if err != nil {
		return in, console.Invalid("Nested group depth must be a whole number.")
	}
	c.Groups = connections.Groups{Strategy: q.Form("groups_strategy"), NestedDepth: depth}
	c.Provider = connections.Provider{EntraTenantID: q.Form("entra_tenant_id"), GraphClientID: q.Form("graph_client_id"),
		GoogleDomains: list(q, "google_domains"), GoogleAdminEmail: q.Form("google_admin_email")}
	if cn.Protocol == connections.OIDC {
		o := connections.OIDCConfig{Issuer: q.Form("issuer"), ClientID: q.Form("client_id"), AuthMethod: q.Form("auth_method"), Scopes: list(q, "scopes"),
			UseUserinfo: q.Bool("use_userinfo"), Prompt: q.Form("prompt"), RPLogout: q.Bool("rp_logout"), AcceptBearer: q.Bool("accept_bearer"),
			BearerAudiences: list(q, "bearer_audiences"), SigningKeyID: q.Form("signing_key_id"),
			SigningCertificate: strings.TrimSpace(q.R.PostFormValue("signing_certificate"))}
		for _, kv := range q.Lines("auth_params") {
			k, v, ok := strings.Cut(kv, "=")
			if !ok || strings.TrimSpace(k) == "" {
				return in, console.Invalid("Write each extra authorisation parameter as key=value.")
			}
			if o.AuthParams == nil {
				o.AuthParams = map[string]string{}
			}
			o.AuthParams[strings.TrimSpace(k)] = strings.TrimSpace(v)
		}
		c.OIDC, c.SAML = &o, nil
		in.Secrets = connections.Secrets{ClientSecret: q.R.PostFormValue("client_secret"), SigningKeyPEM: strings.TrimSpace(q.R.PostFormValue("signing_key_pem")),
			GoogleServiceAccount: strings.TrimSpace(q.R.PostFormValue("google_service_account"))}
	} else {
		s := connections.SAMLConfig{MetadataURL: q.Form("metadata_url"), MetadataXML: strings.TrimSpace(q.R.PostFormValue("metadata_xml")),
			NameIDFormat: q.Form("name_id_format"), AllowIDPInitiated: q.Bool("allow_idp_initiated"), SignRequests: q.Bool("sign_requests"),
			RPLogout: q.Bool("rp_logout")}
		if cfg.SAML != nil {
			s.SPCertificate = cfg.SAML.SPCertificate
		}
		c.SAML, c.OIDC = &s, nil
		in.Secrets = connections.Secrets{GraphClientSecret: q.R.PostFormValue("graph_client_secret")}
	}
	return in, nil
}
