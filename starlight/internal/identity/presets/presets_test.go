package presets

import (
	"strings"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
)

const tenant = "8d0c6f1e-2c43-4c2d-9a33-0b1c2d3e4f50"

func values(extra map[string]string) Values {
	v := Values{"name": "Example SSO", "slug": "corp", "domains": "example.com", "client_id": "starlight", "client_secret": "s3cret",
		"metadata_url": "https://idp.example.com/metadata", "jit": "on"}
	for k, x := range extra {
		v[k] = x
	}
	return v
}

func TestEveryPresetBuildsAValidConnection(t *testing.T) {
	extra := map[string]map[string]string{
		"okta":         {"okta_domain": "example.okta.com"},
		"entra":        {"entra_tenant_id": tenant},
		"google":       {"google_domains": "example.com"},
		"auth0":        {"auth0_domain": "example.us.auth0.com"},
		"onelogin":     {"onelogin_subdomain": "example"},
		"pingone":      {"pingone_region": "auth.pingone.eu", "pingone_environment": "env-1"},
		"duo":          {"issuer": "https://sso-example.sso.duosecurity.com/oidc/abc"},
		"gitlab":       {"gitlab_url": "https://gitlab.example.com"},
		"zitadel":      {"issuer": "https://example-abc123.zitadel.cloud"},
		"keycloak":     {"keycloak_url": "https://sso.example.com", "keycloak_realm": "example"},
		"authentik":    {"authentik_url": "https://auth.example.com", "authentik_slug": "starlight"},
		"pingfederate": {"issuer": "https://pf.example.com"},
		"adfs":         {"adfs_host": "fs.example.com"},
		"generic_oidc": {"issuer": "https://idp.example.com"},
	}
	if n := len(All()); n != 18 {
		t.Fatalf("catalogue has %d presets, want 18", n)
	}
	for _, p := range All() {
		for _, proto := range p.Protocols {
			in, err := p.Build(proto, values(extra[p.Name]))
			if err != nil {
				t.Errorf("%s/%s build: %v", p.Name, proto, err)
				continue
			}
			if err := in.Validate(); err != nil {
				t.Errorf("%s/%s validate: %v", p.Name, proto, err)
			}
			if p.Doc != "docs/sso/"+p.Name+".md" {
				t.Errorf("%s doc path %q", p.Name, p.Doc)
			}
		}
	}
}

func TestIssuerTemplates(t *testing.T) {
	cases := []struct {
		preset string
		v      map[string]string
		want   string
	}{
		{"okta", map[string]string{"okta_domain": "https://example.okta.com/"}, "https://example.okta.com"},
		{"okta", map[string]string{"okta_domain": "example.okta.com", "okta_server": "default"}, "https://example.okta.com/oauth2/default"},
		{"entra", map[string]string{"entra_tenant_id": tenant}, "https://login.microsoftonline.com/" + tenant + "/v2.0"},
		{"auth0", map[string]string{"auth0_domain": "example.eu.auth0.com"}, "https://example.eu.auth0.com/"},
		{"keycloak", map[string]string{"keycloak_url": "sso.example.com/", "keycloak_realm": "r"}, "https://sso.example.com/realms/r"},
		{"gitlab", map[string]string{}, "https://gitlab.com"},
		{"pingone", map[string]string{"pingone_region": "auth.pingone.com", "pingone_environment": "e1"}, "https://auth.pingone.com/e1/as"},
	}
	for _, c := range cases {
		p, _ := Get(c.preset)
		in, err := p.Build(connections.OIDC, values(c.v))
		if err != nil {
			t.Fatalf("%s: %v", c.preset, err)
		}
		if got := in.Config.OIDC.Issuer; got != c.want {
			t.Errorf("%s issuer = %q, want %q", c.preset, got, c.want)
		}
	}
}

func TestProviderDefaults(t *testing.T) {
	entra, _ := Get("entra")
	in, _ := entra.Build(connections.OIDC, values(map[string]string{"entra_tenant_id": tenant}))
	if in.Config.Claims.Subject != "oid" || in.Config.Claims.Tenant != "tid" || in.Config.Groups.Strategy != connections.GroupsEntraGraph {
		t.Fatalf("entra defaults %+v %+v", in.Config.Claims, in.Config.Groups)
	}
	google, _ := Get("google")
	in, _ = google.Build(connections.OIDC, values(map[string]string{"google_domains": "example.com", "google_admin_email": "admin@example.com",
		"google_service_account": `{"type":"service_account"}`}))
	if in.Config.OIDC.AuthParams["hd"] != "example.com" || in.Config.Groups.Strategy != connections.GroupsGoogleDirectory || in.Secrets.GoogleServiceAccount == "" {
		t.Fatalf("google defaults %+v", in.Config)
	}
	if _, err := google.Build(connections.OIDC, values(nil)); err != nil {
		t.Fatal(err)
	}
	noDomains, _ := google.Build(connections.OIDC, Values{"name": "G", "slug": "gw", "client_id": "x"})
	if err := noDomains.Validate(); err == nil || !strings.Contains(err.Error(), "google_domains") {
		t.Fatalf("google without domains validated: %v", err)
	}
	gitlab, _ := Get("gitlab")
	public, _ := gitlab.Build(connections.OIDC, Values{"name": "GitLab", "slug": "gl", "client_id": "x"})
	if err := public.Validate(); err == nil {
		t.Fatal("gitlab.com without required groups validated")
	}
	zitadel, _ := Get("zitadel")
	in, _ = zitadel.Build(connections.OIDC, values(map[string]string{"issuer": "https://z.example.com"}))
	if !in.Config.Claims.GroupsAreKeys {
		t.Fatal("zitadel roles claim must be read as object keys")
	}
	if _, err := zitadel.Build(connections.SAML, values(nil)); err == nil {
		t.Fatal("zitadel accepted SAML")
	}
}

func TestCertificateClientAuthentication(t *testing.T) {
	for _, name := range []string{"entra", "okta", "generic_oidc"} {
		p, _ := Get(name)
		fields := map[string]Field{}
		for _, f := range p.FieldsFor(connections.OIDC) {
			fields[f.Name] = f
		}
		if fields["auth_method"].Kind != KindSelect || fields["signing_certificate"].Kind != KindTextarea || !fields["signing_key_pem"].WriteOnly ||
			!fields["generate_certificate"].ConsoleOnly {
			t.Errorf("%s lacks the certificate fields: %+v", name, fields)
		}
		extra := map[string]string{"entra_tenant_id": tenant, "okta_domain": "example.okta.com", "issuer": "https://idp.example.com",
			"auth_method": connections.AuthPrivateJWT, "generate_certificate": "on", "client_secret": ""}
		in, err := p.Build(connections.OIDC, values(extra))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if _, cert, err := connections.ClientKey(in.Secrets.SigningKeyPEM, in.Config.OIDC.SigningCertificate); err != nil || cert == nil {
			t.Fatalf("%s generated no usable pair: %v", name, err)
		}
		if err := in.Validate(); err != nil || in.Config.OIDC.AuthMethod != connections.AuthPrivateJWT {
			t.Fatalf("%s validate: %v", name, err)
		}
		extra["signing_key_pem"] = in.Secrets.SigningKeyPEM
		if _, err := p.Build(connections.OIDC, values(extra)); err == nil {
			t.Errorf("%s accepted a pasted key alongside generate", name)
		}
	}
	entra, _ := Get("entra")
	if in, _ := entra.Build(connections.OIDC, values(map[string]string{"entra_tenant_id": tenant})); in.Config.OIDC.SigningCertificate != "" {
		t.Fatal("a key pair was generated without being asked for")
	}
}

func TestSAMLPresetsOfferSingleLogout(t *testing.T) {
	for _, p := range All() {
		if !p.Supports(connections.SAML) {
			continue
		}
		found := false
		for _, f := range p.FieldsFor(connections.SAML) {
			found = found || (f.Name == "rp_logout" && f.Kind == KindBool)
		}
		in, err := p.Build(connections.SAML, values(map[string]string{"rp_logout": "on", "entra_tenant_id": tenant, "google_domains": "example.com"}))
		if !found || err != nil || !in.Config.SAML.RPLogout {
			t.Errorf("%s: field %v, build %v, rp_logout %v", p.Name, found, err, in.Config.SAML != nil && in.Config.SAML.RPLogout)
		}
	}
}
