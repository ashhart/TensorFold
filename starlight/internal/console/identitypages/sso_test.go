package identitypages_test

import (
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/scim"
)

var scimToken = regexp.MustCompile(`sls_[a-z2-7_]+`)

func editForm(name string) url.Values {
	return url.Values{"name": {name}, "slug": {"corp"}, "domains": {"example.com"}, "jit": {"on"}, "issuer": {"https://idp.example.com"},
		"client_id": {"starlight"}, "client_secret": {""}, "auth_method": {connections.AuthBasic}, "scopes": {"openid\nemail\nprofile"},
		"subject_claim": {"sub"}, "email_claim": {"email"}, "email_verified_claim": {"email_verified"}, "name_claim": {"name"},
		"groups_claim": {"groups"}, "groups_strategy": {connections.GroupsClaim}, "nested_depth": {"0"}}
}

func TestConnectionFromPresetEditKeepsSecrets(t *testing.T) {
	r := newRig(t)
	admin := r.user("tess@example.com", "security_admin")
	create := url.Values{"preset": {"generic_oidc"}, "protocol": {"oidc"}, "name": {"Corp SSO"}, "slug": {"corp"}, "issuer": {"https://idp.example.com"},
		"client_id": {"starlight"}, "client_secret": {"s3cret-value"}, "domains": {"example.com"}, "jit": {"on"}, "groups_claim": {"groups"}}
	code, body := admin.post("/console/sso", create)
	if code != 200 || !strings.Contains(body, r.srv.URL+"/auth/corp/callback") || strings.Contains(body, "s3cret-value") {
		t.Fatalf("create %d: %.300s", code, body)
	}
	store := r.c.Login.Connections()
	cn, err := store.GetBySlug(r.ctx, "corp")
	if err != nil || cn.Config.OIDC.Issuer != "https://idp.example.com" || !cn.JIT {
		t.Fatalf("stored %+v %v", cn, err)
	}
	code, body = admin.get("/console/sso/corp")
	if code != 200 || strings.Contains(body, "s3cret-value") || !strings.Contains(body, `value="https://idp.example.com"`) {
		t.Fatalf("detail %d leaked or lost settings", code)
	}
	if code, _ := admin.post("/console/sso/corp", editForm("Corp SSO renamed")); code != http.StatusSeeOther {
		t.Fatalf("edit %d", code)
	}
	cn, _ = store.GetBySlug(r.ctx, "corp")
	sec, err := store.Secrets(r.ctx, cn)
	if cn.Name != "Corp SSO renamed" || err != nil || sec.ClientSecret != "s3cret-value" {
		t.Fatalf("edit lost the secret or the change: %q %q %v", cn.Name, sec.ClientSecret, err)
	}
	bad := editForm("x")
	bad.Set("issuer", "http://idp.example.com")
	if code, body := admin.post("/console/sso/corp", bad); code != http.StatusUnprocessableEntity || !strings.Contains(body, "https") {
		t.Fatalf("plain-http issuer accepted: %d", code)
	}
	if code, _ := admin.post("/console/sso/corp/enabled", url.Values{"enabled": {"off"}}); code != http.StatusSeeOther {
		t.Fatalf("disable %d", code)
	}
	if cn, _ = store.GetBySlug(r.ctx, "corp"); cn.Enabled {
		t.Fatal("connection still enabled")
	}
	if code, _ := admin.post("/console/sso/corp/delete", url.Values{"confirm": {"wrong"}}); code != http.StatusUnprocessableEntity {
		t.Fatalf("delete without confirmation %d", code)
	}
	if code, _ := admin.post("/console/sso/corp/delete", url.Values{"confirm": {"corp"}}); code != http.StatusSeeOther {
		t.Fatalf("delete %d", code)
	}
	if _, err := store.GetBySlug(r.ctx, "corp"); err != connections.ErrNotFound {
		t.Fatalf("connection survived deletion: %v", err)
	}
	for _, a := range []string{"sso.connection_created", "sso.connection_updated", "sso.connection_disabled", "sso.connection_deleted"} {
		if r.audited(a) != 1 {
			t.Errorf("audit %s = %d", a, r.audited(a))
		}
	}
}

func TestSCIMTokensShownOnceAndRevocable(t *testing.T) {
	r := newRig(t)
	admin := r.user("uma@example.com", "owner")
	admin.post("/console/sso", url.Values{"preset": {"generic_oidc"}, "protocol": {"oidc"}, "name": {"Corp"}, "slug": {"corp"},
		"issuer": {"https://idp.example.com"}, "client_id": {"starlight"}})
	code, body := admin.post("/console/sso/corp/scim", url.Values{"label": {"okta-provisioning"}, "expires_days": {"30"}})
	tok := scimToken.FindString(body)
	if code != 200 || tok == "" {
		t.Fatalf("issue %d: no token shown", code)
	}
	_, body = admin.get("/console/sso/corp")
	if strings.Contains(body, tok) || !strings.Contains(body, "okta-provisioning") {
		t.Fatal("token shown again, or its label missing")
	}
	cn, _ := r.c.Login.Connections().GetBySlug(r.ctx, "corp")
	list, err := scim.ListTokens(r.ctx, r.pool, cn.ID)
	if err != nil || len(list) != 1 || list[0].ExpiresAt == nil {
		t.Fatalf("tokens %+v %v", list, err)
	}
	if code, _ := admin.post("/console/sso/corp/scim/"+list[0].ID+"/revoke", nil); code != http.StatusSeeOther {
		t.Fatalf("revoke %d", code)
	}
	list, _ = scim.ListTokens(r.ctx, r.pool, cn.ID)
	if list[0].RevokedAt == nil || r.audited("scim.token_issued") != 1 || r.audited("scim.token_revoked") != 1 {
		t.Fatalf("revocation not recorded: %+v", list[0])
	}
}

func TestSAMLConnectionShowsProviderValues(t *testing.T) {
	r := newRig(t)
	admin := r.user("gil@example.com", "security_admin")
	code, body := admin.post("/console/sso", url.Values{"preset": {"generic_saml"}, "protocol": {"saml"}, "name": {"Shibboleth"}, "slug": {"campus"},
		"metadata_url": {"https://idp.example.org/metadata"}, "name_id_format": {"persistent"}})
	if code != 200 || !strings.Contains(body, r.srv.URL+"/auth/campus/saml/acs") {
		t.Fatalf("create %d: %.300s", code, body)
	}
	code, body = admin.get("/console/sso/campus")
	if code != 200 || !strings.Contains(body, "BEGIN CERTIFICATE") || !strings.Contains(body, `name="metadata_url"`) || strings.Contains(body, `name="issuer"`) {
		t.Fatalf("saml detail %d", code)
	}
	edit := url.Values{"name": {"Shibboleth"}, "slug": {"campus"}, "metadata_url": {"https://idp.example.org/metadata"}, "name_id_format": {"transient"},
		"sign_requests": {"on"}, "groups_strategy": {connections.GroupsNone}}
	if code, _ := admin.post("/console/sso/campus", edit); code != http.StatusSeeOther {
		t.Fatalf("saml edit %d", code)
	}
	cn, _ := r.c.Login.Connections().GetBySlug(r.ctx, "campus")
	if cn.Config.SAML.NameIDFormat != "transient" || cn.Config.SAML.SPCertificate == "" {
		t.Fatalf("saml config after edit %+v", cn.Config.SAML)
	}
}

var thumbprint = regexp.MustCompile(`<code>[0-9A-F]{40}</code>`)

func TestCertificateConnectionGeneratedAndDownloaded(t *testing.T) {
	r := newRig(t)
	admin := r.user("ivo@example.com", "security_admin")
	const tenant = "8d0c6f1e-2c43-4c2d-9a33-0b1c2d3e4f50"
	code, body := admin.post("/console/sso", url.Values{"preset": {"entra"}, "protocol": {"oidc"}, "name": {"Entra"}, "slug": {"entra"},
		"entra_tenant_id": {tenant}, "client_id": {"starlight-app"}, "auth_method": {connections.AuthPrivateJWT}, "generate_certificate": {"on"}})
	if code != 200 || !strings.Contains(body, "BEGIN CERTIFICATE") || !strings.Contains(body, "/console/sso/entra/client-certificate") ||
		strings.Contains(body, "PRIVATE KEY") {
		t.Fatalf("create %d: %.400s", code, body)
	}
	store := r.c.Login.Connections()
	cn, _ := store.GetBySlug(r.ctx, "entra")
	sec, _ := store.Secrets(r.ctx, cn)
	if _, cert, err := connections.ClientKey(sec.SigningKeyPEM, cn.Config.OIDC.SigningCertificate); err != nil || cert == nil {
		t.Fatalf("stored pair: %v", err)
	}
	req, _ := http.NewRequest("GET", r.srv.URL+"/console/sso/entra/client-certificate", nil)
	req.AddCookie(admin.ck)
	resp, err := noFollow.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	pemBody, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 || !strings.Contains(resp.Header.Get("Content-Disposition"), `attachment; filename="starlight-entra.pem"`) ||
		string(pemBody) != cn.Config.OIDC.SigningCertificate {
		t.Fatalf("download %d %q", resp.StatusCode, resp.Header.Get("Content-Disposition"))
	}
	code, body = admin.get("/console/sso/entra")
	if code != 200 || !thumbprint.MatchString(body) || !strings.Contains(body, "x5t#S256") || strings.Contains(body, "PRIVATE KEY") {
		t.Fatalf("detail %d", code)
	}
	edit := url.Values{"name": {"Entra"}, "slug": {"entra"}, "issuer": {"https://login.microsoftonline.com/" + tenant + "/v2.0"}, "client_id": {"starlight-app"},
		"auth_method": {connections.AuthPrivateJWT}, "signing_certificate": {cn.Config.OIDC.SigningCertificate}, "subject_claim": {"oid"},
		"tenant_claim": {"tid"}, "email_claim": {"email"}, "groups_claim": {"groups"}, "groups_strategy": {connections.GroupsEntraGraph},
		"entra_tenant_id": {tenant}, "graph_client_id": {"starlight-app"}, "scopes": {"openid\nemail\nprofile"}}
	if code, body := admin.post("/console/sso/entra", edit); code != http.StatusSeeOther {
		t.Fatalf("edit %d: %.300s", code, body)
	}
	if after, _ := store.Secrets(r.ctx, cn); after.SigningKeyPEM != sec.SigningKeyPEM {
		t.Fatal("an edit that left the key field empty lost the key")
	}
	if code, _ := admin.post("/console/sso/entra/client-certificate", nil); code != http.StatusSeeOther {
		t.Fatalf("generate %d", code)
	}
	rotated, _ := store.GetBySlug(r.ctx, "entra")
	if rotated.Config.OIDC.SigningCertificate == cn.Config.OIDC.SigningCertificate || r.audited("sso.client_certificate_generated") != 1 {
		t.Fatal("certificate not replaced, or the replacement not audited")
	}
	viewer := r.user("una@example.com", "")
	if code, _ := viewer.post("/console/sso/entra/client-certificate", nil); code != http.StatusForbidden {
		t.Fatalf("certificate generated without SSO write access: %d", code)
	}
}

func TestPastedKeyNeverReachesAuditors(t *testing.T) {
	r := newRig(t)
	admin := r.user("sec@example.com", "security_admin")
	auditor := r.user("aud@example.com", "auditor")
	certPEM, keyPEM, _ := connections.NewClientKeyPair("starlight-corp")
	form := func(cert string) url.Values {
		return url.Values{"preset": {"generic_oidc"}, "protocol": {"oidc"}, "name": {"Corp"}, "slug": {"corp"}, "issuer": {"https://idp.example.com"},
			"client_id": {"starlight"}, "auth_method": {connections.AuthPrivateJWT}, "signing_certificate": {cert}, "signing_key_pem": {keyPEM}}
	}
	if code, body := admin.post("/console/sso", form(certPEM+keyPEM)); code != http.StatusUnprocessableEntity || !strings.Contains(body, "stored unsealed") {
		t.Fatalf("certificate pasted with its key = %d", code)
	}
	if _, err := r.c.Login.Connections().GetBySlug(r.ctx, "corp"); err != connections.ErrNotFound {
		t.Fatalf("the connection was saved with the key in its certificate field: %v", err)
	}
	if code, body := admin.post("/console/sso", form(certPEM)); code != 200 {
		t.Fatalf("create %d: %.300s", code, body)
	}
	for _, path := range []string{"/console/sso/corp", "/console/sso/corp/client-certificate"} {
		if code, body := auditor.get(path); code != 200 || !strings.Contains(body, "BEGIN CERTIFICATE") || strings.Contains(body, "PRIVATE KEY") {
			t.Fatalf("auditor %s = %d", path, code)
		}
	}
}

func TestSAMLSingleLogoutOptionInTheConsole(t *testing.T) {
	r := newRig(t)
	admin := r.user("lea@example.com", "security_admin")
	code, body := admin.post("/console/sso", url.Values{"preset": {"generic_saml"}, "protocol": {"saml"}, "name": {"Campus"}, "slug": {"campus"},
		"metadata_url": {"https://idp.example.org/metadata"}, "rp_logout": {"on"}})
	if code != 200 || !strings.Contains(body, r.srv.URL+"/auth/campus/saml/slo") {
		t.Fatalf("create %d: %.300s", code, body)
	}
	store := r.c.Login.Connections()
	if cn, _ := store.GetBySlug(r.ctx, "campus"); !cn.Config.SAML.RPLogout {
		t.Fatal("Sign out of the IdP too was not saved")
	}
	if _, body := admin.get("/console/sso/campus"); !strings.Contains(body, `name="rp_logout" checked`) {
		t.Fatal("detail page does not show the sign-out setting")
	}
	edit := url.Values{"name": {"Campus"}, "slug": {"campus"}, "metadata_url": {"https://idp.example.org/metadata"}, "sign_requests": {"on"},
		"groups_strategy": {connections.GroupsNone}}
	if code, _ := admin.post("/console/sso/campus", edit); code != http.StatusSeeOther {
		t.Fatalf("edit %d", code)
	}
	if cn, _ := store.GetBySlug(r.ctx, "campus"); cn.Config.SAML.RPLogout {
		t.Fatal("clearing the checkbox did not turn single logout off")
	}
}
