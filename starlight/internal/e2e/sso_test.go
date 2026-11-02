package e2e_test

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"io"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/config"
	"github.com/ashhart/TensorFold/starlight/internal/discovery"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/identity/oidc/oidcfake"
	"github.com/ashhart/TensorFold/starlight/internal/identity/scim"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

var csrfField = regexp.MustCompile(`name="_csrf" value="([^"]+)"`)

// TestSSOToConnectToInference covers OIDC sign-in, group mapping, Connect PKCE, inference and SCIM deprovisioning.
func TestSSOToConnectToInference(t *testing.T) {
	idp := oidcfake.New("starlight", "client-secret")
	defer idp.Close()
	caFile := filepath.Join(t.TempDir(), "idp-ca.pem")
	if err := os.WriteFile(caFile, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: idp.Server.Certificate().Raw}), 0o600); err != nil {
		t.Fatal(err)
	}
	f := startStarlight(t, func(c *config.Config) { c.Identity.CAFile = caFile })
	f.enrolNode(f.nodeURL)
	ctx := context.Background()
	conn, err := f.starlight.Login.Connections().Create(ctx, connections.Input{Slug: "corp", Name: "Corp SSO", Protocol: connections.OIDC,
		Preset: "generic_oidc", Domains: []string{"example.org"}, Enabled: true, JIT: true,
		Config: connections.Config{OIDC: &connections.OIDCConfig{Issuer: idp.Issuer, ClientID: idp.ClientID, Scopes: []string{"openid", "email"}},
			Claims: connections.ClaimMap{Groups: "groups"}},
		Secrets: connections.Secrets{ClientSecret: idp.ClientSecret}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	group, err := directory.UpsertGroup(ctx, f.pool, &conn.ID, directory.FromClaim, directory.GroupRef{ExternalID: "ml-engineers"})
	if err != nil {
		t.Fatal(err)
	}
	exec(t, f.pool, "INSERT INTO role_bindings (id, role, subject_type, subject_id, project_id) VALUES ($1, 'member', 'group', $2, $3)", ids.New(), group, f.project)
	idp.SetUser(map[string]any{"sub": "u-77", "email": "hana@example.org", "email_verified": true, "name": "Hana", "groups": []string{"ml-engineers"}})

	jar, _ := cookiejar.New(nil)
	loopback := "http://127.0.0.1:41999/callback"
	browser := &http.Client{Jar: jar, Transport: idp.Server.Client().Transport, CheckRedirect: func(req *http.Request, _ []*http.Request) error {
		if strings.HasPrefix(req.URL.String(), loopback) {
			return http.ErrUseLastResponse
		}
		return nil
	}}
	verifier := strings.Repeat("k", 50)
	sum := sha256.Sum256([]byte(verifier))
	authz := url.Values{"client_id": {discovery.ClientConnect}, "redirect_uri": {loopback}, "response_type": {"code"}, "state": {"state-abcdef"},
		"code_challenge": {base64.RawURLEncoding.EncodeToString(sum[:])}, "code_challenge_method": {"S256"}, "scope": {"inference offline_access"},
		"device_name": {"Laptop"}}
	resp, err := browser.Get(f.public.URL + "/auth/corp/start?return_to=" + url.QueryEscape("/oauth/authorize?"+authz.Encode()))
	if err != nil {
		t.Fatal(err)
	}
	page, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	m := csrfField.FindStringSubmatch(string(page))
	if resp.StatusCode != 200 || m == nil || !strings.Contains(string(page), "hana@example.org") {
		t.Fatalf("approval page %d: %.400s", resp.StatusCode, page)
	}
	form := url.Values{"_csrf": {m[1]}, "decision": {"allow"}}
	for k, v := range authz {
		form[k] = v
	}
	req, _ := http.NewRequest("POST", f.public.URL+"/oauth/authorize", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Origin", f.public.URL)
	resp, err = browser.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	back, _ := url.Parse(resp.Header.Get("Location"))
	code := back.Query().Get("code")
	if code == "" || back.Query().Get("state") != "state-abcdef" {
		t.Fatalf("authorize redirect %d %q", resp.StatusCode, resp.Header.Get("Location"))
	}
	resp, err = http.PostForm(f.public.URL+"/oauth/token", url.Values{"grant_type": {"authorization_code"}, "client_id": {discovery.ClientConnect},
		"code": {code}, "redirect_uri": {loopback}, "code_verifier": {verifier}})
	if err != nil {
		t.Fatal(err)
	}
	var tok discovery.TokenResponse
	_ = json.NewDecoder(resp.Body).Decode(&tok)
	resp.Body.Close()
	if !strings.HasPrefix(tok.AccessToken, "sla_") || tok.RefreshToken == "" {
		t.Fatalf("token response %d %+v", resp.StatusCode, tok)
	}
	resp2, body := f.call("GET", "/oauth/userinfo", tok.AccessToken, "")
	if resp2.StatusCode != 200 || !strings.Contains(body, `"slug":"eng"`) || !strings.Contains(body, "qwen3.8-27b") {
		t.Fatalf("userinfo %d %s", resp2.StatusCode, body)
	}
	resp2, body = f.call("POST", "/v1/chat/completions", tok.AccessToken, `{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}]}`)
	if resp2.StatusCode != 200 || resp2.Header.Get("X-Starlight-Route") != "local" {
		t.Fatalf("inference with the Connect token %d %s", resp2.StatusCode, body)
	}

	user := lookupUser(t, f, "hana@example.org")
	expires := time.Now().Add(time.Hour)
	scimToken, _, err := scim.IssueToken(ctx, f.pool, conn.ID, "okta", &expires)
	if err != nil {
		t.Fatal(err)
	}
	patch := `{"schemas":["urn:ietf:params:scim:api:messages:2.0:PatchOp"],"Operations":[{"op":"Replace","value":{"active":"False"}}]}`
	resp2, body = f.call("PATCH", "/scim/v2/Users/"+user.ID, scimToken, patch, "Content-Type", "application/scim+json")
	if resp2.StatusCode != 200 && resp2.StatusCode != 204 {
		t.Fatalf("scim deactivate %d %s", resp2.StatusCode, body)
	}
	time.Sleep(200 * time.Millisecond)
	if resp2, _ = f.call("GET", "/v1/models", tok.AccessToken, ""); resp2.StatusCode != http.StatusUnauthorized {
		t.Fatalf("deprovisioned user's token still works: %d", resp2.StatusCode)
	}
	if st, _ := f.call("POST", "/oauth/token", "", ""); st.StatusCode == 200 {
		t.Fatal("empty token request succeeded")
	}
}

func lookupUser(t *testing.T, f *fleet, email string) directory.User {
	u, err := directory.FindByEmail(context.Background(), f.pool, email)
	if err != nil {
		t.Fatalf("user %s: %v", email, err)
	}
	return u
}
