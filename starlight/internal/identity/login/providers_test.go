package login

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/oidc/oidcfake"
)

const (
	tenantID = "8d0c6f1e-2c43-4c2d-9a33-0b1c2d3e4f50"
	objectID = "0f9e8d7c-6b5a-4938-2716-05f4e3d2c1b0"
)

func entraEdit(in *connections.Input) {
	in.Preset = "entra"
	in.Config.Claims = connections.ClaimMap{Subject: "oid", Tenant: "tid", Email: "email", EmailVerified: "xms_edov", Name: "name",
		Username: "preferred_username", Groups: "groups"}
	in.Config.Groups.Strategy = connections.GroupsEntraGraph
	in.Config.Provider.EntraTenantID = tenantID
}

func TestEntraOverageReadsGroupsFromGraph(t *testing.T) {
	x := newHarness(t, 100)
	p := oidcfake.New("starlight", "secret")
	defer p.Close()
	graph := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/oauth2/v2.0/token"):
			fmt.Fprint(w, `{"access_token":"g","expires_in":3600}`)
		case strings.Contains(r.URL.Path, "/users/"+objectID+"/transitiveMemberOf"):
			fmt.Fprint(w, `{"value":[{"id":"11111111-0000-0000-0000-000000000001","displayName":"Starlight Admins"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer graph.Close()
	x.h.cfg.Entra.LoginBase, x.h.cfg.Entra.GraphBase = graph.URL, graph.URL
	x.oidcConn(p, entraEdit)
	p.SetUser(map[string]any{"sub": "pairwise", "tid": tenantID, "oid": objectID, "email": "ada@example.com", "preferred_username": "ada@example.com",
		"_claim_names": map[string]any{"groups": "src1"}, "_claim_sources": map[string]any{"src1": map[string]any{"endpoint": "https://graph.windows.net/x"}}})
	if code, body := x.get(x.client, "/auth/corp/start"); code != 200 {
		t.Fatalf("entra sign-in = %d %s", code, body)
	}
	u := x.user("ada@example.com")
	if g := x.groups(u.ID); !g["11111111-0000-0000-0000-000000000001"] {
		t.Fatalf("graph groups not synced: %v", g)
	}
	p.SetUser(map[string]any{"sub": "pairwise", "tid": "22222222-2222-2222-2222-222222222222", "oid": objectID, "email": "ada@example.com"})
	if code, body := x.get(x.browser(), "/auth/corp/start"); code != 403 || !strings.Contains(body, "different Microsoft Entra") {
		t.Fatalf("foreign tenant = %d", code)
	}
}

func TestEntraCertificateSignsInAndReadsGraph(t *testing.T) {
	x := newHarness(t, 100)
	p := oidcfake.New("starlight", "secret-the-tenant-forbids")
	defer p.Close()
	certPEM, keyPEM, err := connections.NewClientKeyPair("starlight-corp")
	if err != nil {
		t.Fatal(err)
	}
	if err := p.RequireCertificate(certPEM); err != nil {
		t.Fatal(err)
	}
	p.SetGraphGroups(objectID, map[string]string{"11111111-0000-0000-0000-000000000001": "Starlight Admins"})
	x.h.cfg.Entra.LoginBase, x.h.cfg.Entra.GraphBase = p.Issuer, p.Issuer
	x.oidcConn(p, func(in *connections.Input) {
		entraEdit(in)
		in.Config.OIDC.AuthMethod, in.Config.OIDC.SigningCertificate = connections.AuthPrivateJWT, certPEM
		in.Secrets = connections.Secrets{SigningKeyPEM: keyPEM}
	})
	p.SetUser(map[string]any{"sub": "pairwise", "tid": tenantID, "oid": objectID, "email": "ada@example.com", "preferred_username": "ada@example.com",
		"_claim_names": map[string]any{"groups": "src1"}, "_claim_sources": map[string]any{"src1": map[string]any{"endpoint": "https://graph.windows.net/x"}}})
	if code, body := x.get(x.client, "/auth/corp/start"); code != 200 {
		t.Fatalf("certificate sign-in = %d %s", code, body)
	}
	if g := x.groups(x.user("ada@example.com").ID); !g["11111111-0000-0000-0000-000000000001"] {
		t.Fatalf("graph groups not synced: %v", g)
	}
	got := p.Assertions()
	if len(got) != 2 || got[0].Grant != "authorization_code" || got[1].Grant != "client_credentials" ||
		got[1].Audience != p.Issuer+"/"+tenantID+"/oauth2/v2.0/token" || got[0].ID == got[1].ID {
		t.Fatalf("token endpoint saw %+v", got)
	}
}

func TestGoogleHostedDomainAndDirectoryGroups(t *testing.T) {
	x := newHarness(t, 100)
	p := oidcfake.New("starlight", "secret")
	defer p.Close()
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	der, _ := x509.MarshalPKCS8PrivateKey(key)
	sa, _ := json.Marshal(map[string]string{"type": "service_account", "client_email": "reader@project.iam.example",
		"private_key": string(pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der}))})
	dir := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/token" {
			fmt.Fprint(w, `{"access_token":"d","expires_in":3600}`)
			return
		}
		if r.URL.Query().Get("userKey") == "ada@example.com" {
			fmt.Fprint(w, `{"groups":[{"email":"eng@example.com","name":"Engineering"}]}`)
			return
		}
		fmt.Fprint(w, `{}`)
	}))
	defer dir.Close()
	x.h.cfg.Google.TokenURL, x.h.cfg.Google.DirectoryBase = dir.URL+"/token", dir.URL
	x.oidcConn(p, func(in *connections.Input) {
		in.Preset = "google"
		in.Config.Provider.GoogleDomains = []string{"example.com"}
		in.Config.Provider.GoogleAdminEmail = "admin@example.com"
		in.Config.Groups.Strategy = connections.GroupsGoogleDirectory
		in.Secrets.GoogleServiceAccount = string(sa)
	})
	p.SetUser(map[string]any{"sub": "g-1", "hd": "other.example", "email": "ada@other.example", "email_verified": true})
	if code, body := x.get(x.client, "/auth/corp/start"); code != 403 || !strings.Contains(body, "work account") {
		t.Fatalf("wrong hd = %d", code)
	}
	p.SetUser(map[string]any{"sub": "g-1", "hd": "example.com", "email": "ada@example.com", "email_verified": true})
	if code, _ := x.get(x.browser(), "/auth/corp/start"); code != 200 {
		t.Fatalf("google sign-in = %d", code)
	}
	if g := x.groups(x.user("ada@example.com").ID); !g["eng@example.com"] {
		t.Fatalf("directory groups %v", g)
	}
}

func TestUnverifiedEmailCannotTakeOverAnAccount(t *testing.T) {
	x := newHarness(t, 100)
	p := oidcfake.New("starlight", "secret")
	defer p.Close()
	x.oidcConn(p, nil)
	p.SetUser(map[string]any{"sub": "victim", "email": "ceo@example.com", "email_verified": true})
	if code, _ := x.get(x.client, "/auth/corp/start"); code != 200 {
		t.Fatal("victim sign-in failed")
	}
	p.SetUser(map[string]any{"sub": "attacker", "email": "ceo@example.com", "email_verified": false})
	if code, body := x.get(x.browser(), "/auth/corp/start"); code != 403 || !strings.Contains(body, "could not be linked") {
		t.Fatalf("takeover attempt = %d %s", code, body)
	}
}
