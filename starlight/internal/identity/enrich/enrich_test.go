package enrich

import (
	"context"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/oidc"
	"github.com/ashhart/TensorFold/starlight/internal/identity/oidc/oidcfake"
)

const (
	tenant = "8d0c6f1e-2c43-4c2d-9a33-0b1c2d3e4f50"
	user   = "0f9e8d7c-6b5a-4938-2716-05f4e3d2c1b0"
)

func TestEntraPagesTransitiveGroupsAndCachesTokens(t *testing.T) {
	var tokens atomic.Int32
	var srv *httptest.Server
	srv = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/"+tenant+"/oauth2/v2.0/token":
			_ = r.ParseForm()
			if r.PostForm.Get("grant_type") != "client_credentials" || r.PostForm.Get("client_secret") != "graph-secret" {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}
			tokens.Add(1)
			fmt.Fprint(w, `{"access_token":"graph-token","expires_in":3600}`)
		case strings.HasPrefix(r.URL.Path, "/v1.0/users/"+user+"/transitiveMemberOf"):
			if r.Header.Get("Authorization") != "Bearer graph-token" {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}
			if r.URL.Query().Get("page") == "2" {
				fmt.Fprint(w, `{"value":[{"id":"BBBB","displayName":"Ops"}]}`)
				return
			}
			fmt.Fprintf(w, `{"value":[{"id":"AAAA","displayName":"Engineering"}],"@odata.nextLink":"%s/v1.0/users/%s/transitiveMemberOf/microsoft.graph.group?page=2"}`,
				srv.URL, user)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	e := NewEntra(srv.Client())
	e.LoginBase, e.GraphBase = srv.URL, srv.URL
	for i := 0; i < 2; i++ {
		groups, err := e.Groups(context.Background(), tenant, oidc.SecretCredential("client", "graph-secret"), user)
		if err != nil {
			t.Fatal(err)
		}
		if len(groups) != 2 || groups[0].ExternalID != "aaaa" || groups[1].DisplayName != "Ops" {
			t.Fatalf("groups %+v", groups)
		}
	}
	if tokens.Load() != 1 {
		t.Fatalf("token fetched %d times, want 1", tokens.Load())
	}
	if _, err := e.Groups(context.Background(), tenant, oidc.SecretCredential("client", "graph-secret"), "../me"); err == nil {
		t.Fatal("accepted a non-GUID user id")
	}
}

func TestEntraRefusesForeignPagingLinks(t *testing.T) {
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/token") {
			fmt.Fprint(w, `{"access_token":"t","expires_in":3600}`)
			return
		}
		fmt.Fprint(w, `{"value":[],"@odata.nextLink":"https://collector.example/steal"}`)
	}))
	defer srv.Close()
	e := NewEntra(srv.Client())
	e.LoginBase, e.GraphBase = srv.URL, srv.URL
	if _, err := e.Groups(context.Background(), tenant, oidc.SecretCredential("c", "s"), user); err == nil || !strings.Contains(err.Error(), "leaves") {
		t.Fatalf("followed a foreign link: %v", err)
	}
}

func TestEntraGraphUsesTheConnectionCertificate(t *testing.T) {
	p := oidcfake.New("starlight", "unused")
	defer p.Close()
	certPEM, keyPEM, err := connections.NewClientKeyPair("starlight-entra")
	if err != nil {
		t.Fatal(err)
	}
	if err := p.RequireCertificate(certPEM); err != nil {
		t.Fatal(err)
	}
	p.SetGraphGroups(user, map[string]string{"AAAA": "Engineering"})
	e := NewEntra(p.Server.Client())
	e.LoginBase, e.GraphBase = p.Issuer, p.Issuer
	if _, err := e.Groups(context.Background(), tenant, oidc.SecretCredential("starlight", "graph-secret"), user); err == nil {
		t.Fatal("Graph token issued for a secret where the tenant requires a certificate")
	}
	cred, err := oidc.KeyCredential("starlight", keyPEM, certPEM, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	groups, err := e.Groups(context.Background(), tenant, cred, user)
	if err != nil || len(groups) != 1 || groups[0].ExternalID != "aaaa" {
		t.Fatalf("groups %+v, %v", groups, err)
	}
	got := p.Assertions()
	if len(got) != 1 || got[0].Grant != "client_credentials" || got[0].Audience != p.Issuer+"/"+tenant+"/oauth2/v2.0/token" {
		t.Fatalf("assertions %+v", got)
	}
	if _, err := e.Groups(context.Background(), tenant, oidc.SecretCredential("", ""), user); err == nil {
		t.Fatal("accepted missing Graph credentials")
	}
}

func newServiceAccount(t *testing.T) (string, *rsa.PublicKey) {
	t.Helper()
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	der, _ := x509.MarshalPKCS8PrivateKey(key)
	pemKey := string(pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der}))
	b, _ := json.Marshal(map[string]string{"type": "service_account", "client_email": "reader@project.iam.example", "private_key": pemKey,
		"private_key_id": "kid-1"})
	return string(b), &key.PublicKey
}

func TestGoogleDelegatedDirectoryGroupsWithNesting(t *testing.T) {
	saJSON, pub := newServiceAccount(t)
	var srv *httptest.Server
	srv = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/token":
			_ = r.ParseForm()
			parts := strings.Split(r.PostForm.Get("assertion"), ".")
			sig, _ := base64.RawURLEncoding.DecodeString(parts[2])
			sum := sha256.Sum256([]byte(parts[0] + "." + parts[1]))
			body, _ := base64.RawURLEncoding.DecodeString(parts[1])
			var claims map[string]any
			_ = json.Unmarshal(body, &claims)
			if rsa.VerifyPKCS1v15(pub, crypto.SHA256, sum[:], sig) != nil || claims["sub"] != "admin@example.com" || claims["scope"] != GroupScope ||
				claims["aud"] != srv.URL+"/token" {
				w.WriteHeader(http.StatusUnauthorized)
				fmt.Fprint(w, `{"error":"invalid_grant"}`)
				return
			}
			fmt.Fprint(w, `{"access_token":"dir-token","expires_in":3600}`)
		case "/admin/directory/v1/groups":
			if r.Header.Get("Authorization") != "Bearer dir-token" {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}
			switch key := r.URL.Query().Get("userKey"); {
			case key == "ada@example.com" && r.URL.Query().Get("pageToken") == "":
				fmt.Fprint(w, `{"groups":[{"email":"Eng@example.com","name":"Engineering"}],"nextPageToken":"p2"}`)
			case key == "ada@example.com":
				fmt.Fprint(w, `{"groups":[{"email":"all@example.com","name":"Everyone"}]}`)
			case key == "eng@example.com":
				fmt.Fprint(w, `{"groups":[{"email":"rnd@example.com","name":"R&D"}]}`)
			default:
				fmt.Fprint(w, `{}`)
			}
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	g := NewGoogle(srv.Client())
	g.TokenURL, g.DirectoryBase = srv.URL+"/token", srv.URL
	flat, err := g.Groups(context.Background(), saJSON, "admin@example.com", "ada@example.com", 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(flat) != 2 || flat[0].ExternalID != "eng@example.com" {
		t.Fatalf("direct groups %+v", flat)
	}
	nested, err := g.Groups(context.Background(), saJSON, "admin@example.com", "ada@example.com", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(nested) != 3 || nested[2].ExternalID != "rnd@example.com" {
		t.Fatalf("nested groups %+v", nested)
	}
	if _, err := g.Groups(context.Background(), `{"type":"authorized_user"}`, "admin@example.com", "ada@example.com", 0); err == nil {
		t.Fatal("accepted a non service-account key")
	}
}
