// Package oidcfake is an in-process OpenID Connect provider for end-to-end tests of Starlight's sign-in.
package oidcfake

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/go-jose/go-jose/v4"
)

// Provider is a TLS test IdP that signs RS256 tokens with an in-memory key; it also plays Entra's Graph for overage.
type Provider struct {
	Server       *httptest.Server
	Issuer       string
	ClientID     string
	ClientSecret string

	mu         sync.Mutex
	key        *rsa.PrivateKey
	codes      map[string]grant
	access     map[string]map[string]any
	claims     map[string]any
	mutate     func(map[string]any)
	alg        string
	userinfo   map[string]any
	cert       *x509.Certificate
	jtis       map[string]bool
	assertions []Assertion
	apps       map[string]bool
	graph      map[string]map[string]string
}

type grant struct {
	nonce     string
	challenge string
	redirect  string
	claims    map[string]any
}

// New starts a provider whose issuer is its own TLS URL.
func New(clientID, clientSecret string) *Provider {
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		panic(err)
	}
	p := &Provider{ClientID: clientID, ClientSecret: clientSecret, key: key, codes: map[string]grant{}, access: map[string]map[string]any{},
		claims: map[string]any{"sub": "user-1"}, alg: "RS256", jtis: map[string]bool{}, apps: map[string]bool{}, graph: map[string]map[string]string{}}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /.well-known/openid-configuration", p.discovery)
	mux.HandleFunc("GET /jwks", p.jwks)
	mux.HandleFunc("GET /authorize", p.authorize)
	mux.HandleFunc("POST /token", p.token)
	mux.HandleFunc("POST /{tenant}/oauth2/v2.0/token", p.token)
	mux.HandleFunc("GET /v1.0/users/{oid}/transitiveMemberOf/microsoft.graph.group", p.graphGroups)
	mux.HandleFunc("GET /userinfo", p.userinfoHandler)
	p.Server = httptest.NewTLSServer(mux)
	p.Issuer = p.Server.URL
	return p
}

// Close stops the server.
func (p *Provider) Close() { p.Server.Close() }

// SetUser replaces the claims the next sign-in's ID token carries; sub defaults to user-1.
func (p *Provider) SetUser(claims map[string]any) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.claims = map[string]any{"sub": "user-1"}
	for k, v := range claims {
		p.claims[k] = v
	}
}

// Mutate edits ID token claims just before signing, to forge bad tokens.
func (p *Provider) Mutate(f func(map[string]any)) {
	p.mu.Lock()
	p.mutate = f
	p.mu.Unlock()
}

// SetAlg makes ID tokens use RS256, HS256 or none.
func (p *Provider) SetAlg(alg string) {
	p.mu.Lock()
	p.alg = alg
	p.mu.Unlock()
}

// SetUserinfo sets the claims the userinfo endpoint returns.
func (p *Provider) SetUserinfo(claims map[string]any) {
	p.mu.Lock()
	p.userinfo = claims
	p.mu.Unlock()
}

func (p *Provider) discovery(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, map[string]any{
		"issuer": p.Issuer, "authorization_endpoint": p.Issuer + "/authorize", "token_endpoint": p.Issuer + "/token",
		"jwks_uri": p.Issuer + "/jwks", "userinfo_endpoint": p.Issuer + "/userinfo", "end_session_endpoint": p.Issuer + "/logout",
		"id_token_signing_alg_values_supported": []string{"RS256"}, "backchannel_logout_supported": true,
	})
}

func (p *Provider) jwks(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, jose.JSONWebKeySet{Keys: []jose.JSONWebKey{{Key: &p.key.PublicKey, KeyID: "k1", Algorithm: "RS256", Use: "sig"}}})
}

func (p *Provider) authorize(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	if q.Get("client_id") != p.ClientID || q.Get("response_type") != "code" || q.Get("code_challenge_method") != "S256" {
		http.Error(w, "bad authorization request", http.StatusBadRequest)
		return
	}
	code := randomString()
	p.mu.Lock()
	claims := map[string]any{}
	for k, v := range p.claims {
		claims[k] = v
	}
	p.codes[code] = grant{nonce: q.Get("nonce"), challenge: q.Get("code_challenge"), redirect: q.Get("redirect_uri"), claims: claims}
	p.mu.Unlock()
	target, _ := url.Parse(q.Get("redirect_uri"))
	v := target.Query()
	v.Set("code", code)
	v.Set("state", q.Get("state"))
	target.RawQuery = v.Encode()
	http.Redirect(w, r, target.String(), http.StatusFound)
}

func (p *Provider) token(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad form", http.StatusBadRequest)
		return
	}
	if err := p.authenticate(r); err != nil {
		w.WriteHeader(http.StatusUnauthorized)
		writeJSON(w, map[string]string{"error": "invalid_client", "error_description": err.Error()})
		return
	}
	if r.PostForm.Get("grant_type") == "client_credentials" {
		app := randomString()
		p.mu.Lock()
		p.apps[app] = true
		p.mu.Unlock()
		writeJSON(w, map[string]any{"access_token": app, "token_type": "Bearer", "expires_in": 3600})
		return
	}
	p.mu.Lock()
	g, found := p.codes[r.PostForm.Get("code")]
	delete(p.codes, r.PostForm.Get("code"))
	p.mu.Unlock()
	sum := sha256.Sum256([]byte(r.PostForm.Get("code_verifier")))
	if !found || g.redirect != r.PostForm.Get("redirect_uri") || base64.RawURLEncoding.EncodeToString(sum[:]) != g.challenge {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, map[string]string{"error": "invalid_grant"})
		return
	}
	now := time.Now()
	claims := map[string]any{"iss": p.Issuer, "aud": p.ClientID, "iat": now.Unix(), "exp": now.Add(5 * time.Minute).Unix(),
		"auth_time": now.Unix(), "nonce": g.nonce}
	for k, v := range g.claims {
		claims[k] = v
	}
	p.mu.Lock()
	mutate, alg := p.mutate, p.alg
	p.mu.Unlock()
	if mutate != nil {
		mutate(claims)
	}
	access := randomString()
	p.mu.Lock()
	p.access[access] = claims
	p.mu.Unlock()
	writeJSON(w, map[string]any{"access_token": access, "token_type": "Bearer", "expires_in": 300, "id_token": p.SignWith(alg, claims)})
}

func (p *Provider) userinfoHandler(w http.ResponseWriter, r *http.Request) {
	tok := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	p.mu.Lock()
	claims, ok := p.access[tok]
	extra := p.userinfo
	p.mu.Unlock()
	if !ok {
		w.WriteHeader(http.StatusUnauthorized)
		return
	}
	out := map[string]any{"sub": claims["sub"]}
	for k, v := range extra {
		out[k] = v
	}
	writeJSON(w, out)
}

// Sign returns an RS256 JWT over claims signed with the provider's published key.
func (p *Provider) Sign(claims map[string]any) string { return p.SignWith("RS256", claims) }

// SignWith signs with RS256, HS256 (client secret) or produces an unsigned alg none token.
func (p *Provider) SignWith(alg string, claims map[string]any) string {
	body, _ := json.Marshal(claims)
	switch alg {
	case "none":
		return base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"none","typ":"JWT"}`)) + "." + base64.RawURLEncoding.EncodeToString(body) + "."
	case "HS256":
		s, _ := jose.NewSigner(jose.SigningKey{Algorithm: jose.HS256, Key: []byte(p.ClientSecret)}, (&jose.SignerOptions{}).WithType("JWT"))
		jws, _ := s.Sign(body)
		out, _ := jws.CompactSerialize()
		return out
	}
	s, _ := jose.NewSigner(jose.SigningKey{Algorithm: jose.RS256, Key: jose.JSONWebKey{Key: p.key, KeyID: "k1"}}, (&jose.SignerOptions{}).WithType("JWT"))
	jws, _ := s.Sign(body)
	out, _ := jws.CompactSerialize()
	return out
}

func randomString() string {
	b := make([]byte, 18)
	_, _ = rand.Read(b)
	return base64.RawURLEncoding.EncodeToString(b)
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}
