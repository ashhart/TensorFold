package oidc_test

import (
	"context"
	"errors"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/oidc"
	"github.com/ashhart/TensorFold/starlight/internal/identity/oidc/oidcfake"
)

func conn(p *oidcfake.Provider) connections.Connection {
	return connections.Connection{ID: "c1", Slug: "corp", Protocol: connections.OIDC, Enabled: true, UpdatedAt: time.Unix(1, 0),
		Config: connections.Config{OIDC: &connections.OIDCConfig{Issuer: p.Issuer, ClientID: p.ClientID, AuthMethod: connections.AuthBasic,
			Scopes: []string{"openid", "email"}}}}
}

// signIn drives authorize and exchange without a browser, returning the exchange result.
func signIn(t *testing.T, p *oidcfake.Provider, c connections.Connection, nonce string, maxAge int) (*oidc.Result, error) {
	t.Helper()
	return signInWith(t, p, oidc.NewRP(p.Server.Client(), nil), c, connections.Secrets{ClientSecret: p.ClientSecret}, nonce, maxAge)
}

func signInWith(t *testing.T, p *oidcfake.Provider, rp *oidc.RP, c connections.Connection, sec connections.Secrets, nonce string,
	maxAge int) (*oidc.Result, error) {
	t.Helper()
	client, err := rp.Client(context.Background(), c, sec, "https://starlight.example/auth/corp/callback")
	if err != nil {
		t.Fatal(err)
	}
	verifier := "v-" + strings.Repeat("x", 50)
	authURL := client.AuthURL("state-1", "n-1", verifier, oidc.AuthOptions{MaxAge: maxAge})
	noFollow := &http.Client{Transport: p.Server.Client().Transport, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	resp, err := noFollow.Get(authURL)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	loc, _ := url.Parse(resp.Header.Get("Location"))
	if loc.Query().Get("state") != "state-1" {
		t.Fatalf("state not echoed: %s", loc)
	}
	return client.Exchange(context.Background(), loc.Query().Get("code"), verifier, nonce, maxAge)
}

func TestExchangeVerifiesTheIDToken(t *testing.T) {
	p := oidcfake.New("starlight", "secret")
	defer p.Close()
	p.SetUser(map[string]any{"sub": "u1", "email": "ada@example.com", "sid": "sess-9"})
	res, err := signIn(t, p, conn(p), "n-1", 0)
	if err != nil {
		t.Fatal(err)
	}
	if res.Subject != "u1" || res.Claims["email"] != "ada@example.com" || res.SessionID != "sess-9" || res.AuthTime.IsZero() {
		t.Fatalf("result %+v", res)
	}
}

func TestExchangeRejectsForgedTokens(t *testing.T) {
	cases := map[string]struct {
		alg    string
		mutate func(map[string]any)
		nonce  string
		want   error
	}{
		"nonce mismatch":   {nonce: "other", want: oidc.ErrNonce},
		"wrong audience":   {mutate: func(c map[string]any) { c["aud"] = "someone-else" }},
		"expired":          {mutate: func(c map[string]any) { c["exp"] = time.Now().Add(-time.Hour).Unix() }},
		"wrong issuer":     {mutate: func(c map[string]any) { c["iss"] = "https://evil.example" }},
		"alg none":         {alg: "none"},
		"alg HS256":        {alg: "HS256"},
		"foreign azp":      {mutate: func(c map[string]any) { c["azp"] = "other-client" }, want: oidc.ErrAuthorized},
		"multi aud no azp": {mutate: func(c map[string]any) { c["aud"] = []string{"starlight", "other"} }, want: oidc.ErrAuthorized},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			p := oidcfake.New("starlight", "secret")
			defer p.Close()
			if tc.alg != "" {
				p.SetAlg(tc.alg)
			}
			if tc.mutate != nil {
				p.Mutate(tc.mutate)
			}
			nonce := "n-1"
			if tc.nonce != "" {
				nonce = tc.nonce
			}
			_, err := signIn(t, p, conn(p), nonce, 0)
			if err == nil {
				t.Fatal("forged token accepted")
			}
			if tc.want != nil && !errors.Is(err, tc.want) {
				t.Fatalf("err = %v, want %v", err, tc.want)
			}
		})
	}
}

func TestMaxAgeAndUserinfo(t *testing.T) {
	p := oidcfake.New("starlight", "secret")
	defer p.Close()
	p.Mutate(func(c map[string]any) { c["auth_time"] = time.Now().Add(-2 * time.Hour).Unix() })
	if _, err := signIn(t, p, conn(p), "n-1", 60); !errors.Is(err, oidc.ErrAuthTime) {
		t.Fatalf("stale auth_time = %v", err)
	}
	p.Mutate(nil)
	c := conn(p)
	c.Config.OIDC.UseUserinfo = true
	p.SetUserinfo(map[string]any{"groups": []string{"eng"}, "email": "spoof@example.com"})
	p.SetUser(map[string]any{"sub": "u1", "email": "ada@example.com"})
	res, err := signIn(t, p, c, "n-1", 0)
	if err != nil {
		t.Fatal(err)
	}
	if res.Claims["email"] != "ada@example.com" || res.Claims["groups"] == nil {
		t.Fatalf("userinfo merge must fill gaps only: %+v", res.Claims)
	}
	p.SetUserinfo(map[string]any{"sub": "someone-else"})
	if _, err := signIn(t, p, c, "n-1", 0); !errors.Is(err, oidc.ErrUserinfoSub) {
		t.Fatalf("userinfo sub mismatch = %v", err)
	}
}

func TestTrailingSlashIssuerTolerated(t *testing.T) {
	p := oidcfake.New("starlight", "secret")
	defer p.Close()
	c := conn(p)
	c.Config.OIDC.Issuer = p.Issuer + "/"
	if _, err := signIn(t, p, c, "n-1", 0); err != nil {
		t.Fatalf("trailing slash issuer: %v", err)
	}
}

func TestBearerAndLogoutTokens(t *testing.T) {
	p := oidcfake.New("starlight", "secret")
	defer p.Close()
	c := conn(p)
	rp := oidc.NewRP(p.Server.Client(), nil)
	now := time.Now()
	logout := map[string]any{"iss": p.Issuer, "aud": "starlight", "iat": now.Unix(), "exp": now.Add(2 * time.Minute).Unix(), "jti": "j1", "sid": "s1",
		"events": map[string]any{"http://schemas.openid.net/event/backchannel-logout": map[string]any{}}}
	lt, err := rp.VerifyLogout(context.Background(), c, p.Sign(logout))
	if err != nil || lt.SessionID != "s1" || lt.TokenID != "j1" {
		t.Fatalf("logout = %+v %v", lt, err)
	}
	withNonce := map[string]any{}
	for k, v := range logout {
		withNonce[k] = v
	}
	withNonce["nonce"] = "n"
	if _, err := rp.VerifyLogout(context.Background(), c, p.Sign(withNonce)); err == nil {
		t.Fatal("logout token with a nonce accepted")
	}
	stale := map[string]any{}
	for k, v := range logout {
		stale[k] = v
	}
	stale["iat"] = now.Add(-time.Hour).Unix()
	stale["exp"] = now.Add(time.Hour).Unix()
	if _, err := rp.VerifyLogout(context.Background(), c, p.Sign(stale)); !errors.Is(err, oidc.ErrLogoutStale) {
		t.Fatalf("stale logout = %v", err)
	}
	if _, err := rp.VerifyLogout(context.Background(), c, p.SignWith("none", logout)); err == nil {
		t.Fatal("unsigned logout token accepted")
	}
	if iss, err := oidc.PeekIssuer(p.Sign(logout)); err != nil || iss != p.Issuer {
		t.Fatalf("peek = %q %v", iss, err)
	}
	if !oidc.LooksLikeJWT(p.Sign(logout)) || oidc.LooksLikeJWT("slr_abc") {
		t.Fatal("LooksLikeJWT misclassified")
	}
}
