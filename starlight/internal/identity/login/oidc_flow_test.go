package login

import (
	"context"
	"errors"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/mapping"
	"github.com/ashhart/TensorFold/starlight/internal/identity/oidc"
	"github.com/ashhart/TensorFold/starlight/internal/identity/oidc/oidcfake"
)

func TestOIDCSignInCreatesSessionAndSyncsGroups(t *testing.T) {
	x := newHarness(t, 100)
	p := oidcfake.New("starlight", "secret")
	defer p.Close()
	x.oidcConn(p, nil)
	p.SetUser(map[string]any{"sub": "u-1", "email": "ada@example.com", "email_verified": true, "name": "Ada", "groups": []string{"eng", "ops"}})
	code, body := x.get(x.client, "/auth/corp/start?return_to=/console")
	if code != http.StatusOK || body != "hello ada@example.com" {
		t.Fatalf("sign-in ended with %d %q", code, body)
	}
	u := x.user("ada@example.com")
	if g := x.groups(u.ID); !g["eng"] || !g["ops"] {
		t.Fatalf("groups %v", g)
	}
	if x.audits("auth.login", "success") != 1 {
		t.Fatal("login not audited")
	}
	p.SetUser(map[string]any{"sub": "u-1", "email": "ada@example.com", "email_verified": true, "groups": []string{"eng"}})
	second := x.browser()
	if code, body := x.get(second, "/auth/corp/start"); code != 200 || !strings.Contains(body, "ada@example.com") {
		t.Fatalf("second sign-in %d %q", code, body)
	}
	if g := x.groups(u.ID); g["ops"] {
		t.Fatal("removed IdP group was not removed in Starlight")
	}
}

func TestOIDCStateIsSingleUseAndBrowserBound(t *testing.T) {
	x := newHarness(t, 100)
	p := oidcfake.New("starlight", "secret")
	defer p.Close()
	x.oidcConn(p, nil)
	p.SetUser(map[string]any{"sub": "u-1", "email": "ada@example.com", "email_verified": true})
	cb := x.callbackURL(x.client, "corp")
	if code, _ := x.get(x.browser(), strings.TrimPrefix(cb, x.starlight.URL)); code != http.StatusBadRequest {
		t.Fatalf("callback without the binding cookie = %d", code)
	}
	if code, body := x.get(x.client, strings.TrimPrefix(cb, x.starlight.URL)); code != 200 || !strings.Contains(body, "hello") {
		t.Fatalf("legitimate callback = %d %q", code, body)
	}
	if code, _ := x.get(x.client, strings.TrimPrefix(cb, x.starlight.URL)); code != http.StatusBadRequest {
		t.Fatalf("replayed callback = %d", code)
	}
}

func TestOIDCRefusals(t *testing.T) {
	cases := map[string]struct {
		edit   func(*connections.Input)
		user   map[string]any
		mutate func(map[string]any)
		want   int
		phrase string
	}{
		"nonce mismatch": {user: map[string]any{"email": "a@example.com", "email_verified": true},
			mutate: func(c map[string]any) { c["nonce"] = "forged" }, want: 401, phrase: "could not verify"},
		"jit off": {edit: func(in *connections.Input) { in.JIT = false }, user: map[string]any{"email": "new@example.com", "email_verified": true},
			want: 403, phrase: "not been set up"},
		"unverified jit": {user: map[string]any{"email": "a@example.com", "email_verified": false}, want: 403, phrase: "not verified"},
		"required group": {edit: func(in *connections.Input) { in.Config.Policy.RequiredGroups = []string{"starlight-users"} },
			user: map[string]any{"email": "a@example.com", "email_verified": true, "groups": []string{"eng"}}, want: 403, phrase: "not in a group"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			x := newHarness(t, 100)
			p := oidcfake.New("starlight", "secret")
			defer p.Close()
			x.oidcConn(p, tc.edit)
			p.SetUser(tc.user)
			p.Mutate(tc.mutate)
			code, body := x.get(x.client, "/auth/corp/start")
			if code != tc.want || !strings.Contains(body, tc.phrase) {
				t.Fatalf("got %d, want %d with %q:\n%s", code, tc.want, tc.phrase, body)
			}
			if x.audits("auth.login", "failure") == 0 {
				t.Fatal("refusal not audited")
			}
		})
	}
}

func TestEmailRoutingAndRateLimit(t *testing.T) {
	x := newHarness(t, 2)
	p := oidcfake.New("starlight", "secret")
	defer p.Close()
	x.oidcConn(p, nil)
	stop := *x.client
	stop.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	resp, err := stop.PostForm(x.starlight.URL+"/auth/login", url.Values{"email": {"Ada@Example.com"}, "return_to": {"/console/keys"}})
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	loc := resp.Header.Get("Location")
	if resp.StatusCode != http.StatusSeeOther || !strings.HasPrefix(loc, p.Issuer+"/authorize?") || !strings.Contains(loc, "login_hint=Ada%40Example.com") {
		t.Fatalf("routing = %d %s", resp.StatusCode, loc)
	}
	if code, body := x.post(x.client, "/auth/login", url.Values{"email": {"bo@unknown.example"}}); code != 200 || !strings.Contains(body, "No sign-in method") {
		t.Fatalf("unknown domain = %d", code)
	}
	if code, _ := x.get(x.client, "/auth/corp/start"); code != http.StatusTooManyRequests {
		t.Fatalf("rate limit not enforced: %d", code)
	}
}

func TestBackChannelLogoutRevokesSessions(t *testing.T) {
	x := newHarness(t, 100)
	p := oidcfake.New("starlight", "secret")
	defer p.Close()
	x.oidcConn(p, nil)
	p.SetUser(map[string]any{"sub": "u-1", "email": "ada@example.com", "email_verified": true, "sid": "idp-session-7"})
	if code, _ := x.get(x.client, "/auth/corp/start"); code != 200 {
		t.Fatalf("sign-in = %d", code)
	}
	now := time.Now()
	token := p.Sign(map[string]any{"iss": p.Issuer, "aud": "starlight", "iat": now.Unix(), "exp": now.Add(time.Minute).Unix(), "jti": "logout-1",
		"sid": "idp-session-7", "events": map[string]any{"http://schemas.openid.net/event/backchannel-logout": map[string]any{}}})
	idp := x.browser()
	if code, _ := x.post(idp, "/auth/corp/backchannel-logout", url.Values{"logout_token": {token}}); code != 200 {
		t.Fatalf("logout = %d", code)
	}
	if code, _ := x.get(x.client, "/console"); code != http.StatusUnauthorized {
		t.Fatalf("session survived IdP logout: %d", code)
	}
	if code, _ := x.post(idp, "/auth/corp/backchannel-logout", url.Values{"logout_token": {token}}); code != http.StatusBadRequest {
		t.Fatalf("replayed logout token = %d", code)
	}
	if x.audits("auth.idp_logout", "success") != 1 {
		t.Fatal("IdP logout not audited")
	}
}

func TestLogoutRequiresCSRF(t *testing.T) {
	x := newHarness(t, 100)
	p := oidcfake.New("starlight", "secret")
	defer p.Close()
	x.oidcConn(p, nil)
	p.SetUser(map[string]any{"sub": "u-1", "email": "ada@example.com", "email_verified": true})
	x.get(x.client, "/auth/corp/start")
	if code, _ := x.post(x.client, "/auth/logout", url.Values{"_csrf": {"forged"}}); code != http.StatusForbidden {
		t.Fatalf("forged logout = %d", code)
	}
	s := x.session(x.client)
	if code, body := x.post(x.client, "/auth/logout", url.Values{"_csrf": {s.CSRF()}}); code != 200 || !strings.Contains(body, "signed out") {
		t.Fatalf("logout = %d", code)
	}
	if code, _ := x.get(x.client, "/console"); code != http.StatusUnauthorized {
		t.Fatal("session survived logout")
	}
}

func TestBearerTokensResolveLinkedUsersOnly(t *testing.T) {
	x := newHarness(t, 100)
	p := oidcfake.New("starlight", "secret")
	defer p.Close()
	x.oidcConn(p, func(in *connections.Input) {
		in.Config.OIDC.AcceptBearer, in.Config.OIDC.BearerAudiences = true, []string{"api://starlight"}
	})
	p.SetUser(map[string]any{"sub": "u-1", "email": "ada@example.com", "email_verified": true})
	x.get(x.client, "/auth/corp/start")
	now := time.Now()
	access := func(extra map[string]any) string {
		c := map[string]any{"iss": p.Issuer, "sub": "u-1", "aud": "api://starlight", "iat": now.Unix(), "exp": now.Add(time.Hour).Unix(), "scp": "inference"}
		for k, v := range extra {
			c[k] = v
		}
		return p.Sign(c)
	}
	u, _, err := x.h.BearerUser(context.Background(), access(nil))
	if err != nil || u.Email != "ada@example.com" {
		t.Fatalf("bearer = %+v %v", u, err)
	}
	checks := map[string]struct {
		extra map[string]any
		want  error
	}{
		"wrong audience": {map[string]any{"aud": "starlight"}, oidc.ErrAudience},
		"id token shape": {map[string]any{"scp": nil}, oidc.ErrNotAccessToken},
		"unknown user":   {map[string]any{"sub": "u-2"}, mapping.ErrNotProvisioned},
	}
	for name, c := range checks {
		if _, _, err := x.h.BearerUser(context.Background(), access(c.extra)); !errors.Is(err, c.want) {
			t.Errorf("%s: %v, want %v", name, err, c.want)
		}
	}
}

func TestADisableDuringSignInWins(t *testing.T) {
	x := newHarness(t, 100)
	p := oidcfake.New("starlight", "secret")
	defer p.Close()
	c := x.oidcConn(p, nil)
	p.SetUser(map[string]any{"sub": "u-1", "email": "ada@example.com", "email_verified": true})
	p.Mutate(func(map[string]any) {
		if err := x.h.Connections().SetEnabled(context.Background(), c.ID, false, nil); err != nil {
			t.Error(err)
		}
	})
	if code, body := x.get(x.client, "/auth/corp/start"); code != http.StatusForbidden || !strings.Contains(body, "turned off") {
		t.Fatalf("sign-in finished after its connection was disabled mid-flight: %d %.200s", code, body)
	}
	if code, _ := x.get(x.client, "/console"); code != http.StatusUnauthorized {
		t.Fatalf("a session survived the disable: %d", code)
	}
}
