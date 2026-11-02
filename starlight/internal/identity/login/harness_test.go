package login

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/cookiejar"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/identity/oidc/oidcfake"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
	"github.com/ashhart/TensorFold/starlight/internal/web"
)

// harness runs the login handlers on a TLS server with a browser-like client.
type harness struct {
	t         *testing.T
	pool      *pgxpool.Pool
	h         *Handler
	starlight *httptest.Server
	client    *http.Client
	sessions  *session.Manager
}

func newHarness(t *testing.T, perMinute int) *harness {
	t.Helper()
	pool := testutil.DB(t)
	var handler http.Handler
	starlight := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { handler.ServeHTTP(w, r) }))
	starlight.StartTLS()
	t.Cleanup(starlight.Close)
	sessions := session.New(pool, session.Config{Secure: true})
	renderer, err := web.New(nil)
	if err != nil {
		t.Fatal(err)
	}
	h, err := New(Config{PublicURL: starlight.URL, Pool: pool, Keyring: keyring.ForTest(), Sessions: sessions, Web: renderer, HTTP: starlight.Client(),
		StartsPerMinute: perMinute})
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	h.Register(mux)
	mux.Handle("GET /static/", web.StaticHandler())
	mux.HandleFunc("GET /console", func(w http.ResponseWriter, r *http.Request) {
		_, u, err := sessions.Load(r.Context(), r)
		if err != nil {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		fmt.Fprint(w, "hello "+u.Email)
	})
	handler = httpx.WithRequestID(mux)
	x := &harness{t: t, pool: pool, h: h, starlight: starlight, sessions: sessions}
	x.client = x.browser()
	return x
}

// browser returns a fresh client with its own cookie jar.
func (x *harness) browser() *http.Client {
	jar, _ := cookiejar.New(nil)
	return &http.Client{Transport: x.starlight.Client().Transport, Jar: jar}
}

func (x *harness) oidcConn(p *oidcfake.Provider, edit func(*connections.Input)) connections.Connection {
	x.t.Helper()
	in := connections.Input{Slug: "corp", Name: "Corp SSO", Protocol: connections.OIDC, Preset: "generic_oidc", Domains: []string{"example.com"},
		Enabled: true, JIT: true,
		Config: connections.Config{OIDC: &connections.OIDCConfig{Issuer: p.Issuer, ClientID: p.ClientID, Scopes: []string{"openid", "email"}},
			Claims: connections.ClaimMap{Groups: "groups"}},
		Secrets: connections.Secrets{ClientSecret: p.ClientSecret}}
	if edit != nil {
		edit(&in)
	}
	c, err := x.h.Connections().Create(context.Background(), in, nil)
	if err != nil {
		x.t.Fatal(err)
	}
	return c
}

// get fetches path with the browser, following redirects, and returns status and body.
func (x *harness) get(c *http.Client, path string) (int, string) {
	x.t.Helper()
	resp, err := c.Get(x.starlight.URL + path)
	if err != nil {
		x.t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b)
}

// post submits a form with the browser and returns status and body after redirects.
func (x *harness) post(c *http.Client, path string, form url.Values) (int, string) {
	x.t.Helper()
	resp, err := c.PostForm(x.starlight.URL+path, form)
	if err != nil {
		x.t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b)
}

// callbackURL runs a sign-in up to the IdP's redirect back to Starlight and returns that URL without visiting it.
func (x *harness) callbackURL(c *http.Client, slug string) string {
	x.t.Helper()
	var captured string
	stop := *c
	stop.CheckRedirect = func(req *http.Request, _ []*http.Request) error {
		if strings.Contains(req.URL.Path, "/callback") {
			captured = req.URL.String()
			return http.ErrUseLastResponse
		}
		return nil
	}
	resp, err := stop.Get(x.starlight.URL + "/auth/" + slug + "/start")
	if err != nil {
		x.t.Fatal(err)
	}
	resp.Body.Close()
	if captured == "" {
		x.t.Fatalf("sign-in never reached the callback (status %d)", resp.StatusCode)
	}
	return captured
}

func (x *harness) user(email string) directory.User {
	x.t.Helper()
	u, err := directory.FindByEmail(context.Background(), x.pool, email)
	if err != nil {
		x.t.Fatalf("user %s: %v", email, err)
	}
	return u
}

func (x *harness) groups(userID string) map[string]bool {
	x.t.Helper()
	gs, err := directory.UserGroups(context.Background(), x.pool, userID)
	if err != nil {
		x.t.Fatal(err)
	}
	out := map[string]bool{}
	for _, g := range gs {
		out[g.ExternalID] = true
	}
	return out
}

// session loads the browser's current session for CSRF-protected posts.
func (x *harness) session(c *http.Client) session.Session {
	x.t.Helper()
	req := httptest.NewRequest("GET", x.starlight.URL+"/console", nil)
	u, _ := url.Parse(x.starlight.URL)
	for _, ck := range c.Jar.Cookies(u) {
		req.AddCookie(ck)
	}
	s, _, err := x.sessions.Load(context.Background(), req)
	if err != nil {
		x.t.Fatalf("no session: %v", err)
	}
	return s
}

func (x *harness) audits(action, outcome string) int {
	x.t.Helper()
	var n int
	if err := x.pool.QueryRow(context.Background(), "SELECT count(*) FROM audit_events WHERE action = $1 AND outcome = $2", action, outcome).Scan(&n); err != nil {
		x.t.Fatal(err)
	}
	return n
}
