package identitypages_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/console"
	"github.com/ashhart/TensorFold/starlight/internal/console/identitypages"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/identity/login"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
	"github.com/ashhart/TensorFold/starlight/internal/trust"
	"github.com/ashhart/TensorFold/starlight/internal/web"
)

// rig is a console with identity pages on a test server, and signed-in browsers for each role.
type rig struct {
	t        *testing.T
	ctx      context.Context
	pool     *pgxpool.Pool
	srv      *httptest.Server
	c        *console.Console
	sessions *session.Manager
	project  string
}

// browser is one signed-in user.
type browser struct {
	r    *rig
	user directory.User
	ck   *http.Cookie
	csrf string
}

func newRig(t *testing.T) *rig {
	pool := testutil.DB(t)
	ctx := context.Background()
	r := &rig{t: t, ctx: ctx, pool: pool, project: ids.New()}
	keys := keyring.ForTest()
	r.exec("INSERT INTO settings (installation_id, trust_domain, org_name) VALUES ($1, 'starlight-test', 'Example Corp')", ids.New())
	r.exec("INSERT INTO projects (id, slug, name) VALUES ($1, 'research', 'Research')", r.project)
	if err := trust.Bootstrap(ctx, pool, keys, "starlight-test"); err != nil {
		t.Fatal(err)
	}
	tr, err := trust.Load(ctx, pool, keys, ids.New())
	if err != nil {
		t.Fatal(err)
	}
	var handler http.Handler = http.NotFoundHandler()
	r.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) { handler.ServeHTTP(w, req) }))
	t.Cleanup(r.srv.Close)
	r.sessions = session.New(pool, session.Config{})
	proxies, _ := httpx.NewProxies(nil)
	pages, err := web.New(nil)
	if err != nil {
		t.Fatal(err)
	}
	lh, err := login.New(login.Config{PublicURL: r.srv.URL, Pool: pool, Keyring: keys, Sessions: r.sessions, Proxies: proxies, Web: pages,
		HTTP: http.DefaultClient})
	if err != nil {
		t.Fatal(err)
	}
	r.c, err = console.New(&console.Console{Pool: pool, Keys: keys, Sessions: r.sessions, Login: lh, Trust: tr, Proxies: proxies,
		PublicURL: r.srv.URL, NodeURL: "https://starlight.example.com:9443", Version: "test", Org: func(context.Context) string { return "Example Corp" }},
		identitypages.Templates)
	if err != nil {
		t.Fatal(err)
	}
	identitypages.Register(r.c)
	mux := http.NewServeMux()
	r.c.Register(mux)
	handler = httpx.WithRequestID(mux)
	return r
}

func (r *rig) exec(sql string, args ...any) {
	r.t.Helper()
	if _, err := r.pool.Exec(context.Background(), sql, args...); err != nil {
		r.t.Fatalf("%s: %v", sql, err)
	}
}

// user creates a user holding an organisation role (or none) and signs them in.
func (r *rig) user(email, orgRole string) *browser {
	r.t.Helper()
	u, err := directory.CreateUser(r.ctx, r.pool, directory.NewUser{Email: email, DisplayName: strings.Split(email, "@")[0], Source: "jit"})
	if err != nil {
		r.t.Fatal(err)
	}
	if orgRole != "" {
		r.exec("INSERT INTO role_bindings (id, role, subject_type, subject_id) VALUES ($1, $2, 'user', $3)", ids.New(), orgRole, u.ID)
	}
	return r.signIn(u)
}

func (r *rig) signIn(u directory.User) *browser {
	rec := httptest.NewRecorder()
	s, err := r.sessions.Create(r.ctx, rec, httptest.NewRequest("GET", "/", nil), session.Start{UserID: u.ID})
	if err != nil {
		r.t.Fatal(err)
	}
	return &browser{r: r, user: u, ck: rec.Result().Cookies()[0], csrf: s.CSRF()}
}

var noFollow = &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}

func (b *browser) do(method, path string, form url.Values) (int, string) {
	b.r.t.Helper()
	body := strings.NewReader("")
	if form != nil {
		form.Set("_csrf", b.csrf)
		body = strings.NewReader(form.Encode())
	}
	req, _ := http.NewRequest(method, b.r.srv.URL+path, body)
	req.AddCookie(b.ck)
	if form != nil {
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		req.Header.Set("Origin", b.r.srv.URL)
	}
	resp, err := noFollow.Do(req)
	if err != nil {
		b.r.t.Fatal(err)
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(out)
}

func (b *browser) get(path string) (int, string) { return b.do("GET", path, nil) }

func (b *browser) post(path string, form url.Values) (int, string) {
	if form == nil {
		form = url.Values{}
	}
	return b.do("POST", path, form)
}

// audited counts audit events with an action.
func (r *rig) audited(action string) int {
	var n int
	if err := r.pool.QueryRow(r.ctx, "SELECT count(*) FROM audit_events WHERE action = $1", action).Scan(&n); err != nil {
		r.t.Fatal(err)
	}
	return n
}
