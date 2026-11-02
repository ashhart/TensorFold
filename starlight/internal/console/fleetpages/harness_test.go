package fleetpages_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/authn"
	"github.com/ashhart/TensorFold/starlight/internal/console"
	"github.com/ashhart/TensorFold/starlight/internal/console/fleetpages"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
	"github.com/ashhart/TensorFold/starlight/internal/trust"
)

// rig is a console with the fleet pages on a test server, seeded with one model, deployment, two nodes and a project.
type rig struct {
	t        *testing.T
	ctx      context.Context
	pool     *pgxpool.Pool
	keys     *keyring.Keyring
	srv      *httptest.Server
	c        *console.Console
	sessions *session.Manager
	plans    *fakePlans
	project  string
	model    string
	node     string
	replica  string
}

type browser struct {
	r    *rig
	user directory.User
	ck   *http.Cookie
	csrf string
}

// fakePlans stands in for the plans service: it approves only with the stored hash, as the real one does.
type fakePlans struct {
	pool  *pgxpool.Pool
	calls []string
}

func (f *fakePlans) Approve(r *http.Request, c *authn.Caller, id, sum string) error {
	f.calls = append(f.calls, "approve:"+c.P.UserID)
	tag, err := f.pool.Exec(r.Context(), `UPDATE plans SET status = 'approved', approved_by = $3, approved_at = now()
		WHERE id = $1 AND status = 'pending_approval' AND encode(plan_sha256, 'hex') = $2`, id, sum, c.P.UserID)
	if err == nil && tag.RowsAffected() == 0 {
		return httpx.Errorf(http.StatusConflict, "plan_changed", "approve with the plan_sha256 of the plan you reviewed, while it is pending")
	}
	return err
}

func (f *fakePlans) Reject(r *http.Request, c *authn.Caller, id string) error {
	f.calls = append(f.calls, "reject:"+c.P.UserID)
	_, err := f.pool.Exec(r.Context(), "UPDATE plans SET status = 'rejected' WHERE id = $1 AND status = 'pending_approval'", id)
	return err
}

func newRig(t *testing.T) *rig {
	pool := testutil.DB(t)
	ctx := context.Background()
	r := &rig{t: t, ctx: ctx, pool: pool, keys: keyring.ForTest(), project: ids.New(), model: ids.New(), node: ids.New(), replica: ids.New()}
	r.exec("INSERT INTO settings (installation_id, trust_domain, org_name, offline) VALUES ($1, 'starlight-test', 'Example Corp', false)", ids.New())
	r.exec("INSERT INTO projects (id, slug, name) VALUES ($1, 'research', 'Research')", r.project)
	r.exec("INSERT INTO models (id, name, display_name) VALUES ($1, 'team/coder', 'Coder')", r.model)
	deployment := ids.New()
	r.exec(`INSERT INTO deployments (id, model_id, name, spec, pool) VALUES ($1, $2, 'coder-mac', '{"model":{"kind":"path","path":"/srv/coder"}}', 'mac')`,
		deployment, r.model)
	r.exec(`INSERT INTO nodes (id, name, status, pool, last_seen_at, facts) VALUES ($1, 'studio-1', 'active', 'mac', now(),
		'{"backend":"metal","memory_bytes":274877906944,"gpus":[{"name":"Apple M3 Ultra"}]}'), ($2, 'studio-2', 'pending', 'mac', now(), '{}')`,
		r.node, ids.New())
	r.exec(`INSERT INTO replicas (id, deployment_id, deployment_revision, node_id, state, slots, epoch, live, live_at) VALUES ($1, $2, 1, $3,
		'ready', 4, 1, '{"tokens_per_second": 87.5, "prompt_tokens_per_second": 1200, "running": 2, "waiting": 1, "kv_usage": 0.42}', now())`,
		r.replica, deployment, r.node)
	if err := trust.Bootstrap(ctx, pool, r.keys, "starlight-test"); err != nil {
		t.Fatal(err)
	}
	tr, err := trust.Load(ctx, pool, r.keys, ids.New())
	if err != nil {
		t.Fatal(err)
	}
	var handler http.Handler = http.NotFoundHandler()
	r.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) { handler.ServeHTTP(w, req) }))
	t.Cleanup(r.srv.Close)
	r.sessions = session.New(pool, session.Config{})
	proxies, _ := httpx.NewProxies(nil)
	r.c, err = console.New(&console.Console{Pool: pool, Keys: r.keys, Sessions: r.sessions, Trust: tr, Proxies: proxies, PublicURL: r.srv.URL,
		NodeURL: "https://starlight.example.com:9443", Version: "test", Org: func(context.Context) string { return "Example Corp" }}, fleetpages.Templates)
	if err != nil {
		t.Fatal(err)
	}
	r.plans = &fakePlans{pool: pool}
	fleetpages.Register(r.c, fleetpages.WithPlans(r.plans))
	mux := http.NewServeMux()
	r.c.Register(mux)
	handler = httpx.WithRequestID(mux)
	return r
}

func (r *rig) exec(sql string, args ...any) {
	r.t.Helper()
	if _, err := r.pool.Exec(r.ctx, sql, args...); err != nil {
		r.t.Fatalf("%s: %v", sql, err)
	}
}

func (r *rig) scalar(sql string, args ...any) string {
	r.t.Helper()
	var v string
	if err := r.pool.QueryRow(r.ctx, sql, args...).Scan(&v); err != nil {
		r.t.Fatalf("%s: %v", sql, err)
	}
	return v
}

// user creates a user with an organisation role, or a project role when projectRole is set, and signs them in.
func (r *rig) user(email, orgRole, projectRole string) *browser {
	r.t.Helper()
	u, err := directory.CreateUser(r.ctx, r.pool, directory.NewUser{Email: email, DisplayName: strings.Split(email, "@")[0], Source: "jit"})
	if err != nil {
		r.t.Fatal(err)
	}
	if orgRole != "" {
		r.exec("INSERT INTO role_bindings (id, role, subject_type, subject_id) VALUES ($1, $2, 'user', $3)", ids.New(), orgRole, u.ID)
	}
	if projectRole != "" {
		r.exec("INSERT INTO role_bindings (id, role, subject_type, subject_id, project_id) VALUES ($1, $2, 'user', $3, $4)",
			ids.New(), projectRole, u.ID, r.project)
	}
	rec := httptest.NewRecorder()
	s, err := r.sessions.Create(r.ctx, rec, httptest.NewRequest("GET", "/", nil), session.Start{UserID: u.ID})
	if err != nil {
		r.t.Fatal(err)
	}
	return &browser{r: r, user: u, ck: rec.Result().Cookies()[0], csrf: s.CSRF()}
}

var noFollow = &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}

func (b *browser) do(method, path string, form url.Values, extra ...*http.Cookie) (*http.Response, string) {
	b.r.t.Helper()
	body := strings.NewReader("")
	if form != nil {
		form.Set("_csrf", b.csrf)
		body = strings.NewReader(form.Encode())
	}
	req, _ := http.NewRequest(method, b.r.srv.URL+path, body)
	for _, c := range append([]*http.Cookie{b.ck}, extra...) {
		req.AddCookie(c)
	}
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
	return resp, string(out)
}

func (b *browser) get(path string) (int, string) {
	resp, body := b.do("GET", path, nil)
	return resp.StatusCode, body
}

func (b *browser) post(path string, form url.Values) (int, string) {
	if form == nil {
		form = url.Values{}
	}
	resp, body := b.do("POST", path, form)
	return resp.StatusCode, body
}

// follow posts a form and returns the status and the page the redirect leads to, carrying its one-time notice.
func (b *browser) follow(path string, form url.Values) (int, string) {
	b.r.t.Helper()
	resp, body := b.do("POST", path, form)
	if resp.StatusCode != http.StatusSeeOther {
		return resp.StatusCode, body
	}
	_, page := b.do("GET", resp.Header.Get("Location"), nil, resp.Cookies()...)
	return resp.StatusCode, page
}

func (r *rig) audited(action string) int {
	r.t.Helper()
	var n int
	if err := r.pool.QueryRow(r.ctx, "SELECT count(*) FROM audit_events WHERE action = $1", action).Scan(&n); err != nil {
		r.t.Fatal(err)
	}
	return n
}

// plan stores a pending plan in the rig's project, sealed as the plans service seals it.
func (r *rig) plan(owner directory.User, text string) (id, sum string) {
	r.t.Helper()
	id = ids.New()
	sealed, err := r.keys.Seal([]byte(text), "plans.plan:"+id)
	if err != nil {
		r.t.Fatal(err)
	}
	digest := sha256.Sum256([]byte(text))
	r.exec(`INSERT INTO plans (id, project_id, owner_type, owner_id, title, planner_model, executor_model, egress_sha256, egress_bytes,
		plan_sealed, plan_sha256, status) VALUES ($1, $2, 'user', $3, 'Port the parser', 'frontier/planner', 'team/coder', $4, 2048, $5, $6,
		'pending_approval')`, id, r.project, owner.ID, digest[:], sealed, digest[:])
	return id, hex.EncodeToString(digest[:])
}
