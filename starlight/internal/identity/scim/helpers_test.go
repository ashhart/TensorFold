package scim

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

type env struct {
	t      *testing.T
	pool   *pgxpool.Pool
	srv    *httptest.Server
	connA  string
	connB  string
	tokenA string
	tokenB string
}

func setup(t *testing.T, opts ...func(*Options)) *env {
	t.Helper()
	pool := testutil.DB(t)
	e := &env{t: t, pool: pool, connA: insertConnection(t, pool, "okta-main"), connB: insertConnection(t, pool, "entra-main")}
	var err error
	if e.tokenA, _, err = IssueToken(context.Background(), pool, e.connA, "test", nil); err != nil {
		t.Fatal(err)
	}
	if e.tokenB, _, err = IssueToken(context.Background(), pool, e.connB, "test", nil); err != nil {
		t.Fatal(err)
	}
	o := Options{Pool: pool, BaseURL: "https://starlight.example.com"}
	for _, f := range opts {
		f(&o)
	}
	mux := http.NewServeMux()
	New(o).Register(mux)
	e.srv = httptest.NewServer(mux)
	t.Cleanup(e.srv.Close)
	return e
}

func insertConnection(t *testing.T, pool *pgxpool.Pool, slug string) string {
	t.Helper()
	id := ids.New()
	if _, err := pool.Exec(context.Background(), "INSERT INTO idp_connections (id, slug, name, protocol, preset) VALUES ($1, $2, $2, 'oidc', 'generic')", id, slug); err != nil {
		t.Fatal(err)
	}
	return id
}

type reply struct {
	status int
	body   map[string]any
	header http.Header
}

func (e *env) do(token, method, path string, body any) reply {
	e.t.Helper()
	return e.doWith(token, method, path, body, nil)
}

func (e *env) doWith(token, method, path string, body any, headers map[string]string) reply {
	e.t.Helper()
	var rd io.Reader
	switch b := body.(type) {
	case nil:
	case string:
		rd = strings.NewReader(b)
	default:
		raw, err := json.Marshal(b)
		if err != nil {
			e.t.Fatal(err)
		}
		rd = bytes.NewReader(raw)
	}
	req, err := http.NewRequest(method, e.srv.URL+path, rd)
	if err != nil {
		e.t.Fatal(err)
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	req.Header.Set("Content-Type", "application/scim+json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		e.t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	r := reply{status: resp.StatusCode, header: resp.Header}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &r.body); err != nil {
			e.t.Fatalf("%s %s: non-JSON body %q", method, path, raw)
		}
		if ct := resp.Header.Get("Content-Type"); ct != ContentType {
			e.t.Fatalf("%s %s: content type %q", method, path, ct)
		}
	}
	return r
}

func (e *env) must(r reply, status int) map[string]any {
	e.t.Helper()
	if r.status != status {
		e.t.Fatalf("status %d, want %d: %v", r.status, status, r.body)
	}
	return r.body
}

func resources(t *testing.T, body map[string]any) []map[string]any {
	t.Helper()
	raw, _ := body["Resources"].([]any)
	out := make([]map[string]any, 0, len(raw))
	for _, x := range raw {
		out = append(out, x.(map[string]any))
	}
	return out
}

func q(s string) string { return url.QueryEscape(s) }

// seedKeyAndSession gives a user one console session and one API key so revocation can be observed.
func seedKeyAndSession(t *testing.T, pool *pgxpool.Pool, userID string) (sessionID, keyID string) {
	t.Helper()
	ctx := context.Background()
	project := ids.New()
	if _, err := pool.Exec(ctx, "INSERT INTO projects (id, slug, name) VALUES ($1, $2, 'Test')", project, "p-"+project[:8]); err != nil {
		t.Fatal(err)
	}
	keyID = ids.New()
	tok, hash := secret.Generate(secret.APIKey)
	if _, err := pool.Exec(ctx, `INSERT INTO api_keys (id, public_id, secret_hash, name, owner_type, owner_id, project_id)
		VALUES ($1, $2, $3, 'k', 'user', $4, $5)`, keyID, secret.PublicID(tok), hash, userID, project); err != nil {
		t.Fatal(err)
	}
	sessionID = ids.New()
	_, shash := secret.Generate(secret.Session)
	if _, err := pool.Exec(ctx, `INSERT INTO sessions (id, token_hash, user_id, auth_time, csrf_key, idle_expires_at, expires_at)
		VALUES ($1, $2, $3, now(), '\x00', now() + interval '1 hour', now() + interval '2 hours')`, sessionID, shash, userID); err != nil {
		t.Fatal(err)
	}
	return sessionID, keyID
}

func revoked(t *testing.T, pool *pgxpool.Pool, sessionID, keyID string) (sessionRevoked, keyRevoked bool) {
	t.Helper()
	ctx := context.Background()
	if err := pool.QueryRow(ctx, "SELECT revoked_at IS NOT NULL FROM sessions WHERE id = $1", sessionID).Scan(&sessionRevoked); err != nil {
		t.Fatal(err)
	}
	if err := pool.QueryRow(ctx, "SELECT revoked_at IS NOT NULL FROM api_keys WHERE id = $1", keyID).Scan(&keyRevoked); err != nil {
		t.Fatal(err)
	}
	return
}
