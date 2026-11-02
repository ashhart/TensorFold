package gateway

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgconn"

	"github.com/ashhart/TensorFold/starlight/internal/authn"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

func TestWatchdogStopsCallersWhoLoseTheProject(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	exec := func(sql string, args ...any) {
		t.Helper()
		if _, err := pool.Exec(ctx, sql, args...); err != nil {
			t.Fatalf("%s: %v", sql, err)
		}
	}
	u, err := directory.CreateUser(ctx, pool, directory.NewUser{Email: "eve@example.com", Source: "jit"})
	if err != nil {
		t.Fatal(err)
	}
	project, binding := ids.New(), ids.New()
	exec("INSERT INTO projects (id, slug, name) VALUES ($1, 'research', 'Research')", project)
	exec("INSERT INTO role_bindings (id, role, subject_type, subject_id, project_id) VALUES ($1, 'member', 'user', $2, $3)", binding, u.ID, project)
	key, keyHash := secret.Generate(secret.APIKey)
	exec("INSERT INTO api_keys (id, public_id, secret_hash, name, owner_type, owner_id, project_id) VALUES ($1, $2, $3, 'k', 'user', $4, $5)",
		ids.New(), secret.PublicID(key), keyHash, u.ID, project)
	access, accessHash := secret.Generate(secret.Access)
	exec("INSERT INTO access_tokens (id, token_hash, user_id, client_id, scope, expires_at) VALUES ($1, $2, $3, 'c', 'inference', now() + interval '5 minutes')",
		ids.New(), accessHash, u.ID)
	g := &Gateway{Auth: &authn.Authenticator{Pool: pool, TTL: time.Millisecond}}
	ip := netip.MustParseAddr("192.0.2.10")
	calls := map[string]*call{}
	for name, tok := range map[string]string{"api key": key, "access token": access} {
		c, err := g.Auth.Authenticate(ctx, tok, "", ip)
		if err != nil || c.P.ProjectID != project {
			t.Fatalf("%s: %+v %v", name, c, err)
		}
		calls[name] = &call{token: tok, ip: ip, caller: c}
		if g.revoked(ctx, calls[name]) {
			t.Fatalf("%s: a caller in good standing was revoked", name)
		}
	}
	exec("UPDATE role_bindings SET role = 'viewer' WHERE id = $1", binding)
	exec("INSERT INTO projects (id, slug, name, everyone) VALUES ($1, 'commons', 'Commons', true)", ids.New())
	time.Sleep(5 * time.Millisecond)
	for name, c := range calls {
		if !g.revoked(ctx, c) {
			t.Errorf("%s: a stream kept running in a project its caller may no longer use", name)
		}
	}
}

func TestDatabaseOutagesAnswer503(t *testing.T) {
	g := &Gateway{}
	for _, err := range []error{context.DeadlineExceeded, &pgconn.PgError{Code: "57P01"}, &pgconn.PgError{Code: "53300"}} {
		w := httptest.NewRecorder()
		g.refuse(w, httpx.OpenAI, fmt.Errorf("lookup: %w", err))
		if w.Code != http.StatusServiceUnavailable || w.Header().Get("Retry-After") != "1" || !strings.Contains(w.Body.String(), "temporarily_unavailable") {
			t.Errorf("%v answered %d %s", err, w.Code, w.Body.String())
		}
	}
}
