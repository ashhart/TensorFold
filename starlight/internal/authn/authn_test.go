package authn_test

import (
	"context"
	"encoding/base64"
	"errors"
	"net"
	"net/http"
	"net/netip"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/authn"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/identity/oidc"
	"github.com/ashhart/TensorFold/starlight/internal/identity/principal"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/levels"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

type fixture struct {
	pool    *pgxpool.Pool
	user    directory.User
	project string
	other   string
}

func setup(t *testing.T) fixture {
	pool := testutil.DB(t)
	ctx := context.Background()
	u, _ := directory.CreateUser(ctx, pool, directory.NewUser{Email: "eve@example.com", Source: "jit"})
	f := fixture{pool: pool, user: u, project: ids.New(), other: ids.New()}
	exec := func(sql string, args ...any) {
		if _, err := pool.Exec(ctx, sql, args...); err != nil {
			t.Fatal(err)
		}
	}
	exec("INSERT INTO projects (id, slug, name) VALUES ($1, 'research', 'Research'), ($2, 'finance', 'Finance')", f.project, f.other)
	exec("INSERT INTO role_bindings (id, role, subject_type, subject_id, project_id) VALUES ($1, 'member', 'user', $2, $3)", ids.New(), u.ID, f.project)
	return f
}

func (f fixture) key(t *testing.T, ownerType, ownerID string, cidrs []string, class any) string {
	tok, hash := secret.Generate(secret.APIKey)
	if cidrs == nil {
		cidrs = []string{}
	}
	if _, err := f.pool.Exec(context.Background(), `INSERT INTO api_keys (id, public_id, secret_hash, name, owner_type, owner_id, project_id,
		models, priority, classification, allowed_cidrs) VALUES ($1,$2,$3,'k',$4,$5,$6,$7,'batch',$8,$9::cidr[])`,
		ids.New(), secret.PublicID(tok), hash, ownerType, ownerID, f.project, []string{"m1"}, class, cidrs); err != nil {
		t.Fatal(err)
	}
	return tok
}

var anyIP = netip.MustParseAddr("192.0.2.10")

func TestAPIKeyConstraintsAndRevocation(t *testing.T) {
	f := setup(t)
	ctx := context.Background()
	a := &authn.Authenticator{Pool: f.pool, TTL: 50 * time.Millisecond}
	tok := f.key(t, "user", f.user.ID, nil, "confidential")
	c, err := a.Authenticate(ctx, tok, "", anyIP)
	if err != nil {
		t.Fatal(err)
	}
	if c.P.ProjectID != f.project || c.P.KeyID == "" || c.Priority != levels.Batch || c.ClassFloor != levels.Confidential || !c.AllowsModel("m1") || c.AllowsModel("m2") {
		t.Fatalf("caller = %+v %+v", c, c.P)
	}
	if _, err := f.pool.Exec(ctx, "DELETE FROM role_bindings"); err != nil {
		t.Fatal(err)
	}
	time.Sleep(60 * time.Millisecond)
	if _, err := a.Authenticate(ctx, tok, "", anyIP); err == nil {
		t.Fatal("key kept working after its owner left the project")
	}
}

func TestCIDRAllowlist(t *testing.T) {
	f := setup(t)
	a := &authn.Authenticator{Pool: f.pool}
	tok := f.key(t, "user", f.user.ID, []string{"198.51.100.0/24"}, nil)
	if _, err := a.Authenticate(context.Background(), tok, "", anyIP); err == nil {
		t.Fatal("key used outside its CIDR allowlist")
	}
	if _, err := a.Authenticate(context.Background(), tok, "", netip.MustParseAddr("198.51.100.7")); err != nil {
		t.Fatal(err)
	}
}

func TestAccessTokenProjectSelection(t *testing.T) {
	f := setup(t)
	ctx := context.Background()
	tok, hash := secret.Generate(secret.Access)
	if _, err := f.pool.Exec(ctx, "INSERT INTO access_tokens (id, token_hash, user_id, client_id, scope, expires_at) VALUES ($1,$2,$3,'c','inference',now()+interval '5 minutes')",
		ids.New(), hash, f.user.ID); err != nil {
		t.Fatal(err)
	}
	a := &authn.Authenticator{Pool: f.pool}
	c, err := a.Authenticate(ctx, tok, "", anyIP)
	if err != nil || c.P.ProjectID != f.project || c.P.Via != principal.ViaAccess || c.Priority != levels.Interactive {
		t.Fatalf("default project: %+v %v", c, err)
	}
	if _, err := a.Authenticate(ctx, tok, "finance", anyIP); err != authn.ErrProject {
		t.Fatalf("acting in a non-member project: %v", err)
	}
	if c, err := a.Authenticate(ctx, tok, "research", anyIP); err != nil || c.P.ProjectID != f.project {
		t.Fatalf("slug hint: %v", err)
	}
}

func TestTaskTokenRevocationCascades(t *testing.T) {
	f := setup(t)
	ctx := context.Background()
	root, child := ids.New(), ids.New()
	rootTok, rootHash := secret.Generate(secret.Task)
	childTok, childHash := secret.Generate(secret.Task)
	ins := `INSERT INTO tasks (id, root_id, parent_id, ancestors, depth, name, owner_type, owner_id, project_id, token_hash, models, max_depth,
		max_concurrency, budget_tokens, expires_at) VALUES ($1,$2,$3,$4,$5,'t','user',$6,$7,$8,$9,2,4,$10,now()+interval '1 hour')`
	if _, err := f.pool.Exec(ctx, ins, root, root, nil, []string{}, 0, f.user.ID, f.project, rootHash, []string{"m1"}, int64(1000)); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(ctx, ins, child, root, root, []string{root}, 1, f.user.ID, f.project, childHash, []string{"m1"}, nil); err != nil {
		t.Fatal(err)
	}
	a := &authn.Authenticator{Pool: f.pool, TTL: time.Millisecond}
	c, err := a.Authenticate(ctx, childTok, "", anyIP)
	if err != nil || c.Task == nil || c.Task.RootID != root || c.Task.BudgetTokens == nil || *c.Task.BudgetTokens != 1000 || !c.Task.LocalOnly {
		t.Fatalf("child task: %+v %v", c, err)
	}
	if _, err := f.pool.Exec(ctx, "UPDATE tasks SET revoked_at = now() WHERE id = $1", root); err != nil {
		t.Fatal(err)
	}
	time.Sleep(5 * time.Millisecond)
	if _, err := a.Authenticate(ctx, childTok, "", anyIP); err == nil {
		t.Fatal("child task survived its root's revocation")
	}
	if _, err := a.Authenticate(ctx, rootTok, "", anyIP); err == nil {
		t.Fatal("revoked root still authenticates")
	}
}

func TestGarbageRefused(t *testing.T) {
	f := setup(t)
	a := &authn.Authenticator{Pool: f.pool}
	for _, tok := range []string{"", "sk-123", "slr_" + "a", "eyJhbGciOiJub25lIn0.e30."} {
		if _, err := a.Authenticate(context.Background(), tok, "", anyIP); err == nil {
			t.Errorf("accepted %q", tok)
		}
	}
}

// downPool is a pool on a port nothing listens on, as when Postgres is down.
func downPool(t *testing.T) *pgxpool.Pool {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	ln.Close()
	pool, err := pgxpool.New(context.Background(), "postgres://starlight@"+addr+"/x?sslmode=disable&connect_timeout=1")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	return pool
}

func TestJWTWithTheDatabaseDownIsNotInvalid(t *testing.T) {
	tok := "eyJhbGciOiJSUzI1NiJ9." + base64.RawURLEncoding.EncodeToString([]byte(`{"iss":"https://idp.example.com","sub":"u1"}`)) + ".c2ln"
	resolver := func(pool *pgxpool.Pool) authn.JWTResolver {
		bearer := oidc.NewBearer(oidc.NewRP(http.DefaultClient, time.Now), connections.NewStore(pool, nil))
		return func(ctx context.Context, raw string) (string, error) {
			_, _, err := bearer.Verify(ctx, raw)
			return "", err
		}
	}
	down := downPool(t)
	a := &authn.Authenticator{Pool: down, JWT: resolver(down)}
	if _, err := a.Authenticate(context.Background(), tok, "", anyIP); errors.Is(err, authn.ErrInvalid) || !db.Unavailable(err) {
		t.Fatalf("a bearer JWT checked while the database is down: %v", err)
	}
	up := testutil.DB(t)
	a = &authn.Authenticator{Pool: up, JWT: resolver(up)}
	if _, err := a.Authenticate(context.Background(), tok, "", anyIP); !errors.Is(err, authn.ErrInvalid) {
		t.Fatalf("a JWT from an issuer no connection trusts: %v", err)
	}
}

func TestStalledDatabaseAnswersPromptly(t *testing.T) {
	f := setup(t)
	ctx := context.Background()
	small, err := db.Open(ctx, f.pool.Config().ConnString(), db.Options{MaxConns: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer small.Close()
	blocker, err := f.pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer blocker.Rollback(ctx)
	if _, err := blocker.Exec(ctx, "LOCK TABLE api_keys IN ACCESS EXCLUSIVE MODE"); err != nil {
		t.Fatal(err)
	}
	a := &authn.Authenticator{Pool: small, Timeout: 300 * time.Millisecond}
	start, errs := time.Now(), make(chan error, 4)
	for i := 0; i < cap(errs); i++ {
		go func() {
			tok, _ := secret.Generate(secret.APIKey)
			_, err := a.Authenticate(ctx, tok, "", anyIP)
			errs <- err
		}()
	}
	for i := 0; i < cap(errs); i++ {
		if err := <-errs; errors.Is(err, authn.ErrInvalid) || !db.Unavailable(err) {
			t.Errorf("authentication while the database stalls: %v", err)
		}
	}
	if took := time.Since(start); took > 3*time.Second {
		t.Fatalf("authentication waited %v on a stalled database", took)
	}
}

func TestTerminatedConnectionIsUnavailable(t *testing.T) {
	f := setup(t)
	ctx := context.Background()
	one, err := db.Open(ctx, f.pool.Config().ConnString(), db.Options{MaxConns: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer one.Close()
	var pid int
	if err := one.QueryRow(ctx, "SELECT pg_backend_pid()").Scan(&pid); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(ctx, "SELECT pg_terminate_backend($1, 5000)", pid); err != nil {
		t.Fatal(err)
	}
	tok, _ := secret.Generate(secret.APIKey)
	if _, err := (&authn.Authenticator{Pool: one}).Authenticate(ctx, tok, "", anyIP); errors.Is(err, authn.ErrInvalid) || !db.Unavailable(err) {
		t.Fatalf("authentication on a connection the server terminated: %v", err)
	}
}
