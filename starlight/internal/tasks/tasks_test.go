package tasks_test

import (
	"context"
	"net/netip"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/authn"
	"github.com/ashhart/TensorFold/starlight/internal/catalog"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/tasks"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

var ip = netip.MustParseAddr("192.0.2.1")

func TestDelegationNeverWidens(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	u, _ := directory.CreateUser(ctx, pool, directory.NewUser{Email: "fay@example.com", Source: "jit"})
	project, coder, planner, dep := ids.New(), ids.New(), ids.New(), ids.New()
	exec := func(sql string, args ...any) {
		if _, err := pool.Exec(ctx, sql, args...); err != nil {
			t.Fatal(err)
		}
	}
	exec("INSERT INTO projects (id, slug, name) VALUES ($1, 'eng', 'Eng')", project)
	exec("INSERT INTO role_bindings (id, role, subject_type, subject_id, project_id) VALUES ($1, 'member', 'user', $2, $3)", ids.New(), u.ID, project)
	exec("INSERT INTO models (id, name, display_name) VALUES ($1, 'coder', 'Coder'), ($2, 'planner', 'Planner')", coder, planner)
	exec("INSERT INTO model_grants (id, model_id) VALUES ($1, $2), ($3, $4)", ids.New(), coder, ids.New(), planner)
	exec(`INSERT INTO deployments (id, model_id, name, spec, revision) VALUES ($1, $2, 'coder', '{"model":{"kind":"path","path":"/m"}}', 3)`, dep, coder)
	tok, hash := secret.Generate(secret.Access)
	exec("INSERT INTO access_tokens (id, token_hash, user_id, client_id, scope, expires_at) VALUES ($1,$2,$3,'c','inference',now()+interval '1 hour')", ids.New(), hash, u.ID)
	auth := &authn.Authenticator{Pool: pool, TTL: time.Millisecond}
	svc := &tasks.Service{Pool: pool, Auth: auth, Catalog: &catalog.Catalog{Pool: pool}}
	user, err := auth.Authenticate(ctx, tok, "", ip)
	if err != nil {
		t.Fatal(err)
	}
	budget := int64(5000)
	root, err := svc.Create(ctx, user, tasks.Spec{Name: "refactor", Models: []string{"coder", "planner"}, MaxConcurrency: 4, MaxDepth: 1,
		BudgetTokens: &budget, PinRevisions: true})
	if err != nil {
		t.Fatal(err)
	}
	if !root.LocalOnly || root.Pins["coder"][0].Revision != 3 || root.Token == "" {
		t.Fatalf("root = %+v", root)
	}
	rootCaller, err := auth.Authenticate(ctx, root.Token, "", ip)
	if err != nil {
		t.Fatal(err)
	}
	child, err := svc.Create(ctx, rootCaller, tasks.Spec{Name: "tests", Models: []string{"coder"}, MaxConcurrency: 99})
	if err != nil {
		t.Fatal(err)
	}
	if child.Depth != 1 || child.MaxConcurrency != 4 || child.RootID != root.ID || len(child.Pins["coder"]) != 1 {
		t.Fatalf("child = %+v", child)
	}
	if _, err := svc.Create(ctx, rootCaller, tasks.Spec{Models: []string{"other"}}); err == nil {
		t.Fatal("child widened its models")
	}
	if _, err := svc.Create(ctx, rootCaller, tasks.Spec{Models: []string{"coder"}, BudgetTokens: &budget}); err == nil {
		t.Fatal("child set its own budget")
	}
	childCaller, err := auth.Authenticate(ctx, child.Token, "", ip)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := svc.Create(ctx, childCaller, tasks.Spec{Models: []string{"coder"}}); err == nil {
		t.Fatal("grandchild exceeded max_depth")
	}
	if n, err := tasks.Revoke(ctx, svc, root.ID); err != nil || n != 2 {
		t.Fatalf("revoke = %d, %v", n, err)
	}
	time.Sleep(5 * time.Millisecond)
	if _, err := auth.Authenticate(ctx, child.Token, "", ip); err == nil {
		t.Fatal("child token outlived its root")
	}
}
