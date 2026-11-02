package tasks_test

import (
	"context"
	"net/netip"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/admin/keys"
	"github.com/ashhart/TensorFold/starlight/internal/authn"
	"github.com/ashhart/TensorFold/starlight/internal/catalog"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/levels"
	"github.com/ashhart/TensorFold/starlight/internal/tasks"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

func TestTasksKeepTheirMintingKeysLimits(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	u, _ := directory.CreateUser(ctx, pool, directory.NewUser{Email: "ren@example.com", Source: "jit"})
	project, model := ids.New(), ids.New()
	for _, q := range []struct {
		sql  string
		args []any
	}{
		{"INSERT INTO projects (id, slug, name) VALUES ($1, 'ci', 'CI')", []any{project}},
		{"INSERT INTO role_bindings (id, role, subject_type, subject_id, project_id) VALUES ($1, 'member', 'user', $2, $3)", []any{ids.New(), u.ID, project}},
		{"INSERT INTO models (id, name, display_name) VALUES ($1, 'coder', 'Coder')", []any{model}},
		{"INSERT INTO model_grants (id, model_id) VALUES ($1, $2)", []any{ids.New(), model}},
	} {
		if _, err := pool.Exec(ctx, q.sql, q.args...); err != nil {
			t.Fatal(err)
		}
	}
	expires := time.Now().Add(time.Hour)
	tok, key, err := keys.Issue(ctx, pool, keys.Spec{Name: "ci", OwnerType: "user", OwnerID: u.ID, ProjectID: project, Classification: "restricted",
		CIDRs: []string{"192.0.2.0/24"}, ExpiresAt: &expires, Priority: "batch"})
	if err != nil {
		t.Fatal(err)
	}
	inside, outside := netip.MustParseAddr("192.0.2.9"), netip.MustParseAddr("203.0.113.7")
	auth := &authn.Authenticator{Pool: pool, TTL: time.Millisecond}
	svc := &tasks.Service{Pool: pool, Auth: auth, Catalog: &catalog.Catalog{Pool: pool}}
	c, err := auth.Authenticate(ctx, tok, "", inside)
	if err != nil {
		t.Fatal(err)
	}
	local := false
	task, err := svc.Create(ctx, c, tasks.Spec{Models: []string{"coder"}, LocalOnly: &local, TTLSeconds: 7 * 24 * 3600})
	if err != nil {
		t.Fatal(err)
	}
	if task.ExpiresAt.After(expires.Add(time.Second)) {
		t.Fatalf("task outlives its key: %s > %s", task.ExpiresAt, expires)
	}
	if _, err := auth.Authenticate(ctx, task.Token, "", outside); err == nil {
		t.Fatal("task token used outside the minting key's network allowlist")
	}
	tc, err := auth.Authenticate(ctx, task.Token, "", inside)
	if err != nil || tc.ClassFloor != levels.Restricted || tc.P.KeyID != key.ID || tc.MaxPriority != levels.Batch {
		t.Fatalf("task caller lost the key's limits: %+v %v", tc, err)
	}
	if err := keys.Revoke(ctx, pool, key.ID); err != nil {
		t.Fatal(err)
	}
	time.Sleep(2 * time.Millisecond)
	if _, err := auth.Authenticate(ctx, task.Token, "", inside); err == nil {
		t.Fatal("task token outlived its revoked key")
	}
}
