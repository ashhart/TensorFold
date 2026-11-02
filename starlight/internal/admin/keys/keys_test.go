package keys_test

import (
	"context"
	"net/netip"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/admin/keys"
	"github.com/ashhart/TensorFold/starlight/internal/authn"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

func TestIssueRotateRevoke(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	u, _ := directory.CreateUser(ctx, pool, directory.NewUser{Email: "jo@example.com", Source: "jit"})
	project := ids.New()
	if _, err := pool.Exec(ctx, "INSERT INTO projects (id, slug, name) VALUES ($1, 'ops', 'Ops')", project); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, "INSERT INTO role_bindings (id, role, subject_type, subject_id, project_id) VALUES ($1, 'member', 'user', $2, $3)", ids.New(), u.ID, project); err != nil {
		t.Fatal(err)
	}
	ip := netip.MustParseAddr("192.0.2.4")
	auth := &authn.Authenticator{Pool: pool, TTL: time.Millisecond}
	tok, k, err := keys.Issue(ctx, pool, keys.Spec{Name: "ci", OwnerType: "user", OwnerID: u.ID, ProjectID: project, Priority: "batch", CIDRs: []string{"192.0.2.0/24"}})
	if err != nil || k.PublicID == "" || len(k.CIDRs) != 1 {
		t.Fatalf("issue %+v %v", k, err)
	}
	if _, err := auth.Authenticate(ctx, tok, "", ip); err != nil {
		t.Fatal(err)
	}
	newTok, nk, err := keys.Rotate(ctx, pool, k.ID, time.Hour, u.ID)
	if err != nil || nk.Priority != "batch" || nk.ID == k.ID {
		t.Fatalf("rotate %+v %v", nk, err)
	}
	for _, tk := range []string{tok, newTok} {
		if _, err := auth.Authenticate(ctx, tk, "", ip); err != nil {
			t.Fatalf("key unusable during grace: %v", err)
		}
	}
	if err := keys.Revoke(ctx, pool, k.ID); err != nil {
		t.Fatal(err)
	}
	time.Sleep(2 * time.Millisecond)
	if _, err := auth.Authenticate(ctx, tok, "", ip); err == nil {
		t.Fatal("revoked key still authenticates")
	}
	live, _ := keys.List(ctx, pool, keys.Filter{ProjectID: project, LiveOnly: true})
	if len(live) != 1 || live[0].ID != nk.ID {
		t.Fatalf("live keys %+v", live)
	}
	for _, bad := range []keys.Spec{{Name: "", OwnerType: "user", OwnerID: u.ID, ProjectID: project}, {Name: "x", OwnerType: "user", OwnerID: u.ID,
		ProjectID: project, CIDRs: []string{"not-a-network"}}, {Name: "x", OwnerType: "robot", OwnerID: u.ID, ProjectID: project}} {
		if _, _, err := keys.Issue(ctx, pool, bad); err == nil {
			t.Errorf("accepted %+v", bad)
		}
	}
}
