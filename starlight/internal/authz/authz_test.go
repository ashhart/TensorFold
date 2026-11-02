package authz_test

import (
	"context"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/authz"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

func TestRoleScopes(t *testing.T) {
	if !authz.ValidRole("owner", false) || authz.ValidRole("owner", true) || !authz.ValidRole("member", true) || authz.ValidRole("member", false) {
		t.Fatal("role scope table is wrong")
	}
	g := authz.Grants{Org: map[string]bool{"auditor": true}}
	if !g.Can(authz.AuditRead, "") || g.Can(authz.SSOWrite, "") || g.Can(authz.Inference, "p1") || !g.Can(authz.ProjectUsage, "p1") {
		t.Fatal("auditor grants are wrong")
	}
	g = authz.Grants{Projects: map[string]map[string]bool{"p1": {"member": true}}}
	if !g.Can(authz.Inference, "p1") || g.Can(authz.Inference, "p2") || g.Can(authz.ProjectApprove, "p1") || g.AnyOrg() {
		t.Fatal("member grants are wrong")
	}
	if g.EveryProject(authz.Inference) || !(authz.Grants{Org: map[string]bool{"owner": true}}).EveryProject(authz.ProjectApprove) ||
		(authz.Grants{Org: map[string]bool{"auditor": true}}).EveryProject(authz.ProjectApprove) {
		t.Fatal("organisation-wide project permissions are wrong")
	}
}

func TestLoadUserThroughGroupsAndEveryone(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	u, _ := directory.CreateUser(ctx, pool, directory.NewUser{Email: "cy@example.com", Source: "jit"})
	gid, err := directory.UpsertGroup(ctx, pool, nil, directory.FromLocal, directory.GroupRef{ExternalID: "eng"})
	if err != nil {
		t.Fatal(err)
	}
	eng, all, archived := ids.New(), ids.New(), ids.New()
	for _, p := range []struct {
		id, slug string
		everyone bool
	}{{eng, "eng", false}, {all, "general", true}, {archived, "old", false}} {
		if _, err := pool.Exec(ctx, "INSERT INTO projects (id, slug, name, everyone) VALUES ($1, $2, $2, $3)", p.id, p.slug, p.everyone); err != nil {
			t.Fatal(err)
		}
	}
	bind := func(role, st, sid string, project any) {
		if _, err := pool.Exec(ctx, "INSERT INTO role_bindings (id, role, subject_type, subject_id, project_id) VALUES ($1,$2,$3,$4,$5)",
			ids.New(), role, st, sid, project); err != nil {
			t.Fatal(err)
		}
	}
	bind("project_admin", "group", gid, eng)
	bind("member", "user", u.ID, archived)
	bind("fleet_operator", "user", u.ID, nil)
	if _, err := pool.Exec(ctx, "UPDATE projects SET archived_at = now() WHERE id = $1", archived); err != nil {
		t.Fatal(err)
	}
	g, err := authz.LoadUser(ctx, pool, u.ID, []string{gid})
	if err != nil {
		t.Fatal(err)
	}
	if !g.Can(authz.ProjectApprove, eng) || !g.Can(authz.Inference, all) || g.Can(authz.Inference, archived) || !g.Can(authz.NodesWrite, "") {
		t.Fatalf("grants = %+v", g)
	}
	without, _ := authz.LoadUser(ctx, pool, u.ID, nil)
	if without.Can(authz.ProjectApprove, eng) {
		t.Fatal("group grant applied without group membership")
	}
}
