package people_test

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/admin/people"
	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

func TestLastOwnerSurvivesConcurrentRemovals(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	direct, _ := directory.CreateUser(ctx, pool, directory.NewUser{Email: "eli@example.com", Source: "jit"})
	viaGroup, _ := directory.CreateUser(ctx, pool, directory.NewUser{Email: "fen@example.com", Source: "jit"})
	gid, err := directory.UpsertGroup(ctx, pool, nil, directory.FromLocal, directory.GroupRef{ExternalID: "admins"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, "INSERT INTO group_members (group_id, user_id) VALUES ($1, $2)", gid, viaGroup.ID); err != nil {
		t.Fatal(err)
	}
	ev := audit.Actor(nil)
	b1, err := people.Grant(ctx, pool, ev, "owner", "user", direct.ID, "")
	if err != nil {
		t.Fatal(err)
	}
	b2, err := people.Grant(ctx, pool, ev, "owner", "group", gid, "")
	if err != nil {
		t.Fatal(err)
	}
	if n, _ := people.ActiveOwners(ctx, pool); n != 2 {
		t.Fatalf("owners = %d", n)
	}
	if ok, _ := people.IsOwner(ctx, pool, viaGroup.ID); !ok {
		t.Fatal("group ownership not recognised")
	}
	var wg sync.WaitGroup
	errs := make([]error, 2)
	for i, b := range []string{b1, b2} {
		wg.Add(1)
		go func(i int, b string) {
			defer wg.Done()
			errs[i] = people.Revoke(ctx, pool, ev, b)
		}(i, b)
	}
	wg.Wait()
	refused := 0
	for _, err := range errs {
		if errors.Is(err, people.ErrLastOwner) {
			refused++
		} else if err != nil {
			t.Fatal(err)
		}
	}
	if n, _ := people.ActiveOwners(ctx, pool); refused != 1 || n != 1 {
		t.Fatalf("refused %d, owners left %d", refused, n)
	}
	remaining := direct.ID
	if ok, _ := people.IsOwner(ctx, pool, viaGroup.ID); ok {
		remaining = viaGroup.ID
	}
	if err := people.SetStatus(ctx, pool, ev, remaining, directory.Suspended); !errors.Is(err, people.ErrLastOwner) {
		t.Fatalf("suspending the last owner: %v", err)
	}
	if _, err := people.Grant(ctx, pool, ev, "member", "user", direct.ID, ""); !errors.Is(err, people.ErrRole) {
		t.Fatalf("project role at org scope: %v", err)
	}
	if _, err := people.Grant(ctx, pool, ev, "auditor", "user", ids.New(), ""); !errors.Is(err, people.ErrNotFound) {
		t.Fatalf("grant to a missing user: %v", err)
	}
}

func TestOwnersCountOnceWhetherDirectOrThroughGroup(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	ev := audit.Actor(nil)
	user := func(email, status string) string {
		u, err := directory.CreateUser(ctx, pool, directory.NewUser{Email: email, Source: "jit", Status: status})
		if err != nil {
			t.Fatal(err)
		}
		return u.ID
	}
	both, member, away := user("gil@example.com", ""), user("hal@example.com", ""), user("ida@example.com", directory.Suspended)
	gid, err := directory.UpsertGroup(ctx, pool, nil, directory.FromLocal, directory.GroupRef{ExternalID: "owners"})
	if err != nil {
		t.Fatal(err)
	}
	join := func(userID string) {
		if _, err := pool.Exec(ctx, "INSERT INTO group_members (group_id, user_id) VALUES ($1, $2)", gid, userID); err != nil {
			t.Fatal(err)
		}
	}
	join(both)
	join(away)
	direct, err := people.Grant(ctx, pool, ev, "owner", "user", both, "")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := people.Grant(ctx, pool, ev, "owner", "user", away, ""); err != nil {
		t.Fatal(err)
	}
	group, err := people.Grant(ctx, pool, ev, "owner", "group", gid, "")
	if err != nil {
		t.Fatal(err)
	}
	if n, _ := people.ActiveOwners(ctx, pool); n != 1 {
		t.Fatalf("owners = %d, want gil once and not the suspended ida", n)
	}
	for _, status := range []string{directory.Suspended, directory.Deactivated} {
		if err := people.SetStatus(ctx, pool, ev, both, status); !errors.Is(err, people.ErrLastOwner) {
			t.Fatalf("%s the only owner, who holds the role twice: %v", status, err)
		}
	}
	join(member)
	if err := people.Revoke(ctx, pool, ev, direct); err != nil {
		t.Fatalf("revoking a direct role the group also grants: %v", err)
	}
	if err := people.SetStatus(ctx, pool, ev, both, directory.Suspended); err != nil {
		t.Fatalf("suspending one of two owners: %v", err)
	}
	if err := people.Revoke(ctx, pool, ev, group); !errors.Is(err, people.ErrLastOwner) {
		t.Fatalf("revoking the group of the last owner: %v", err)
	}
	if err := people.SetStatus(ctx, pool, ev, member, directory.Deactivated); !errors.Is(err, people.ErrLastOwner) {
		t.Fatalf("deactivating the last owner, who holds the role through a group: %v", err)
	}
	if n, _ := people.ActiveOwners(ctx, pool); n != 1 {
		t.Fatalf("owners = %d after refusals", n)
	}
}
