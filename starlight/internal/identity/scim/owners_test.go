package scim

import (
	"context"
	"net/http"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

var suspend = map[string]any{"Operations": []any{map[string]any{"op": "replace", "path": "active", "value": false}}}

func (e *env) bindOwner(subjectType, id string) {
	e.t.Helper()
	if _, err := e.pool.Exec(context.Background(), "INSERT INTO role_bindings (id, role, subject_type, subject_id) VALUES ($1, 'owner', $2, $3)",
		ids.New(), subjectType, id); err != nil {
		e.t.Fatal(err)
	}
}

func (e *env) denied(action string) int {
	e.t.Helper()
	var n int
	if err := e.pool.QueryRow(context.Background(), `SELECT count(*) FROM audit_events WHERE action = $1 AND outcome = 'denied'
		AND actor_type = 'scim' AND details->>'reason' = 'last_owner'`, action).Scan(&n); err != nil {
		e.t.Fatal(err)
	}
	return n
}

func TestLastOwnerIsNeitherSuspendedNorDeleted(t *testing.T) {
	e := setup(t)
	ada := e.must(e.do(e.tokenA, "POST", "/scim/v2/Users", newUser("ada@example.com")), 201)["id"].(string)
	bo := e.must(e.do(e.tokenA, "POST", "/scim/v2/Users", newUser("bo@example.com")), 201)["id"].(string)
	e.bindOwner("user", ada)
	put := newUser("ada@example.com")
	put["active"] = false
	for _, r := range []reply{
		e.do(e.tokenA, "PATCH", "/scim/v2/Users/"+ada, suspend),
		e.do(e.tokenA, "PUT", "/scim/v2/Users/"+ada, put),
		e.do(e.tokenA, "DELETE", "/scim/v2/Users/"+ada, nil),
	} {
		if detail, _ := r.body["detail"].(string); r.status != 409 || r.body["status"] != "409" || !strings.Contains(detail, "make someone else an owner") {
			t.Fatalf("removing the last owner: %d %v", r.status, r.body)
		}
	}
	if got := e.must(e.do(e.tokenA, "GET", "/scim/v2/Users/"+ada, nil), 200); got["active"] != true {
		t.Fatalf("last owner after refusals: %v", got)
	}
	if s, d := e.denied("scim.user.suspend"), e.denied("scim.user.delete"); s != 2 || d != 1 {
		t.Fatalf("refusals audited: %d suspensions, %d deletions", s, d)
	}
	gid := e.must(e.do(e.tokenA, "POST", "/scim/v2/Groups", map[string]any{"displayName": "Starlight owners",
		"members": []any{map[string]any{"value": bo}}}), 201)["id"].(string)
	e.bindOwner("group", gid)
	e.must(e.do(e.tokenA, "PATCH", "/scim/v2/Users/"+ada, suspend), 200)
	if r := e.do(e.tokenA, "DELETE", "/scim/v2/Users/"+bo, nil); r.status != 409 {
		t.Fatalf("deleted the last owner, who holds the role through a group: %d %v", r.status, r.body)
	}
	leave := map[string]any{"Operations": []any{map[string]any{"op": "remove", "path": `members[value eq "` + bo + `"]`}}}
	e.must(e.do(e.tokenA, "PATCH", "/scim/v2/Groups/"+gid, leave), 204)
	e.must(e.do(e.tokenA, "PATCH", "/scim/v2/Users/"+bo, suspend), 200)
}

func TestAdoptingTheLastOwnerAsSuspendedIsRefused(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	jit, err := directory.CreateUser(ctx, e.pool, directory.NewUser{Email: "eve@example.com", Source: "jit"})
	if err != nil {
		t.Fatal(err)
	}
	if err := directory.LinkIdentity(ctx, e.pool, jit.ID, e.connA, "sub-eve", "eve@example.com"); err != nil {
		t.Fatal(err)
	}
	e.bindOwner("user", jit.ID)
	adopt := newUser("eve@example.com")
	adopt["active"] = false
	if r := e.do(e.tokenA, "POST", "/scim/v2/Users", adopt); r.status != 409 || e.denied("scim.user.suspend") != 1 {
		t.Fatalf("adopting the last owner as suspended: %d %v", r.status, r.body)
	}
}

func TestConcurrentSuspensionsKeepAnOwner(t *testing.T) {
	e := setup(t)
	var owners []string
	for _, email := range []string{"cy@example.com", "di@example.com"} {
		id := e.must(e.do(e.tokenA, "POST", "/scim/v2/Users", newUser(email)), 201)["id"].(string)
		e.bindOwner("user", id)
		owners = append(owners, id)
	}
	statuses := make([]int, len(owners))
	var wg sync.WaitGroup
	for i, id := range owners {
		wg.Add(1)
		go func() {
			defer wg.Done()
			req, _ := http.NewRequest("PATCH", e.srv.URL+"/scim/v2/Users/"+id, strings.NewReader(`{"Operations":[{"op":"replace","path":"active","value":false}]}`))
			req.Header.Set("Authorization", "Bearer "+e.tokenA)
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Error(err)
				return
			}
			resp.Body.Close()
			statuses[i] = resp.StatusCode
		}()
	}
	wg.Wait()
	slices.Sort(statuses)
	var active int
	if err := e.pool.QueryRow(context.Background(), "SELECT count(*) FROM users WHERE status = 'active' AND id = ANY($1::uuid[])", owners).Scan(&active); err != nil {
		t.Fatal(err)
	}
	if statuses[0] != 200 || statuses[1] != 409 || active != 1 {
		t.Fatalf("concurrent suspensions answered %v and left %d active owners", statuses, active)
	}
}
