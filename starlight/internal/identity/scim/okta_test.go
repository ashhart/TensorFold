package scim

import (
	"context"
	"strings"
	"testing"
)

// TestOktaProvisioningSequence replays the requests Okta's SCIM 2.0 connector sends for a user and a pushed group.
func TestOktaProvisioningSequence(t *testing.T) {
	e := setup(t)
	tok := e.tokenA

	empty := e.must(e.do(tok, "GET", "/scim/v2/Users?filter="+q(`userName eq "dev@example.com"`)+"&startIndex=1&count=100", nil), 200)
	if empty["totalResults"].(float64) != 0 || len(resources(t, empty)) != 0 {
		t.Fatalf("lookup before create: %v", empty)
	}

	created := e.must(e.do(tok, "POST", "/scim/v2/Users", map[string]any{
		"schemas":     []string{URNUser},
		"userName":    "dev@example.com",
		"name":        map[string]any{"givenName": "Dev", "familyName": "Example"},
		"emails":      []any{map[string]any{"primary": true, "value": "dev@example.com", "type": "work"}},
		"displayName": "Dev Example",
		"locale":      "en-US",
		"externalId":  "00u1abcd",
		"groups":      []any{},
		"password":    "never-stored",
		"active":      true,
	}), 201)
	id := created["id"].(string)
	if created["userName"] != "dev@example.com" || created["active"] != true || created["externalId"] != "00u1abcd" {
		t.Fatalf("created = %v", created)
	}
	if _, has := created["password"]; has {
		t.Fatal("password echoed back")
	}
	var stored string
	if err := e.pool.QueryRow(context.Background(), "SELECT scim_attributes::text FROM users WHERE id = $1", id).Scan(&stored); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(stored, "never-stored") {
		t.Fatal("password was stored")
	}

	found := e.must(e.do(tok, "GET", "/scim/v2/Users?filter="+q(`userName eq "DEV@example.com"`), nil), 200)
	if len(resources(t, found)) != 1 || resources(t, found)[0]["id"] != id {
		t.Fatalf("case-insensitive lookup failed: %v", found)
	}

	sessionID, keyID := seedKeyAndSession(t, e.pool, id)
	off := e.must(e.do(tok, "PATCH", "/scim/v2/Users/"+id, map[string]any{
		"schemas":    []string{URNPatchOp},
		"Operations": []any{map[string]any{"op": "replace", "value": map[string]any{"active": false}}},
	}), 200)
	if off["active"] != false {
		t.Fatalf("deactivate: %v", off)
	}
	if s, k := revoked(t, e.pool, sessionID, keyID); !s || !k {
		t.Fatalf("suspension left credentials live: session revoked %v, key revoked %v", s, k)
	}
	on := e.must(e.do(tok, "PATCH", "/scim/v2/Users/"+id, map[string]any{
		"schemas":    []string{URNPatchOp},
		"Operations": []any{map[string]any{"op": "replace", "value": map[string]any{"active": true}}},
	}), 200)
	if on["active"] != true {
		t.Fatalf("reactivate: %v", on)
	}

	put := e.must(e.do(tok, "PUT", "/scim/v2/Users/"+id, map[string]any{
		"schemas":  []string{URNUser},
		"id":       id,
		"userName": "dev@example.com",
		"name":     map[string]any{"givenName": "Devon", "familyName": "Example"},
		"emails":   []any{map[string]any{"primary": true, "value": "devon@example.com", "type": "work"}},
		"active":   true,
	}), 200)
	if put["name"].(map[string]any)["givenName"] != "Devon" || put["displayName"] != nil {
		t.Fatalf("put replaced badly: %v", put)
	}
	var email, display string
	if err := e.pool.QueryRow(context.Background(), "SELECT email, display_name FROM users WHERE id = $1", id).Scan(&email, &display); err != nil {
		t.Fatal(err)
	}
	if email != "devon@example.com" || display != "Devon Example" {
		t.Fatalf("columns = %q %q", email, display)
	}

	group := e.must(e.do(tok, "POST", "/scim/v2/Groups", map[string]any{"schemas": []string{URNGroup}, "displayName": "Engineering", "members": []any{}}), 201)
	gid := group["id"].(string)
	if r := e.do(tok, "PATCH", "/scim/v2/Groups/"+gid, map[string]any{"schemas": []string{URNPatchOp}, "Operations": []any{
		map[string]any{"op": "add", "path": "members", "value": []any{map[string]any{"value": id, "display": "dev@example.com"}}},
	}}); r.status != 204 {
		t.Fatalf("add member: %d %v", r.status, r.body)
	}
	g := e.must(e.do(tok, "GET", "/scim/v2/Groups/"+gid, nil), 200)
	if ms := g["members"].([]any); len(ms) != 1 || ms[0].(map[string]any)["value"] != id {
		t.Fatalf("members = %v", g["members"])
	}
	user := e.must(e.do(tok, "GET", "/scim/v2/Users/"+id, nil), 200)
	if gs := user["groups"].([]any); len(gs) != 1 || gs[0].(map[string]any)["display"] != "Engineering" {
		t.Fatalf("user groups = %v", user["groups"])
	}
	if r := e.do(tok, "PATCH", "/scim/v2/Groups/"+gid, map[string]any{"schemas": []string{URNPatchOp}, "Operations": []any{
		map[string]any{"op": "remove", "path": `members[value eq "` + id + `"]`},
	}}); r.status != 204 {
		t.Fatalf("remove member: %d %v", r.status, r.body)
	}
	if r := e.do(tok, "PATCH", "/scim/v2/Groups/"+gid, map[string]any{"schemas": []string{URNPatchOp}, "Operations": []any{
		map[string]any{"op": "replace", "value": map[string]any{"id": gid, "displayName": "Platform Engineering"}},
	}}); r.status != 204 {
		t.Fatalf("rename: %d %v", r.status, r.body)
	}
	g = e.must(e.do(tok, "GET", "/scim/v2/Groups/"+gid, nil), 200)
	if g["displayName"] != "Platform Engineering" || len(g["members"].([]any)) != 0 {
		t.Fatalf("group after rename = %v", g)
	}
	e.must(e.do(tok, "DELETE", "/scim/v2/Groups/"+gid, nil), 204)
	e.must(e.do(tok, "GET", "/scim/v2/Groups/"+gid, nil), 404)

	e.must(e.do(tok, "DELETE", "/scim/v2/Users/"+id, nil), 204)
	e.must(e.do(tok, "GET", "/scim/v2/Users/"+id, nil), 404)
	var status string
	if err := e.pool.QueryRow(context.Background(), "SELECT status FROM users WHERE id = $1", id).Scan(&status); err != nil || status != "deactivated" {
		t.Fatalf("status after delete = %q %v", status, err)
	}
	again := e.must(e.do(tok, "POST", "/scim/v2/Users", map[string]any{"userName": "dev@example.com", "emails": []any{map[string]any{"value": "dev@example.com", "primary": true}}}), 201)
	if again["id"] == id {
		t.Fatal("re-created user reused the deleted id")
	}
	assertAudit(t, e, "scim.user.create", "scim.user.update", "scim.user.suspend", "scim.user.reactivate", "scim.group.create",
		"scim.group.members", "scim.group.update", "scim.group.delete", "scim.user.delete")
}

func assertAudit(t *testing.T, e *env, actions ...string) {
	t.Helper()
	for _, a := range actions {
		var n int
		if err := e.pool.QueryRow(context.Background(), "SELECT count(*) FROM audit_events WHERE action = $1 AND actor_type = 'scim'", a).Scan(&n); err != nil {
			t.Fatal(err)
		}
		if n == 0 {
			t.Errorf("no audit event %s", a)
		}
	}
}
