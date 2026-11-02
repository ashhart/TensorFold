package scim

import (
	"context"
	"testing"
)

// TestEntraProvisioningSequence replays Microsoft Entra ID's provisioning requests, including its non-compliant shapes.
func TestEntraProvisioningSequence(t *testing.T) {
	e := setup(t)
	tok := e.tokenB

	probe := e.must(e.do(tok, "GET", "/scim/v2/Users?filter="+q(`userName eq "8f1c2f6e-5a77-4b61-9b8e-3d2f1b7a9c10"`), nil), 200)
	if probe["totalResults"].(float64) != 0 {
		t.Fatalf("test connection probe: %v", probe)
	}
	created := e.must(e.do(tok, "POST", "/scim/v2/Users", map[string]any{
		"schemas":     []string{URNUser, URNEnterprise},
		"externalId":  "alice",
		"userName":    "alice@example.com",
		"active":      true,
		"displayName": "Alice Example",
		"emails":      []any{map[string]any{"primary": true, "type": "work", "value": "alice@example.com"}},
		"name":        map[string]any{"formatted": "Alice Example", "familyName": "Example", "givenName": "Alice"},
		"title":       "Engineer",
		"roles":       []any{map[string]any{"primary": "False", "type": "WindowsAzureActiveDirectoryRole", "value": "Reader"}},
		URNEnterprise: map[string]any{"employeeNumber": "1001", "department": "Research", "manager": "manager-id"},
	}), 201)
	id := created["id"].(string)
	ext := created[URNEnterprise].(map[string]any)
	if ext["department"] != "Research" || ext["manager"].(map[string]any)["value"] != "manager-id" {
		t.Fatalf("enterprise extension = %v", ext)
	}
	if schemas := created["schemas"].([]any); len(schemas) != 2 {
		t.Fatalf("schemas = %v", schemas)
	}

	e.must(e.do(tok, "PATCH", "/scim/v2/Users/"+id, map[string]any{
		"schemas": []string{URNPatchOp},
		"Operations": []any{
			map[string]any{"op": "Replace", "path": `emails[type eq "work"].value`, "value": "alice.example@example.com"},
			map[string]any{"op": "Replace", "path": "name.familyName", "value": "Sample"},
			map[string]any{"op": "Add", "path": `phoneNumbers[type eq "mobile"].value`, "value": "+1 555 0100"},
			map[string]any{"op": "Replace", "path": URNEnterprise + ":department", "value": "Platform"},
		},
	}), 200)
	got := e.must(e.do(tok, "GET", "/scim/v2/Users/"+id, nil), 200)
	email := got["emails"].([]any)[0].(map[string]any)["value"]
	phone := got["phoneNumbers"].([]any)[0].(map[string]any)
	if email != "alice.example@example.com" || got["name"].(map[string]any)["familyName"] != "Sample" || phone["type"] != "mobile" ||
		got[URNEnterprise].(map[string]any)["department"] != "Platform" {
		t.Fatalf("after patch = %v", got)
	}
	var column string
	_ = e.pool.QueryRow(context.Background(), "SELECT email FROM users WHERE id = $1", id).Scan(&column)
	if column != "alice.example@example.com" {
		t.Fatalf("email column = %q", column)
	}

	sessionID, keyID := seedKeyAndSession(t, e.pool, id)
	off := e.must(e.do(tok, "PATCH", "/scim/v2/Users/"+id, map[string]any{"schemas": []string{URNPatchOp},
		"Operations": []any{map[string]any{"op": "Replace", "path": "active", "value": "False"}}}), 200)
	if off["active"] != false {
		t.Fatalf("string False not honoured: %v", off)
	}
	if s, k := revoked(t, e.pool, sessionID, keyID); !s || !k {
		t.Fatal("suspension left credentials live")
	}
	e.must(e.do(tok, "PATCH", "/scim/v2/Users/"+id, map[string]any{"schemas": []string{URNPatchOp},
		"Operations": []any{map[string]any{"op": "Replace", "path": "active", "value": "True"}}}), 200)

	pathless := e.must(e.do(tok, "PATCH", "/scim/v2/Users/"+id, map[string]any{"schemas": []string{URNPatchOp},
		"Operations": []any{map[string]any{"op": "Replace", "value": map[string]any{
			"name.givenName": "Ally", URNEnterprise + ":employeeNumber": "2002", "urn:ietf:params:scim:schemas:core:2.0:User:title": "Lead"}}}}), 200)
	if pathless["name"].(map[string]any)["givenName"] != "Ally" || pathless["title"] != "Lead" ||
		pathless[URNEnterprise].(map[string]any)["employeeNumber"] != "2002" || pathless["active"] != true {
		t.Fatalf("path-less dotted and URN keys = %v", pathless)
	}

	group := e.must(e.do(tok, "POST", "/scim/v2/Groups", map[string]any{"schemas": []string{URNGroup},
		"externalId": "7f2d9c1e-0000-4000-8000-000000000001", "displayName": "Sales", "members": []any{}}), 201)
	gid := group["id"].(string)
	lookup := e.must(e.do(tok, "GET", "/scim/v2/Groups?filter="+q(`displayName eq "sales"`)+"&excludedAttributes=members", nil), 200)
	items := resources(t, lookup)
	if len(items) != 1 || items[0]["externalId"] != "7f2d9c1e-0000-4000-8000-000000000001" {
		t.Fatalf("group lookup = %v", lookup)
	}
	if _, has := items[0]["members"]; has {
		t.Fatal("excludedAttributes=members still returned members")
	}
	e.must(e.do(tok, "PATCH", "/scim/v2/Groups/"+gid, map[string]any{"schemas": []string{URNPatchOp}, "Operations": []any{
		map[string]any{"op": "Add", "path": "members", "value": []any{map[string]any{"$ref": nil, "value": id}}}}}), 204)
	member := e.must(e.do(tok, "GET", "/scim/v2/Groups?filter="+q(`id eq "`+gid+`" and members eq "`+id+`"`)+"&excludedAttributes=members", nil), 200)
	if member["totalResults"].(float64) != 1 {
		t.Fatalf("membership check = %v", member)
	}
	e.must(e.do(tok, "PATCH", "/scim/v2/Groups/"+gid, map[string]any{"schemas": []string{URNPatchOp}, "Operations": []any{
		map[string]any{"op": "Remove", "path": "members", "value": []any{map[string]any{"$ref": nil, "value": id}}},
		map[string]any{"op": "Replace", "path": "displayName", "value": "Sales Team"}}}), 204)
	g := e.must(e.do(tok, "GET", "/scim/v2/Groups/"+gid, nil), 200)
	if g["displayName"] != "Sales Team" || len(g["members"].([]any)) != 0 {
		t.Fatalf("group = %v", g)
	}
	e.must(e.do(tok, "DELETE", "/scim/v2/Users/"+id, nil), 204)
	assertAudit(t, e, "scim.user.create", "scim.user.update", "scim.user.suspend", "scim.user.reactivate", "scim.group.members", "scim.user.delete")
}
