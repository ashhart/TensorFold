package scim

import (
	"context"
	"testing"
)

func TestCannotProvisionIntoAnotherConnectionsDomain(t *testing.T) {
	e := setup(t)
	if _, err := e.pool.Exec(context.Background(), "UPDATE idp_connections SET domains = '{corp.example}' WHERE id = $1", e.connA); err != nil {
		t.Fatal(err)
	}
	user := func(email string) map[string]any {
		return map[string]any{"schemas": []string{"urn:ietf:params:scim:schemas:core:2.0:User"}, "userName": email,
			"emails": []any{map[string]any{"value": email, "primary": true}}, "active": true}
	}
	if r := e.do(e.tokenB, "POST", "/scim/v2/Users", user("ceo@corp.example")); r.status != 400 {
		t.Fatalf("connection B created a user in connection A's domain: %d", r.status)
	}
	e.must(e.do(e.tokenA, "POST", "/scim/v2/Users", user("ceo@corp.example")), 201)
	created := e.must(e.do(e.tokenB, "POST", "/scim/v2/Users", user("kai@partner.example")), 201)
	patch := map[string]any{"schemas": []string{"urn:ietf:params:scim:api:messages:2.0:PatchOp"},
		"Operations": []any{map[string]any{"op": "replace", "path": "emails", "value": []any{map[string]any{"value": "kai@corp.example", "primary": true}}}}}
	if r := e.do(e.tokenB, "PATCH", "/scim/v2/Users/"+created["id"].(string), patch); r.status != 400 {
		t.Fatalf("connection B moved its user into connection A's domain: %d", r.status)
	}
}
