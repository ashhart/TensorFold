package scim

import (
	"testing"
	"time"
)

// TestFiltersRunOnPostgres executes every supported filter form through the API and checks the matches.
func TestFiltersRunOnPostgres(t *testing.T) {
	e := setup(t)
	before := time.Now().Add(-time.Minute).UTC().Format(time.RFC3339)
	mk := func(body map[string]any) string {
		return e.must(e.do(e.tokenA, "POST", "/scim/v2/Users", body), 201)["id"].(string)
	}
	ann := mk(map[string]any{"userName": "ann@example.com", "externalId": "E-1", "active": true, "title": "Engineer",
		"name":        map[string]any{"givenName": "Ann", "familyName": "Lee"},
		"emails":      []any{map[string]any{"value": "ann@example.com", "type": "work", "primary": true}, map[string]any{"value": "ann@home.example", "type": "home"}},
		URNEnterprise: map[string]any{"department": "R&D", "manager": map[string]any{"value": "boss"}}})
	mk(map[string]any{"userName": "bob_50%@example.com", "active": false, "emails": []any{map[string]any{"value": "bob@example.com"}},
		"phoneNumbers": []any{map[string]any{"value": "+1 555 0101", "type": "mobile"}}})
	mk(map[string]any{"userName": "cy@example.com", "displayName": "Cy", "emails": []any{map[string]any{"value": "cy@example.com", "type": "work"}}})
	g := e.must(e.do(e.tokenA, "POST", "/scim/v2/Groups", map[string]any{"displayName": "Readers", "externalId": "G-1",
		"members": []any{map[string]any{"value": ann}}}), 201)["id"].(string)
	e.must(e.do(e.tokenA, "POST", "/scim/v2/Groups", map[string]any{"displayName": "Empty"}), 201)

	users := map[string]float64{
		`active eq true`: 2, `active eq false`: 1, `active ne true`: 1, `active pr`: 3,
		`externalId eq "E-1"`: 1, `externalId eq "e-1"`: 0, `externalId pr`: 1,
		`name.givenName sw "an"`: 1, `name.familyName eq null`: 2, `title pr`: 1, `title eq "engineer"`: 1,
		`emails co "home.example"`: 1, `emails.value eq "BOB@example.com"`: 1, `emails ne "cy@example.com"`: 2,
		`emails[type eq "work"]`: 2, `emails[type eq "home" and value ew ".example"]`: 1, `emails[type eq "work"].value sw "cy"`: 1,
		`phoneNumbers pr`: 1, `phoneNumbers[type eq "mobile"]`: 1,
		`userName co "50%"`: 1, `userName co "_5"`: 1, `userName co "%"`: 1, `userName gt "b"`: 2, `userName le "ann@example.com"`: 1,
		`displayName eq "Cy"`: 1, `displayName pr`: 1,
		`meta.created gt "` + before + `"`: 3, `meta.lastModified lt "` + before + `"`: 0,
		`urn:ietf:params:scim:schemas:extension:enterprise:2.0:User:department eq "r&d"`:     1,
		`urn:ietf:params:scim:schemas:extension:enterprise:2.0:User:manager.value eq "boss"`: 1,
		`id eq "` + ann + `"`: 1, `not (id eq "` + ann + `")`: 2,
		`(active eq true and title pr) or userName sw "bob"`: 2,
	}
	for f, want := range users {
		got := e.must(e.do(e.tokenA, "GET", "/scim/v2/Users?filter="+q(f), nil), 200)["totalResults"].(float64)
		if got != want {
			t.Errorf("users %s: %v, want %v", f, got, want)
		}
	}
	groups := map[string]float64{
		`displayName eq "readers"`: 1, `externalId eq "G-1"`: 1, `externalId pr`: 1, `members pr`: 1, `not (members pr)`: 1,
		`members eq "` + ann + `"`: 1, `members[value eq "` + ann + `"]`: 1, `members.value eq "` + ann + `"`: 1,
		`members[display co "ann"]`: 1, `id eq "` + g + `" and members ne "` + ann + `"`: 0,
	}
	for f, want := range groups {
		got := e.must(e.do(e.tokenA, "GET", "/scim/v2/Groups?filter="+q(f), nil), 200)["totalResults"].(float64)
		if got != want {
			t.Errorf("groups %s: %v, want %v", f, got, want)
		}
	}
}
