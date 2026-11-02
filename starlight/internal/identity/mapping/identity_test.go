package mapping

import (
	"errors"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
)

const tenant = "8d0c6f1e-2c43-4c2d-9a33-0b1c2d3e4f50"

func entraConn() connections.Connection {
	return connections.Connection{ID: "c1", Preset: "entra", Domains: []string{"example.com"}, Config: connections.Config{
		Claims: connections.ClaimMap{Subject: "oid", Tenant: "tid", Email: "email", EmailVerified: "xms_edov", Name: "name",
			Username: "preferred_username", Groups: "groups"},
		Groups:   connections.Groups{Strategy: connections.GroupsEntraGraph},
		Provider: connections.Provider{EntraTenantID: tenant},
	}}
}

func TestEntraSubjectTenantAndEmailTrust(t *testing.T) {
	c := entraConn()
	claims := map[string]any{"tid": tenant, "oid": "AAAA0000-0000-0000-0000-000000000001", "email": "ada@example.com",
		"preferred_username": "ada@example.com", "groups": []any{"g1", "g2"}}
	id, err := FromClaims(c, "pairwise-sub", claims)
	if err != nil {
		t.Fatal(err)
	}
	if id.Subject != tenant+":aaaa0000-0000-0000-0000-000000000001" || id.RawSubject != "pairwise-sub" {
		t.Fatalf("subject %q raw %q", id.Subject, id.RawSubject)
	}
	if !id.EmailVerified || len(id.Groups) != 2 || !id.GroupsKnown || id.Overage {
		t.Fatalf("identity %+v", id)
	}
	claims["email"] = "ceo@example.com"
	id, _ = FromClaims(c, "s", claims)
	if id.EmailVerified {
		t.Fatal("trusted a mail attribute that differs from the UPN without xms_edov")
	}
	claims["xms_edov"] = true
	id, _ = FromClaims(c, "s", claims)
	if !id.EmailVerified {
		t.Fatal("xms_edov=true not trusted")
	}
	claims["tid"] = "11111111-1111-1111-1111-111111111111"
	if _, err := FromClaims(c, "s", claims); !errors.Is(err, ErrTenant) {
		t.Fatalf("foreign tenant = %v", err)
	}
	delete(claims, "tid")
	if _, err := FromClaims(c, "s", claims); !errors.Is(err, ErrTenant) {
		t.Fatalf("missing tenant = %v", err)
	}
}

func TestEntraOverage(t *testing.T) {
	claims := map[string]any{"tid": tenant, "oid": "o1", "_claim_names": map[string]any{"groups": "src1"},
		"_claim_sources": map[string]any{"src1": map[string]any{"endpoint": "https://graph.windows.net/x"}}}
	id, err := FromClaims(entraConn(), "s", claims)
	if err != nil || !id.Overage || len(id.Groups) != 0 {
		t.Fatalf("overage %+v %v", id, err)
	}
}

func TestGoogleRequiresHostedDomainAndVerifiedEmail(t *testing.T) {
	c := connections.Connection{Preset: "google", Config: connections.Config{Provider: connections.Provider{GoogleDomains: []string{"example.com"}}}}
	ok := map[string]any{"hd": "example.com", "email": "ada@example.com", "email_verified": true}
	if id, err := FromClaims(c, "1", ok); err != nil || !id.EmailVerified {
		t.Fatalf("google ok = %+v %v", id, err)
	}
	for name, claims := range map[string]map[string]any{
		"consumer":   {"email": "ada@personal.example", "email_verified": true},
		"other hd":   {"hd": "other.example", "email": "a@other.example", "email_verified": true},
		"unverified": {"hd": "example.com", "email": "ada@example.com", "email_verified": false},
	} {
		if _, err := FromClaims(c, "1", claims); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
}

func TestGroupShapes(t *testing.T) {
	c := connections.Connection{Config: connections.Config{Claims: connections.ClaimMap{Groups: "roles", GroupsAreKeys: true},
		Groups: connections.Groups{Strategy: connections.GroupsClaim}}}
	id, _ := FromClaims(c, "s", map[string]any{"roles": map[string]any{"admin": map[string]any{}, "dev": map[string]any{}}})
	if len(id.Groups) != 2 || id.Groups[0].ExternalID != "admin" {
		t.Fatalf("object keys %+v", id.Groups)
	}
	c.Config.Claims.GroupsAreKeys = false
	id, _ = FromClaims(c, "s", map[string]any{"roles": "solo"})
	if len(id.Groups) != 1 || id.Groups[0].ExternalID != "solo" {
		t.Fatalf("string group %+v", id.Groups)
	}
	c.Config.Groups.Strategy = connections.GroupsSCIM
	id, _ = FromClaims(c, "s", map[string]any{"roles": []any{"x"}})
	if id.GroupsKnown {
		t.Fatal("claim groups synced although SCIM owns groups")
	}
}

func TestSAMLSubjects(t *testing.T) {
	c := connections.Connection{Config: connections.Config{Claims: connections.ClaimMap{Email: "mail"}, Policy: connections.Policy{TrustEmail: true}}}
	in := SAMLInput{NameID: "transient-123", NameIDFormat: transientFormat, Attributes: map[string][]string{"mail": {"ada@example.com"}}}
	if _, err := FromSAML(c, in); !errors.Is(err, ErrMissingSubject) {
		t.Fatalf("transient NameID = %v", err)
	}
	in.NameIDFormat = emailFormat
	in.NameID = "ada@example.com"
	in.Attributes = map[string][]string{}
	id, err := FromSAML(c, in)
	if err != nil || id.Subject != "ada@example.com" || id.Email != "ada@example.com" || !id.EmailVerified {
		t.Fatalf("email NameID = %+v %v", id, err)
	}
	ec := entraConn()
	ec.Config.Claims = connections.ClaimMap{Subject: "oid-attr", Tenant: "tid-attr", Email: "mail", Username: "upn"}
	ec.Config.Groups.Strategy = connections.GroupsClaim
	in = SAMLInput{NameID: "x", Attributes: map[string][]string{"oid-attr": {"O1"}, "tid-attr": {tenant}, "mail": {"ada@example.com"},
		"upn": {"ada@example.com"}, "http://schemas.microsoft.com/claims/groups.link": {"https://graph.example"}}}
	id, err = FromSAML(ec, in)
	if err != nil || id.Subject != tenant+":o1" || !id.EmailVerified || !id.Overage {
		t.Fatalf("entra saml = %+v %v", id, err)
	}
}

func TestPolicy(t *testing.T) {
	c := connections.Connection{Config: connections.Config{Policy: connections.Policy{AllowedDomains: []string{"example.com"}, RequiredGroups: []string{"Engineering"}}}}
	id := Identity{Email: "ada@example.com"}
	if err := CheckPolicy(c, id); !errors.Is(err, ErrGroupRequired) {
		t.Fatalf("no groups = %v", err)
	}
	id.Groups = []directory.GroupRef{{ExternalID: "g1", DisplayName: "engineering"}}
	if err := CheckPolicy(c, id); err != nil {
		t.Fatalf("group by name = %v", err)
	}
	id.Email = "ada@other.example"
	if err := CheckPolicy(c, id); !errors.Is(err, ErrDomainNotAllowed) {
		t.Fatalf("domain = %v", err)
	}
}
