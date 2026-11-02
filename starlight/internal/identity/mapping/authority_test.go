package mapping_test

import (
	"context"
	"errors"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/identity/mapping"
)

func partner(t *testing.T, s *connections.Store, domains []string) connections.Connection {
	t.Helper()
	c, err := s.Create(context.Background(), connections.Input{Slug: "partner", Name: "Partner", Protocol: connections.OIDC, Preset: "generic_oidc",
		Domains: domains, Enabled: true, JIT: true, Config: connections.Config{OIDC: &connections.OIDCConfig{Issuer: "https://idp.partner.example", ClientID: "p"},
			Policy: connections.Policy{TrustEmail: true}}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestAnotherConnectionCannotClaimAnAccount(t *testing.T) {
	pool, corp := setup(t, true)
	owner, err := resolve(pool, corp, mapping.Identity{Subject: "corp|owner", Email: "owner@example.com", EmailVerified: true})
	if err != nil {
		t.Fatal(err)
	}
	p := partner(t, connections.NewStore(pool, keyring.ForTest()), []string{"partner.example"})
	got, err := resolve(pool, p, mapping.Identity{Subject: "partner-17", Email: "owner@example.com", EmailVerified: true})
	if !errors.Is(err, mapping.ErrForeignDomain) || got.ID == owner.ID {
		t.Fatalf("partner identity resolved to %q, %v", got.ID, err)
	}
}

func TestLinkingNeedsTheDomainOrOwnProvisioning(t *testing.T) {
	pool, corp := setup(t, true)
	ctx := context.Background()
	p := partner(t, connections.NewStore(pool, keyring.ForTest()), []string{"partner.example"})
	if _, err := directory.CreateUser(ctx, pool, directory.NewUser{Email: "lee@elsewhere.example", Source: "api"}); err != nil {
		t.Fatal(err)
	}
	if _, err := resolve(pool, p, mapping.Identity{Subject: "p-lee", Email: "lee@elsewhere.example", EmailVerified: true}); !errors.Is(err, mapping.ErrEmailUnverified) {
		t.Fatalf("verified email outside the connection's domains linked: %v", err)
	}
	provisioned, err := directory.CreateUser(ctx, pool, directory.NewUser{Email: "ana@elsewhere.example", Source: "scim", SCIMConnectionID: &corp.ID})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := resolve(pool, p, mapping.Identity{Subject: "p-ana", Email: "ana@elsewhere.example", EmailVerified: true}); !errors.Is(err, mapping.ErrIdentityConflict) {
		t.Fatalf("linked into another connection's provisioned user: %v", err)
	}
	if u, err := resolve(pool, corp, mapping.Identity{Subject: "c-ana", Email: "ana@elsewhere.example"}); err != nil || u.ID != provisioned.ID {
		t.Fatalf("own provisioned user not linked: %+v %v", u, err)
	}
	if _, err := resolve(pool, p, mapping.Identity{Subject: "p-new", Email: "new@example.com", EmailVerified: true}); !errors.Is(err, mapping.ErrForeignDomain) {
		t.Fatalf("created a user in another connection's domain: %v", err)
	}
}
