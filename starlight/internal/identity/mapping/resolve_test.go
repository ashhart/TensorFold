package mapping_test

import (
	"context"
	"errors"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/identity/connections"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/identity/mapping"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

func setup(t *testing.T, jit bool) (*pgxpool.Pool, connections.Connection) {
	t.Helper()
	pool := testutil.DB(t)
	s := connections.NewStore(pool, keyring.ForTest())
	c, err := s.Create(context.Background(), connections.Input{Slug: "corp", Name: "Corp", Protocol: connections.OIDC, Preset: "generic_oidc", Domains: []string{"example.com"},
		Enabled: true, JIT: jit, Config: connections.Config{OIDC: &connections.OIDCConfig{Issuer: "https://idp.example.com", ClientID: "t"},
			Claims: connections.ClaimMap{Groups: "groups"}}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return pool, c
}

func resolve(pool *pgxpool.Pool, c connections.Connection, id mapping.Identity) (directory.User, error) {
	var u directory.User
	err := db.Tx(context.Background(), pool, func(tx pgx.Tx) error {
		var err error
		u, err = mapping.Resolve(context.Background(), tx, c, id)
		return err
	})
	return u, err
}

func groupNames(t *testing.T, pool *pgxpool.Pool, userID string) map[string]bool {
	t.Helper()
	gs, err := directory.UserGroups(context.Background(), pool, userID)
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]bool{}
	for _, g := range gs {
		out[g.ExternalID] = true
	}
	return out
}

func TestJITCreatesLinksAndSyncsGroups(t *testing.T) {
	pool, c := setup(t, true)
	id := mapping.Identity{Subject: "s1", Email: "ada@example.com", EmailVerified: true, Name: "Ada", GroupsKnown: true,
		Groups: []directory.GroupRef{{ExternalID: "eng"}, {ExternalID: "ops"}}}
	u, err := resolve(pool, c, id)
	if err != nil {
		t.Fatal(err)
	}
	if g := groupNames(t, pool, u.ID); !g["eng"] || !g["ops"] {
		t.Fatalf("groups %v", g)
	}
	id.Groups = []directory.GroupRef{{ExternalID: "eng"}}
	again, err := resolve(pool, c, id)
	if err != nil || again.ID != u.ID {
		t.Fatalf("second sign-in = %+v %v", again, err)
	}
	if g := groupNames(t, pool, u.ID); !g["eng"] || g["ops"] {
		t.Fatalf("removed group still present: %v", g)
	}
}

func TestJITOffAndUnverifiedEmail(t *testing.T) {
	pool, c := setup(t, false)
	if _, err := resolve(pool, c, mapping.Identity{Subject: "s1", Email: "new@example.com", EmailVerified: true}); !errors.Is(err, mapping.ErrNotProvisioned) {
		t.Fatalf("jit off = %v", err)
	}
	if _, err := directory.CreateUser(context.Background(), pool, directory.NewUser{Email: "ceo@example.com", Source: "api"}); err != nil {
		t.Fatal(err)
	}
	if _, err := resolve(pool, c, mapping.Identity{Subject: "attacker", Email: "ceo@example.com"}); !errors.Is(err, mapping.ErrEmailUnverified) {
		t.Fatalf("unverified collision = %v", err)
	}
	u, err := resolve(pool, c, mapping.Identity{Subject: "real-ceo", Email: "ceo@example.com", EmailVerified: true})
	if err != nil {
		t.Fatalf("verified link = %v", err)
	}
	if _, err := resolve(pool, c, mapping.Identity{Subject: "second-subject", Email: "ceo@example.com", EmailVerified: true}); !errors.Is(err, mapping.ErrIdentityConflict) {
		t.Fatalf("second subject for a linked user = %v", err)
	}
	ctx := context.Background()
	if err := db.Tx(ctx, pool, func(tx pgx.Tx) error { return directory.SetStatus(ctx, tx, u.ID, directory.Suspended, "test") }); err != nil {
		t.Fatal(err)
	}
	if _, err := resolve(pool, c, mapping.Identity{Subject: "real-ceo", Email: "ceo@example.com", EmailVerified: true}); !errors.Is(err, mapping.ErrDisabled) {
		t.Fatalf("suspended user = %v", err)
	}
}

func TestSCIMProvisionedUserLinksWithoutVerifiedEmail(t *testing.T) {
	pool, c := setup(t, false)
	ctx := context.Background()
	u, err := directory.CreateUser(ctx, pool, directory.NewUser{Email: "bo@example.com", Source: "scim", SCIMConnectionID: &c.ID})
	if err != nil {
		t.Fatal(err)
	}
	got, err := resolve(pool, c, mapping.Identity{Subject: "bo-sub", Email: "bo@example.com"})
	if err != nil || got.ID != u.ID {
		t.Fatalf("scim link = %+v %v", got, err)
	}
	if found, err := mapping.Lookup(ctx, pool, c, mapping.Identity{Subject: "bo-sub"}); err != nil || found.ID != u.ID {
		t.Fatalf("lookup = %+v %v", found, err)
	}
	if _, err := mapping.Lookup(ctx, pool, c, mapping.Identity{Subject: "unknown"}); !errors.Is(err, mapping.ErrNotProvisioned) {
		t.Fatalf("lookup unknown = %v", err)
	}
}

func TestSCIMOwnsProfilesAndReprovisionedUsersRelink(t *testing.T) {
	pool, c := setup(t, false)
	ctx := context.Background()
	u, err := directory.CreateUser(ctx, pool, directory.NewUser{Email: "cy@example.com", DisplayName: "Cy (SCIM)", Source: "scim", SCIMConnectionID: &c.ID})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := resolve(pool, c, mapping.Identity{Subject: "cy-sub", Email: "cy@example.com", EmailVerified: true, Name: "Cy From Claims"}); err != nil {
		t.Fatal(err)
	}
	got, _ := directory.GetUser(ctx, pool, u.ID)
	if got.DisplayName != "Cy (SCIM)" {
		t.Fatalf("login overwrote a SCIM-owned profile: %q", got.DisplayName)
	}
	if _, err := pool.Exec(ctx, "DELETE FROM identities WHERE user_id = $1", u.ID); err != nil {
		t.Fatal(err)
	}
	if err := db.Tx(ctx, pool, func(tx pgx.Tx) error { return directory.SetStatus(ctx, tx, u.ID, directory.Deactivated, "scim delete") }); err != nil {
		t.Fatal(err)
	}
	fresh, err := directory.CreateUser(ctx, pool, directory.NewUser{Email: "cy@example.com", Source: "scim", SCIMConnectionID: &c.ID})
	if err != nil {
		t.Fatal(err)
	}
	relinked, err := resolve(pool, c, mapping.Identity{Subject: "cy-sub", Email: "cy@example.com"})
	if err != nil || relinked.ID != fresh.ID {
		t.Fatalf("re-provisioned user not relinked: %+v %v", relinked, err)
	}
}

func TestJITProfileRefresh(t *testing.T) {
	pool, c := setup(t, true)
	u, err := resolve(pool, c, mapping.Identity{Subject: "s", Email: "di@example.com", EmailVerified: true, Name: "Di"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := resolve(pool, c, mapping.Identity{Subject: "s", Email: "di@example.com", EmailVerified: true, Name: "Di Renamed"}); err != nil {
		t.Fatal(err)
	}
	got, _ := directory.GetUser(context.Background(), pool, u.ID)
	if got.DisplayName != "Di Renamed" {
		t.Fatalf("jit profile not refreshed: %q", got.DisplayName)
	}
}
