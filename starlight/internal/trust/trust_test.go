package trust_test

import (
	"context"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
	"github.com/ashhart/TensorFold/starlight/internal/trust"
)

func TestRotationPublishesBeforeSigning(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	keys := keyring.New(mustMaster(t))
	if _, err := keys.Rotate(ctx, pool); err != nil {
		t.Fatal(err)
	}
	if err := trust.Bootstrap(ctx, pool, keys, "starlight.example"); err != nil {
		t.Fatal(err)
	}
	tr, err := trust.Load(ctx, pool, keys, "gw-1")
	if err != nil {
		t.Fatal(err)
	}
	first, _ := tr.Signer()
	if err := trust.RotatePermitKey(ctx, pool, keys); err != nil {
		t.Fatal(err)
	}
	if err := tr.Refresh(ctx); err != nil {
		t.Fatal(err)
	}
	if s, _ := tr.Signer(); s.KID != first.KID || len(tr.PublicKeys()) != 2 {
		t.Fatalf("signing with %s before nodes had the new key; published %d", s.KID, len(tr.PublicKeys()))
	}
	exec(t, pool, "UPDATE permit_keys SET changed_at = now() - interval '3 minutes' WHERE status = 'next'")
	if err := tr.Refresh(ctx); err != nil {
		t.Fatal(err)
	}
	second, _ := tr.Signer()
	if second.KID == first.KID || len(tr.PublicKeys()) != 2 {
		t.Fatalf("after the window signer %s, published %d", second.KID, len(tr.PublicKeys()))
	}
	exec(t, pool, "UPDATE permit_keys SET changed_at = now() - interval '3 days' WHERE status = 'verify'")
	if err := tr.Refresh(ctx); err != nil {
		t.Fatal(err)
	}
	if pub := tr.PublicKeys(); len(pub) != 1 || pub[0].KID != second.KID {
		t.Fatalf("old key still published: %+v", pub)
	}
}

func TestRewrapRetiresThePreviousMaster(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	old := mustMaster(t)
	k1 := keyring.New(old)
	if _, err := k1.Rotate(ctx, pool); err != nil {
		t.Fatal(err)
	}
	sealed, err := k1.Seal([]byte("client-secret"), "idp:1")
	if err != nil {
		t.Fatal(err)
	}
	next := mustMaster(t)
	k2 := keyring.New(next, old)
	if err := k2.Load(ctx, pool); err != nil {
		t.Fatal(err)
	}
	if n, err := k2.Rewrap(ctx, pool); err != nil || n != 1 {
		t.Fatalf("rewrapped %d, %v", n, err)
	}
	k3 := keyring.New(next)
	if err := k3.Load(ctx, pool); err != nil {
		t.Fatalf("new master alone cannot load: %v", err)
	}
	if got, err := k3.Open(sealed, "idp:1"); err != nil || string(got) != "client-secret" {
		t.Fatalf("open after rewrap = %q, %v", got, err)
	}
	if err := keyring.New(old).Load(ctx, pool); err == nil {
		t.Fatal("the retired master still unwraps data keys")
	}
}

func mustMaster(t *testing.T) *keyring.Master {
	m, err := keyring.LoadMaster(keyring.GenerateMasterKey(), "")
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func exec(t *testing.T, pool *pgxpool.Pool, sql string) {
	t.Helper()
	if _, err := pool.Exec(context.Background(), sql); err != nil {
		t.Fatal(err)
	}
}
