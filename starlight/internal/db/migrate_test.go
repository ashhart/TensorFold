package db_test

import (
	"context"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

func TestMigrationsApplyAndAreIdempotent(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	applied, err := db.Migrate(ctx, pool)
	if err != nil {
		t.Fatal(err)
	}
	if len(applied) != 0 {
		t.Fatalf("template was not fully migrated; applied %v", applied)
	}
	n, err := db.Pending(ctx, pool)
	if err != nil || n != 0 {
		t.Fatalf("pending = %d, %v", n, err)
	}
	var tables int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'").Scan(&tables); err != nil {
		t.Fatal(err)
	}
	if tables < 40 {
		t.Fatalf("only %d tables", tables)
	}
}

func TestEditedMigrationRefused(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	if _, err := pool.Exec(ctx, "UPDATE schema_migrations SET sha256 = 'x' WHERE version = 1"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Migrate(ctx, pool); err == nil {
		t.Fatal("migrate accepted edited history")
	}
}

func TestNewerSchemaRefused(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	if _, err := pool.Exec(ctx, "INSERT INTO schema_migrations (version, name, sha256) VALUES (999, 'future', 'x')"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Migrate(ctx, pool); err == nil {
		t.Fatal("migrate accepted a newer schema")
	}
	if _, err := db.Pending(ctx, pool); err == nil {
		t.Fatal("an older build would start against a newer schema")
	}
}
