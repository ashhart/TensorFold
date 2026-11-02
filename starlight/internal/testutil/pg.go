// Package testutil gives integration tests an isolated, migrated Postgres database each.
package testutil

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// EnvURL names the admin connection integration tests use; tests skip when it is unset.
const EnvURL = "STARLIGHT_TEST_DATABASE_URL"

// DB returns a pool on a fresh database cloned from a migrated template and dropped after the test.
func DB(t testing.TB) *pgxpool.Pool {
	t.Helper()
	admin := os.Getenv(EnvURL)
	if admin == "" {
		t.Skip(EnvURL + " is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	tmpl, err := template(ctx, admin)
	if err != nil {
		t.Fatal(err)
	}
	name := "starlight_t_" + strings.ReplaceAll(ids.New(), "-", "")[:24]
	conn, err := pgx.Connect(ctx, admin)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close(ctx)
	if _, err := conn.Exec(ctx, fmt.Sprintf("CREATE DATABASE %s TEMPLATE %s", name, tmpl)); err != nil {
		t.Fatal(err)
	}
	pool, err := db.Open(ctx, withDB(admin, name), db.Options{MaxConns: 64, ApplicationName: "starlight-test"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		c, err := pgx.Connect(context.Background(), admin)
		if err != nil {
			return
		}
		defer c.Close(context.Background())
		_, _ = c.Exec(context.Background(), fmt.Sprintf("DROP DATABASE IF EXISTS %s WITH (FORCE)", name))
	})
	return pool
}

func template(ctx context.Context, admin string) (string, error) {
	all, err := db.Migrations()
	if err != nil {
		return "", err
	}
	h := sha256.New()
	for _, m := range all {
		h.Write([]byte(m.Sum))
	}
	name := "starlight_tmpl_" + hex.EncodeToString(h.Sum(nil))[:16]
	conn, err := pgx.Connect(ctx, admin)
	if err != nil {
		return "", err
	}
	defer conn.Close(ctx)
	if _, err := conn.Exec(ctx, "SELECT pg_advisory_lock(7314002)"); err != nil {
		return "", err
	}
	defer conn.Exec(context.Background(), "SELECT pg_advisory_unlock(7314002)")
	var exists bool
	if err := conn.QueryRow(ctx, "SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname = $1)", name).Scan(&exists); err != nil {
		return "", err
	}
	if exists {
		return name, nil
	}
	build := name + "_build"
	_, _ = conn.Exec(ctx, fmt.Sprintf("DROP DATABASE IF EXISTS %s WITH (FORCE)", build))
	if _, err := conn.Exec(ctx, fmt.Sprintf("CREATE DATABASE %s", build)); err != nil {
		return "", err
	}
	pool, err := db.Open(ctx, withDB(admin, build), db.Options{MaxConns: 2})
	if err != nil {
		return "", err
	}
	_, err = db.Migrate(ctx, pool)
	pool.Close()
	if err != nil {
		return "", err
	}
	if _, err := conn.Exec(ctx, fmt.Sprintf("ALTER DATABASE %s RENAME TO %s", build, name)); err != nil {
		return "", err
	}
	return name, nil
}

func withDB(raw, name string) string {
	u, err := url.Parse(raw)
	if err != nil {
		return raw
	}
	u.Path = "/" + name
	return u.String()
}
