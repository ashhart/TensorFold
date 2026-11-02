package db

import (
	"context"
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"fmt"
	"io/fs"
	"sort"
	"strconv"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

//go:embed migrations/*.sql
var migrations embed.FS

// migrationLock serializes concurrent migrators across every Starlight instance.
const migrationLock = 7314_0001

// Migration is one embedded schema step.
type Migration struct {
	Version int
	Name    string
	SQL     string
	Sum     string
}

// Migrations lists the embedded steps in version order.
func Migrations() ([]Migration, error) {
	entries, err := fs.ReadDir(migrations, "migrations")
	if err != nil {
		return nil, err
	}
	var out []Migration
	for _, e := range entries {
		name := e.Name()
		head, _, ok := strings.Cut(name, "_")
		if !ok || !strings.HasSuffix(name, ".sql") {
			return nil, fmt.Errorf("migration %q: want NNNN_name.sql", name)
		}
		v, err := strconv.Atoi(head)
		if err != nil {
			return nil, fmt.Errorf("migration %q: %w", name, err)
		}
		body, err := migrations.ReadFile("migrations/" + name)
		if err != nil {
			return nil, err
		}
		sum := sha256.Sum256(body)
		out = append(out, Migration{Version: v, Name: name, SQL: string(body), Sum: hex.EncodeToString(sum[:])})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Version < out[j].Version })
	for i := range out {
		if out[i].Version != i+1 {
			return nil, fmt.Errorf("migration versions must be contiguous from 1; found %d at position %d", out[i].Version, i+1)
		}
	}
	return out, nil
}

// Migrate applies pending migrations under an advisory lock and refuses edited history.
func Migrate(ctx context.Context, pool *pgxpool.Pool) (applied []string, err error) {
	all, err := Migrations()
	if err != nil {
		return nil, err
	}
	conn, err := pool.Acquire(ctx)
	if err != nil {
		return nil, err
	}
	defer conn.Release()
	if _, err := conn.Exec(ctx, "SELECT pg_advisory_lock($1)", migrationLock); err != nil {
		return nil, err
	}
	defer conn.Exec(context.WithoutCancel(ctx), "SELECT pg_advisory_unlock($1)", migrationLock)
	if _, err := conn.Exec(ctx, `CREATE TABLE IF NOT EXISTS schema_migrations (
		version int PRIMARY KEY, name text NOT NULL, sha256 text NOT NULL, applied_at timestamptz NOT NULL DEFAULT now())`); err != nil {
		return nil, err
	}
	done := map[int]string{}
	rows, err := conn.Query(ctx, "SELECT version, sha256 FROM schema_migrations")
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		var v int
		var sum string
		if err := rows.Scan(&v, &sum); err != nil {
			rows.Close()
			return nil, err
		}
		done[v] = sum
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}
	for v := range done {
		if v > len(all) {
			return nil, fmt.Errorf("database is at schema %d, newer than this build's %d; upgrade Starlight first", v, len(all))
		}
	}
	for _, m := range all {
		if sum, ok := done[m.Version]; ok {
			if sum != m.Sum {
				return nil, fmt.Errorf("migration %s was edited after it was applied", m.Name)
			}
			continue
		}
		err := pgx.BeginFunc(ctx, conn.Conn(), func(tx pgx.Tx) error {
			if _, err := tx.Exec(ctx, m.SQL); err != nil {
				return fmt.Errorf("%s: %w", m.Name, err)
			}
			_, err := tx.Exec(ctx, "INSERT INTO schema_migrations (version, name, sha256) VALUES ($1, $2, $3)", m.Version, m.Name, m.Sum)
			return err
		})
		if err != nil {
			return applied, err
		}
		applied = append(applied, m.Name)
	}
	return applied, nil
}

// Pending reports how many embedded migrations the database has not applied; a newer database is an error.
func Pending(ctx context.Context, q Querier) (int, error) {
	all, err := Migrations()
	if err != nil {
		return 0, err
	}
	var n, top int
	if err := q.QueryRow(ctx, "SELECT count(*), coalesce(max(version), 0) FROM schema_migrations").Scan(&n, &top); err != nil {
		return len(all), nil
	}
	if top > len(all) {
		return 0, fmt.Errorf("database is at schema %d, newer than this build's %d; run the Starlight release that migrated it", top, len(all))
	}
	return len(all) - n, nil
}
