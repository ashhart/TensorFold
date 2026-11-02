package admission

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/db"
)

// holding maps every lease and queue entry this process holds to its gateway; Heartbeat keeps only these alive.
var holding = struct {
	sync.Mutex
	ids map[string]string
}{ids: map[string]string{}}

func hold(id, gateway string) {
	holding.Lock()
	holding.ids[id] = gateway
	holding.Unlock()
}

func drop(id string) {
	holding.Lock()
	delete(holding.ids, id)
	holding.Unlock()
}

func held(gateway string) []string {
	holding.Lock()
	defer holding.Unlock()
	out := []string{}
	for id, g := range holding.ids {
		if g == gateway {
			out = append(out, id)
		}
	}
	return out
}

// Heartbeat keeps the leases and queue entries this process holds for gatewayID alive until ctx ends.
func Heartbeat(ctx context.Context, pool *pgxpool.Pool, gatewayID, hostname, version string) {
	_, _ = pool.Exec(ctx, `INSERT INTO gateways (id, hostname, version) VALUES ($1, $2, $3)
		ON CONFLICT (id) DO UPDATE SET heartbeat_at = now()`, gatewayID, hostname, version)
	t := time.NewTicker(15 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			_, _ = pool.Exec(context.Background(), "DELETE FROM gateways WHERE id = $1", gatewayID)
			return
		case <-t.C:
		}
		beat(ctx, pool, gatewayID)
	}
}

// beat refreshes only rows this process still holds, so a lease or entry it lost track of goes stale and is reaped.
func beat(ctx context.Context, pool *pgxpool.Pool, gatewayID string) {
	ids := held(gatewayID)
	_, _ = pool.Exec(ctx, "UPDATE leases SET heartbeat_at = now() WHERE id = ANY($1::uuid[])", ids)
	_, _ = pool.Exec(ctx, "UPDATE admission_queue SET heartbeat_at = now() WHERE id = ANY($1::uuid[])", ids)
	_, _ = pool.Exec(ctx, "UPDATE gateways SET heartbeat_at = now() WHERE id = $1", gatewayID)
}

// Reap frees leases whose gateway stopped heartbeating and charges their reservations in full, in one sorted pass.
func Reap(ctx context.Context, pool *pgxpool.Pool) (int, error) {
	reaped := 0
	err := db.Tx(ctx, pool, func(tx pgx.Tx) error {
		rows, err := tx.Query(ctx, `DELETE FROM leases WHERE heartbeat_at < now() - interval '`+staleSQL+`'
			RETURNING model_id, budget_keys, starlight_amounts(budget_keys, reserved_tokens, reserved_micros)`)
		if err != nil {
			return err
		}
		var keys []string
		var amounts []int64
		models := map[string]bool{}
		n := 0
		for rows.Next() {
			var model string
			var k []string
			var a []int64
			if err := rows.Scan(&model, &k, &a); err != nil {
				rows.Close()
				return err
			}
			keys, amounts = append(keys, k...), append(amounts, a...)
			models[model] = true
			n++
		}
		rows.Close()
		if err := rows.Err(); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, "SELECT starlight_settle_counters($1, $2, $2)", keys, amounts); err != nil {
			return err
		}
		for m := range models {
			if _, err := tx.Exec(ctx, "SELECT pg_notify('starlight_capacity', $1)", m); err != nil {
				return err
			}
		}
		reaped = n
		_, err = tx.Exec(ctx, "DELETE FROM admission_queue WHERE heartbeat_at < now() - interval '"+queueSQL+"' OR deadline < now() - interval '1 minute'")
		return err
	})
	return reaped, err
}

// Prune drops expired counter windows and dead gateway rows.
func Prune(ctx context.Context, pool *pgxpool.Pool) error {
	stmts := []string{
		"DELETE FROM counters WHERE (key LIKE 'rpm:%' OR key LIKE 'tpm:%' OR key LIKE 'rl:%') AND window_start < now() - interval '1 hour'",
		"DELETE FROM counters WHERE key LIKE '%:day:%' AND window_start < now() - interval '45 days'",
		"DELETE FROM counters WHERE key LIKE '%:month:%' AND window_start < now() - interval '400 days'",
		"DELETE FROM gateways WHERE heartbeat_at < now() - interval '10 minutes'",
	}
	for _, s := range stmts {
		if _, err := pool.Exec(ctx, s); err != nil {
			return err
		}
	}
	return nil
}

// EnsurePartitions creates monthly usage partitions for this month and the next so the default partition stays empty.
func EnsurePartitions(ctx context.Context, pool *pgxpool.Pool, now time.Time) error {
	start := time.Date(now.UTC().Year(), now.UTC().Month(), 1, 0, 0, 0, 0, time.UTC)
	for i := 0; i < 2; i++ {
		from, to := start.AddDate(0, i, 0), start.AddDate(0, i+1, 0)
		name := fmt.Sprintf("usage_events_%04d%02d", from.Year(), from.Month())
		var exists bool
		if err := pool.QueryRow(ctx, "SELECT EXISTS (SELECT 1 FROM pg_class WHERE relname = $1)", name).Scan(&exists); err != nil {
			return err
		}
		if exists {
			continue
		}
		var stray int
		if err := pool.QueryRow(ctx, "SELECT count(*) FROM usage_events_default WHERE created_at >= $1 AND created_at < $2", from, to).Scan(&stray); err != nil {
			return err
		}
		if stray > 0 {
			continue
		}
		sql := fmt.Sprintf("CREATE TABLE IF NOT EXISTS %s PARTITION OF usage_events FOR VALUES FROM ('%s') TO ('%s')",
			name, from.Format(time.RFC3339), to.Format(time.RFC3339))
		if _, err := pool.Exec(ctx, sql); err != nil {
			return err
		}
	}
	return nil
}
