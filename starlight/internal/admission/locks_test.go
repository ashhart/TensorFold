package admission_test

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/admission"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/levels"
)

// active counts running statements whose text contains needle, only those waiting for a lock when locked is set.
func active(t *testing.T, pool *pgxpool.Pool, needle string, locked bool) int {
	t.Helper()
	var n int
	if err := pool.QueryRow(context.Background(), `SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()
		AND pid <> pg_backend_pid() AND state = 'active' AND (NOT $2 OR wait_event_type = 'Lock') AND query LIKE '%' || $1 || '%'`,
		needle, locked).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

// eventually fails the test unless cond holds within five seconds.
func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	for deadline := time.Now().Add(5 * time.Second); !cond(); time.Sleep(5 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatalf("gave up waiting until %s", what)
		}
	}
}

func TestAdmitAndSettleNeverDeadlockOnRelatedLimits(t *testing.T) {
	f := newFleet(t, 1000)
	ctx := context.Background()
	project := ids.New()
	limits := []admission.Limit{
		{ScopeType: "project", ScopeID: project, Kind: "tokens_per_day", Value: 1 << 50},
		{ScopeType: "project", ScopeID: project, ModelID: f.model, Kind: "tokens_per_day", Value: 1 << 50},
	}
	s := admission.Subject{ProjectID: project, ModelID: f.model}
	stop := time.Now().Add(2 * time.Second)
	var cycles atomic.Int64
	errs := make(chan error, 16)
	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			a := f.admitter()
			for time.Now().Before(stop) {
				req := f.request(levels.Standard, 10*time.Second)
				req.Scopes, req.Rates, req.Budgets = admission.Build(limits, s, time.Now(), 100, 0)
				l, err := a.Admit(ctx, req)
				if err == nil {
					err = a.Settle(ctx, l, admission.Usage{Prompt: 10, Completion: 5, Observed: true}, record())
				}
				if err != nil {
					errs <- err
					return
				}
				cycles.Add(1)
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	var counters, leases int
	var low, high, reserved int64
	if err := f.pool.QueryRow(ctx, "SELECT count(*), min(used), max(used), max(reserved), (SELECT count(*) FROM leases) FROM counters").
		Scan(&counters, &low, &high, &reserved, &leases); err != nil {
		t.Fatal(err)
	}
	if want := 15 * cycles.Load(); want == 0 || counters != 2 || low != want || high != want || reserved != 0 || leases != 0 {
		t.Fatalf("%d cycles of 15 tokens: %d counters used %d to %d, reserved %d, %d leases left", cycles.Load(), counters, low, high, reserved, leases)
	}
}

func TestReapSettlesCountersInAdmissionOrder(t *testing.T) {
	f := newFleet(t, 2)
	ctx := context.Background()
	a := f.admitter()
	day := time.Now().UTC().Truncate(24 * time.Hour)
	project := "tokens:day:project:" + ids.New()
	model := project + "/model:" + f.model
	for _, key := range []string{model, project} {
		req := f.request(levels.Standard, time.Second)
		req.Budgets = []admission.Counter{{Key: key, Window: day, Limit: 1000, Amount: 100}}
		if _, err := a.Admit(ctx, req); err != nil {
			t.Fatal(err)
		}
	}
	f.exec(t, "UPDATE leases SET heartbeat_at = now() - interval '2 minutes'")
	blocker, err := f.pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer blocker.Rollback(ctx)
	if _, err := blocker.Exec(ctx, "SELECT 1 FROM counters WHERE key = $1 FOR UPDATE", project); err != nil {
		t.Fatal(err)
	}
	reaped := make(chan error, 1)
	go func() {
		_, err := admission.Reap(ctx, f.pool)
		reaped <- err
	}()
	eventually(t, "reap waits for a counter", func() bool { return active(t, f.pool, "starlight_settle_counters", true) > 0 })
	probe, err := f.pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := probe.Exec(ctx, "SELECT 1 FROM counters WHERE key = $1 FOR UPDATE NOWAIT", model); err != nil {
		t.Errorf("reap locked %s before %s, the reverse of admission's order: %v", model, project, err)
	}
	_ = probe.Rollback(ctx)
	_ = blocker.Rollback(ctx)
	if err := <-reaped; err != nil {
		t.Fatal(err)
	}
	var charged int
	_ = f.pool.QueryRow(ctx, "SELECT count(*) FROM counters WHERE used = 100 AND reserved = 0").Scan(&charged)
	if charged != 2 {
		t.Fatalf("reap charged %d of 2 counters in full", charged)
	}
}
