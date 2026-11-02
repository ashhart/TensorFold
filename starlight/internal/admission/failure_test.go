package admission_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/admission"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/levels"
)

func TestCallerLeavingDuringAdmissionLeavesNoLease(t *testing.T) {
	f := newFleet(t, 1)
	ctx := context.Background()
	a := f.admitter()
	day := time.Now().UTC().Truncate(24 * time.Hour)
	round := func(i int) {
		blocker, err := f.pool.Begin(ctx)
		if err != nil {
			t.Fatal(err)
		}
		defer blocker.Rollback(ctx)
		if _, err := blocker.Exec(ctx, "SELECT pg_advisory_xact_lock(hashtextextended('starlight:target:' || $1::text, 0))", f.replica); err != nil {
			t.Fatal(err)
		}
		req := f.request(levels.Standard, 5*time.Second)
		req.Budgets = []admission.Counter{{Key: "tokens:day:project:p4", Window: day, Limit: 1000, Amount: 100}}
		rctx, leave := context.WithCancel(ctx)
		defer leave()
		done := make(chan error, 1)
		go func() {
			_, err := a.Admit(rctx, req)
			done <- err
		}()
		eventually(t, "admission waits for the target", func() bool { return active(t, f.pool, "starlight_admit", true) > 0 })
		leave()
		if err := blocker.Commit(ctx); err != nil {
			t.Fatal(err)
		}
		if err := <-done; !errors.Is(err, context.Canceled) {
			t.Fatalf("round %d: admission for a caller who left returned %v", i, err)
		}
		eventually(t, "the admission statement ends", func() bool { return active(t, f.pool, "starlight_admit", false) == 0 })
		var leases int
		var reserved int64
		_ = f.pool.QueryRow(ctx, "SELECT count(*) FROM leases").Scan(&leases)
		_ = f.pool.QueryRow(ctx, "SELECT reserved FROM counters WHERE key = 'tokens:day:project:p4'").Scan(&reserved)
		if leases != 0 || reserved != 0 {
			t.Fatalf("round %d: a caller who left kept %d leases and %d reserved tokens", i, leases, reserved)
		}
	}
	for i := 0; i < 3; i++ {
		round(i)
	}
}

func TestStalledDatabaseFailsAdmissionPromptly(t *testing.T) {
	f := newFleet(t, 1)
	ctx := context.Background()
	a := f.admitter()
	a.Timeout = 300 * time.Millisecond
	blocker, err := f.pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer blocker.Rollback(ctx)
	if _, err := blocker.Exec(ctx, "LOCK TABLE leases IN ACCESS EXCLUSIVE MODE"); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	_, err = a.Admit(ctx, f.request(levels.Standard, time.Minute))
	if took := time.Since(start); !db.Unavailable(err) || took > 3*time.Second {
		t.Fatalf("admission while the database stalls returned %v after %v", err, took)
	}
}
