package audit_test

import (
	"context"
	"sync"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

func TestChainSurvivesConcurrencyAndDetectsTampering(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	var wg sync.WaitGroup
	for i := 0; i < 40; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			e := audit.Actor(nil)
			e.Action = "test.event"
			e.Details = map[string]any{"i": i, "nested": map[string]any{"b": 2, "a": 1}}
			if err := audit.Log(ctx, pool, e); err != nil {
				t.Error(err)
			}
		}(i)
	}
	wg.Wait()
	res, err := audit.Verify(ctx, pool)
	if err != nil || res.Checked != 40 || res.From != 1 || res.To != 40 {
		t.Fatalf("verify = %+v, %v", res, err)
	}
	if _, err := pool.Exec(ctx, "UPDATE audit_events SET details = '{\"i\": 999}' WHERE seq = 17"); err != nil {
		t.Fatal(err)
	}
	if _, err := audit.Verify(ctx, pool); err == nil {
		t.Fatal("tampered event not detected")
	}
}

func TestPrunedPrefixVerifiesFromAnchor(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	for i := 0; i < 5; i++ {
		e := audit.Actor(nil)
		e.Action = "test.event"
		if err := audit.Log(ctx, pool, e); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := pool.Exec(ctx, "DELETE FROM audit_events WHERE seq <= 2"); err != nil {
		t.Fatal(err)
	}
	res, err := audit.Verify(ctx, pool)
	if err != nil || res.From != 3 || res.Checked != 3 {
		t.Fatalf("verify after prune = %+v, %v", res, err)
	}
	if _, err := pool.Exec(ctx, "DELETE FROM audit_events WHERE seq = 4"); err != nil {
		t.Fatal(err)
	}
	if _, err := audit.Verify(ctx, pool); err == nil {
		t.Fatal("gap not detected")
	}
}
