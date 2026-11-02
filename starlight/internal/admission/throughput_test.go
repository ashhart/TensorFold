package admission_test

import (
	"context"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/admission"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/levels"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

// TestAdmissionScalesWithReplicas measures admit-and-settle cycles a second for one model; STARLIGHT_SCALE=1 runs it.
func TestAdmissionScalesWithReplicas(t *testing.T) {
	if os.Getenv("STARLIGHT_SCALE") == "" {
		t.Skip("set STARLIGHT_SCALE=1 to measure admission throughput")
	}
	for _, replicas := range []int{1, 4, 8} {
		pool := testutil.DB(t)
		ctx := context.Background()
		model, dep := ids.New(), ids.New()
		exec := func(sql string, args ...any) {
			if _, err := pool.Exec(ctx, sql, args...); err != nil {
				t.Fatal(err)
			}
		}
		exec("INSERT INTO models (id, name, display_name) VALUES ($1, 'bench-model', 'Bench')", model)
		exec(`INSERT INTO deployments (id, model_id, name, spec) VALUES ($1, $2, 'bench', '{"model":{"kind":"path","path":"/m"}}')`, dep, model)
		for i := 0; i < replicas; i++ {
			node := ids.New()
			exec("INSERT INTO nodes (id, name, status, address, last_seen_at) VALUES ($1, $2, 'active', '192.0.2.10:7443', now())", node, fmt.Sprintf("node-%d", i))
			exec("INSERT INTO replicas (id, deployment_id, deployment_revision, node_id, state, slots, epoch) VALUES ($1, $2, 1, $3, 'ready', $4, 1)",
				ids.New(), dep, node, 1024/replicas)
		}
		var done atomic.Int64
		var wg sync.WaitGroup
		stop := time.Now().Add(3 * time.Second)
		for g := 0; g < 64; g++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				a := &admission.Admitter{Pool: pool, GatewayID: ids.New(), Poll: 50 * time.Millisecond}
				for time.Now().Before(stop) {
					l, err := a.Admit(ctx, admission.Request{RequestID: ids.New(), ModelID: model, Priority: levels.Standard, Class: levels.Internal,
						Local: true, Tokens: 100, QueueUntil: time.Now().Add(10 * time.Second), Affinity: uint64(time.Now().UnixNano())})
					if err != nil {
						t.Error(err)
						return
					}
					if err := a.Settle(ctx, l, admission.Usage{Prompt: 10, Completion: 5, Observed: true}, admission.Record{RequestID: "r",
						PrincipalType: "user", PrincipalID: ids.New(), ProjectID: ids.New(), ModelName: "m", Protocol: "openai.chat", Status: 200,
						Outcome: "ok", Classification: "internal"}); err != nil {
						t.Error(err)
						return
					}
					done.Add(1)
				}
			}()
		}
		wg.Wait()
		t.Logf("%d replicas: %.0f admit-and-settle cycles a second from 64 callers", replicas, float64(done.Load())/3)
	}
}
