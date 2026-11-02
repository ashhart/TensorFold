package admission

import (
	"context"
	"slices"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/levels"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

func execer(t *testing.T, pool *pgxpool.Pool) func(string, ...any) {
	return func(sql string, args ...any) {
		t.Helper()
		if _, err := pool.Exec(context.Background(), sql, args...); err != nil {
			t.Fatalf("%s: %v", sql, err)
		}
	}
}

func usage() Record {
	return Record{RequestID: "r", PrincipalType: "user", PrincipalID: ids.New(), ProjectID: ids.New(), ModelName: "m", Protocol: "openai.chat",
		Status: 200, Outcome: "ok", Classification: "internal"}
}

func TestHeartbeatKeepsOnlyWhatThisProcessHolds(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	exec := execer(t, pool)
	model, dep, node := ids.New(), ids.New(), ids.New()
	exec("INSERT INTO models (id, name, display_name) VALUES ($1, 'qwen3.8-27b', 'Qwen')", model)
	exec(`INSERT INTO deployments (id, model_id, name, spec) VALUES ($1, $2, 'qwen-mac', '{"model":{"kind":"path","path":"/m"}}')`, dep, model)
	exec("INSERT INTO nodes (id, name, status, address, last_seen_at) VALUES ($1, 'studio-1', 'active', '192.0.2.10:7443', now())", node)
	exec("INSERT INTO replicas (id, deployment_id, deployment_revision, node_id, state, slots, epoch) VALUES ($1, $2, 1, $3, 'ready', 2, 1)",
		ids.New(), dep, node)
	a := &Admitter{Pool: pool, GatewayID: ids.New(), Poll: 50 * time.Millisecond}
	request := func() Request {
		return Request{RequestID: ids.New(), ModelID: model, Priority: levels.Standard, Class: levels.Internal, Local: true, Tokens: 100,
			QueueUntil: time.Now().Add(5 * time.Second)}
	}
	kept, err := a.Admit(ctx, request())
	if err != nil {
		t.Fatal(err)
	}
	lost, err := a.Admit(ctx, request())
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Settle(ctx, lost, Usage{Observed: true}, Record{PrincipalID: "not-a-uuid"}); err == nil {
		t.Fatal("a settlement Postgres refuses succeeded")
	}
	waiting := request()
	waited := make(chan error, 1)
	go func() {
		l, err := a.Admit(ctx, waiting)
		if err == nil {
			err = a.Settle(ctx, l, Usage{Observed: true}, usage())
		}
		waited <- err
	}()
	for len(held(a.GatewayID)) < 2 {
		time.Sleep(time.Millisecond)
	}
	exec(`INSERT INTO admission_queue (id, request_id, gateway_id, model_id, priority, base_rank, candidates, deadline)
		VALUES ($1, 'orphan', $2, $3, 'standard', 1000, '{}', now() + interval '1 minute')`, ids.New(), a.GatewayID, model)
	exec("UPDATE leases SET heartbeat_at = now() - interval '2 minutes'")
	exec("UPDATE admission_queue SET heartbeat_at = now() - interval '2 minutes'")
	beat(ctx, pool, a.GatewayID)
	var fresh []string
	if err := pool.QueryRow(ctx, `SELECT array(SELECT id::text FROM leases WHERE heartbeat_at > now() - interval '1 minute'
		UNION ALL SELECT request_id FROM admission_queue WHERE heartbeat_at > now() - interval '1 minute')`).Scan(&fresh); err != nil {
		t.Fatal(err)
	}
	if want := []string{kept.ID, waiting.RequestID}; !slices.Equal(slices.Sorted(slices.Values(fresh)), slices.Sorted(slices.Values(want))) {
		t.Fatalf("heartbeat refreshed %v, want only lease %s and queued request %s", fresh, kept.ID, waiting.RequestID)
	}
	if n, err := Reap(ctx, pool); err != nil || n != 1 {
		t.Fatalf("reaped %d leases, %v", n, err)
	}
	if err := <-waited; err != nil {
		t.Fatalf("the queued request lost its place: %v", err)
	}
	var orphans int
	_ = pool.QueryRow(ctx, "SELECT count(*) FROM admission_queue WHERE request_id = 'orphan'").Scan(&orphans)
	if orphans != 0 {
		t.Fatal("a queue entry no process holds survived the reaper")
	}
}

func TestRankGroupNeedsEveryNodeCleared(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	exec := execer(t, pool)
	model, dep, lead, member, replica := ids.New(), ids.New(), ids.New(), ids.New(), ids.New()
	exec("INSERT INTO models (id, name, display_name) VALUES ($1, 'glm-flash', 'GLM')", model)
	exec(`INSERT INTO deployments (id, model_id, name, pool, spec) VALUES ($1, $2, 'glm-pair', 'spark',
		'{"model":{"kind":"path","path":"/glm"},"nodes":2,"tensor_parallel":"glm"}')`, dep, model)
	exec(`INSERT INTO nodes (id, name, pool, status, address, last_seen_at, max_classification) VALUES
		($1, 'vault-spark', 'spark', 'active', 'vault:7443', now(), 'restricted'), ($2, 'lab-spark', 'spark', 'active', 'lab:7443', now(), 'restricted')`,
		lead, member)
	exec(`INSERT INTO replicas (id, deployment_id, deployment_revision, spec, node_id, state, slots, epoch, rank_port) VALUES
		($1, $2, 1, '{"nodes":2,"tensor_parallel":"glm"}', $3, 'ready', 4, 1, 29500)`, replica, dep, lead)
	exec("INSERT INTO replica_members (replica_id, member, node_id, state) VALUES ($1, 1, $2, 'ready')", replica, member)
	a := &Admitter{Pool: pool, GatewayID: ids.New()}
	req := Request{RequestID: ids.New(), ModelID: model, Priority: levels.Standard, Class: levels.Restricted, Local: true}
	eligible := func(when string) bool {
		t.Helper()
		local, _, _, err := candidates(ctx, pool, req)
		if err != nil {
			t.Fatal(err)
		}
		l, _, err := a.claim(ctx, req, nil, Target{ReplicaID: replica, NodeID: lead})
		if err != nil {
			t.Fatal(err)
		}
		if l != nil {
			if err := a.Settle(ctx, l, Usage{Observed: true}, usage()); err != nil {
				t.Fatal(err)
			}
		}
		if (len(local) == 1) != (l != nil) {
			t.Fatalf("%s: target selection (%d candidates) and starlight_admit (lease %v) disagree", when, len(local), l != nil)
		}
		return l != nil
	}
	if !eligible("cleared") {
		t.Fatal("a group whose nodes are all live and cleared was refused")
	}
	for _, c := range []struct{ name, set, reset string }{
		{"a public-only member", "UPDATE nodes SET max_classification = 'public' WHERE id = $1", "UPDATE nodes SET max_classification = 'restricted' WHERE id = $1"},
		{"a silent member", "UPDATE nodes SET last_seen_at = now() - interval '2 minutes' WHERE id = $1", "UPDATE nodes SET last_seen_at = now() WHERE id = $1"},
		{"a draining member", "UPDATE nodes SET status = 'draining' WHERE id = $1", "UPDATE nodes SET status = 'active' WHERE id = $1"},
		{"a member never seen", "UPDATE nodes SET last_seen_at = NULL WHERE id = $1", "UPDATE nodes SET last_seen_at = now() WHERE id = $1"},
	} {
		exec(c.set, member)
		if eligible(c.name) {
			t.Errorf("a restricted request may run on a group with %s", c.name)
		}
		exec(c.reset, member)
	}
}
