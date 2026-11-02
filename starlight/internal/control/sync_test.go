package control

import (
	"context"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/scheduler"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
	"github.com/ashhart/TensorFold/starlight/internal/trust"
)

func TestRoutineSyncsKeepTheFailureTimersRunning(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	exec := func(sql string, args ...any) {
		t.Helper()
		if _, err := pool.Exec(ctx, sql, args...); err != nil {
			t.Fatalf("%s: %v", sql, err)
		}
	}
	model, dep, lead, member, grp, solo, soloNode := ids.New(), ids.New(), ids.New(), ids.New(), ids.New(), ids.New(), ids.New()
	exec("INSERT INTO settings (installation_id, trust_domain) VALUES ($1, 'starlight-test')", ids.New())
	exec("INSERT INTO models (id, name, display_name) VALUES ($1, 'glm-flash', 'GLM')", model)
	exec(`INSERT INTO deployments (id, model_id, name, pool, spec, mode, min_replicas, max_replicas) VALUES ($1, $2, 'glm-pair', 'spark',
		'{"model":{"kind":"path","path":"/glm"},"nodes":2,"tensor_parallel":"glm","memory_gib":160}', 'warm', 2, 2)`, dep, model)
	for _, n := range []string{lead, member, soloNode} {
		exec(`INSERT INTO nodes (id, name, pool, status, address, last_seen_at) VALUES ($1, $2, 'spark', 'active', 'x:7443', now())`, n, "n"+n[len(n)-6:])
	}
	exec(`INSERT INTO replicas (id, deployment_id, deployment_revision, spec, node_id, state, slots, epoch, rank_port, updated_at) VALUES
		($1, $2, 1, '{"model":{"kind":"path","path":"/glm"},"nodes":2,"tensor_parallel":"glm"}', $3, 'starting', 4, 1, 29500, now())`, grp, dep, lead)
	exec(`INSERT INTO replica_members (replica_id, member, node_id, state, updated_at) VALUES ($1, 1, $2, 'failed', now() - interval '1 minute')`, grp, member)
	exec(`INSERT INTO replicas (id, deployment_id, deployment_revision, spec, node_id, state, slots, epoch, updated_at) VALUES
		($1, $2, 1, '{"model":{"kind":"path","path":"/glm"}}', $3, 'failed', 4, 1, now() - interval '5 minutes')`, solo, dep, soloNode)
	s := &Server{Pool: pool, Trust: &trust.Trust{}}
	for _, c := range []struct{ node, replica, state string }{{member, grp, "failed"}, {lead, grp, "starting"}, {soloNode, solo, "failed"}} {
		req := wire.SyncRequest{Replicas: []wire.ReplicaStatus{{ReplicaID: c.replica, Generation: 1, State: c.state, Error: "x"}}}
		if err := s.observe(ctx, c.node, req); err != nil {
			t.Fatal(err)
		}
	}
	inst := ids.New()
	var epoch int64
	if err := pool.QueryRow(ctx, `UPDATE controller_leader SET holder = $1, epoch = epoch + 1, expires_at = now() + interval '15 seconds' WHERE id
		RETURNING epoch`, inst).Scan(&epoch); err != nil {
		t.Fatal(err)
	}
	if err := (&scheduler.Controller{Pool: pool, Instance: inst}).Pass(ctx, epoch); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{grp, solo} {
		var desired string
		if err := pool.QueryRow(ctx, "SELECT desired FROM replicas WHERE id = $1", id).Scan(&desired); err != nil || desired != "stopped" {
			t.Fatalf("replica %s stayed %q after its nodes reported the same failure again: %v", id, desired, err)
		}
	}
	exec("UPDATE replicas SET state = 'stopped' WHERE id = $1", grp)
	exec("UPDATE replica_members SET state = 'draining' WHERE replica_id = $1", grp)
	resp, err := s.desired(ctx, member)
	if err != nil || len(resp.Assignments) != 1 || resp.Assignments[0].ReplicaID != grp || resp.Assignments[0].Desired != "stopped" {
		t.Fatalf("a member still draining stopped hearing about its group once rank 0 stopped: %+v %v", resp.Assignments, err)
	}
}
