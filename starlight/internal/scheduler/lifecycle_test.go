package scheduler

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/audit"
	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

// fleetDB is a migrated database with this test as the elected controller.
type fleetDB struct {
	t     *testing.T
	ctx   context.Context
	pool  *pgxpool.Pool
	c     *Controller
	epoch int64
}

func newFleetDB(t *testing.T) *fleetDB {
	pool := testutil.DB(t)
	ctx := context.Background()
	c := &Controller{Pool: pool, Instance: ids.New()}
	epoch, ok, err := (&Elector{Pool: pool, Instance: c.Instance}).acquire(ctx)
	if err != nil || !ok {
		t.Fatal(err)
	}
	return &fleetDB{t: t, ctx: ctx, pool: pool, c: c, epoch: epoch}
}

func (f *fleetDB) exec(sql string, args ...any) {
	f.t.Helper()
	if _, err := f.pool.Exec(f.ctx, sql, args...); err != nil {
		f.t.Fatalf("%s: %v", sql, err)
	}
}

func (f *fleetDB) pass() {
	f.t.Helper()
	if err := f.c.Pass(f.ctx, f.epoch); err != nil {
		f.t.Fatal(err)
	}
}

// sparkPair deploys a two-node GLM group over three GB10s, places it, and marks both of its seats ready.
func (f *fleetDB) sparkPair() (dep, replica, lead, member string) {
	model := ids.New()
	dep = ids.New()
	f.exec("INSERT INTO models (id, name, display_name) VALUES ($1, 'glm-flash', 'GLM')", model)
	f.exec(`INSERT INTO deployments (id, model_id, name, pool, spec, mode, min_replicas) VALUES ($1, $2, 'glm-pair', 'spark',
		'{"model":{"kind":"path","path":"/glm"},"nodes":2,"tensor_parallel":"glm","backend":"cuda","memory_gib":160}', 'warm', 1)`, dep, model)
	for i, name := range []string{"spark-1", "spark-2", "spark-3"} {
		f.exec(`INSERT INTO nodes (id, name, pool, status, address, last_seen_at, facts) VALUES ($1, $2, 'spark', 'active', $3, now(),
			('{"memory_bytes": 137438953472, "backend": "cuda", "gpus": [{"name": "NVIDIA GB10", "memory_bytes": 0}], "rank_host": "10.0.0.' || $4 || '"}')::jsonb)`,
			ids.New(), name, name+".example:7443", fmt.Sprint(i+1))
	}
	f.pass()
	if err := f.pool.QueryRow(f.ctx, `SELECT r.id, r.node_id, m.node_id FROM replicas r JOIN replica_members m ON m.replica_id = r.id
		WHERE r.deployment_id = $1`, dep).Scan(&replica, &lead, &member); err != nil {
		f.t.Fatal(err)
	}
	f.exec("UPDATE replicas SET state = 'ready', ready_at = now() WHERE id = $1", replica)
	f.exec("UPDATE replica_members SET state = 'ready' WHERE replica_id = $1", replica)
	return dep, replica, lead, member
}

func (f *fleetDB) desired(replica string) string {
	f.t.Helper()
	var d string
	if err := f.pool.QueryRow(f.ctx, "SELECT desired FROM replicas WHERE id = $1", replica).Scan(&d); err != nil {
		f.t.Fatal(err)
	}
	return d
}

func TestALostLeadStopsItsGroupAndFreesTheFollower(t *testing.T) {
	f := newFleetDB(t)
	dep, replica, lead, _ := f.sparkPair()
	f.exec("UPDATE nodes SET last_seen_at = now() - interval '2 minutes' WHERE id = $1", lead)
	f.pass()
	if f.desired(replica) != "stopped" {
		t.Fatal("a group whose lead was lost kept its follower running")
	}
	f.exec("UPDATE replica_members SET state = 'stopped' WHERE replica_id = $1", replica)
	f.pass()
	var placed int
	if err := f.pool.QueryRow(f.ctx, "SELECT count(*) FROM replicas WHERE deployment_id = $1 AND id <> $2 AND desired = 'running'", dep, replica).
		Scan(&placed); err != nil || placed != 1 {
		t.Fatalf("no replacement on the two healthy nodes once the follower stopped: %d %v", placed, err)
	}
}

func TestDrainingAMemberNodeStopsItsGroup(t *testing.T) {
	f := newFleetDB(t)
	_, replica, _, member := f.sparkPair()
	var name string
	if err := f.pool.QueryRow(f.ctx, "SELECT name FROM nodes WHERE id = $1", member).Scan(&name); err != nil {
		t.Fatal(err)
	}
	svc := &fleet.Service{Pool: f.pool, Keys: keyring.ForTest()}
	if err := svc.Do(f.ctx, audit.Event{ActorType: "user", ActorID: ids.New(), ActorLabel: "admin@example.com"}, func(tx *fleet.Tx) error {
		_, err := tx.SetNodeStatus(f.ctx, name, fleet.NodeDrain)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if f.desired(replica) != "stopped" {
		t.Fatal("draining a node that runs a group's rank left the group running")
	}
	var gen int64
	if err := f.pool.QueryRow(f.ctx, "SELECT sync_generation FROM nodes WHERE id = $1", member).Scan(&gen); err != nil || gen == 0 {
		t.Fatalf("the member node was not woken: generation %d %v", gen, err)
	}
}

func TestAReplacementGroupWaitsForItsLeadsRendezvousPorts(t *testing.T) {
	f := newFleetDB(t)
	model, dep := ids.New(), ids.New()
	f.exec("INSERT INTO models (id, name, display_name) VALUES ($1, 'glm-flash', 'GLM')", model)
	f.exec(`INSERT INTO deployments (id, model_id, name, pool, spec, mode, min_replicas) VALUES ($1, $2, 'glm-pair', 'gpu',
		'{"model":{"kind":"path","path":"/glm"},"nodes":2,"tensor_parallel":"glm","backend":"cuda","memory_gib":100}', 'warm', 1)`, dep, model)
	for i, name := range []string{"gpu-a", "gpu-b", "gpu-c"} {
		cached := "{}"
		if name == "gpu-a" {
			cached = "{path:/glm}"
		}
		f.exec(`INSERT INTO nodes (id, name, pool, status, address, last_seen_at, cached_artifacts, facts) VALUES ($1, $2, 'gpu', 'active', $3, now(), $4::text[],
			('{"memory_bytes": 1099511627776, "backend": "cuda", "gpus": [{"index":0,"uuid":"GPU-' || $5 || '000","name": "NVIDIA H200", "memory_bytes": 150000000000},
			{"index":1,"uuid":"GPU-' || $5 || '111","name": "NVIDIA H200", "memory_bytes": 150000000000}]}')::jsonb)`, ids.New(), name, name+".example:7443", cached, fmt.Sprint(i))
	}
	f.pass()
	var first string
	if err := f.pool.QueryRow(f.ctx, "SELECT id FROM replicas WHERE deployment_id = $1", dep).Scan(&first); err != nil {
		t.Fatal(err)
	}
	f.exec("UPDATE replicas SET state = 'ready', ready_at = now() WHERE id = $1", first)
	f.exec("UPDATE replica_members SET state = 'ready' WHERE replica_id = $1", first)
	f.exec("UPDATE nodes SET last_seen_at = now() - interval '2 minutes' WHERE id = (SELECT node_id FROM replica_members WHERE replica_id = $1)", first)
	f.pass()
	var clash int
	if err := f.pool.QueryRow(f.ctx, `SELECT count(*) FROM replicas a JOIN replicas b ON a.node_id = b.node_id AND a.rank_port = b.rank_port AND a.id < b.id
		WHERE a.deployment_id = $1 AND b.deployment_id = $1 AND a.state NOT IN ('stopped', 'failed', 'lost') AND b.state NOT IN ('stopped', 'failed', 'lost')`,
		dep).Scan(&clash); err != nil || clash > 0 {
		t.Fatalf("a new group shares its lead's rendezvous block with a rank 0 still running: %d %v", clash, err)
	}
}

func TestUnifiedRolloutWaitsForTheOldGroupToRelease(t *testing.T) {
	spec := wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/glm"}, Nodes: 2, TensorParallel: "glm", MemoryGiB: 200}
	d := deployment{id: "d", pool: "spark", mode: "warm", revision: 2, min: 1, max: 1, spec: spec}
	s := &snapshot{nodes: []*node{gb10("a"), gb10("b")}, failedOn: map[string]map[string]bool{}, queue: map[string]int{},
		replicas: map[string][]replica{"d": {{id: "old", deployment: "d", node: "a", desired: "stopped", state: "draining", revision: 1,
			share: 100 << 30, port: 29500, members: []held{{node: "b", state: "draining"}}}}}}
	s.account()
	for _, a := range plan(s, d, time.Now()) {
		if a.start != nil {
			t.Fatalf("a new 200 GiB group started on two 128 GiB GB10s while the old one still drains there: %+v", a.start.seats)
		}
	}
}

func h200(id string) *node {
	return &node{id: id, pool: "gpu", backend: "cuda", memory: 1 << 40, taken: map[string]bool{}, cached: map[string]bool{}, ports: map[int]bool{},
		gpus: []wire.GPU{{Name: "NVIDIA H200", UUID: "GPU-" + id + "0000", MemoryBytes: 140 << 30}}}
}

func TestStopFirstRolloutWaitsForEveryMemberSeat(t *testing.T) {
	spec := wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/glm"}, Nodes: 2, TensorParallel: "glm", MemoryGiB: 100}
	d := deployment{id: "d", pool: "gpu", mode: "warm", revision: 2, min: 2, max: 2, spec: spec}
	s := &snapshot{nodes: []*node{h200("a"), h200("b"), h200("c"), h200("e")}, failedOn: map[string]map[string]bool{}, queue: map[string]int{},
		replicas: map[string][]replica{"d": {
			{id: "old1", deployment: "d", node: "a", desired: "stopped", state: "stopped", revision: 1, devices: []string{"GPU-a0000"},
				members: []held{{node: "b", state: "draining", devices: []string{"GPU-b0000"}}}},
			{id: "old2", deployment: "d", node: "c", desired: "running", state: "ready", revision: 1, devices: []string{"GPU-c0000"},
				members: []held{{node: "e", state: "ready", devices: []string{"GPU-e0000"}}}},
		}}}
	s.account()
	for _, a := range plan(s, d, time.Now()) {
		if a.stop == "old2" {
			t.Fatal("old2 was stopped while old1's member still holds its GPU")
		}
	}
}

func TestStopFirstRolloutWaitsForTheNewReplicaToBeReady(t *testing.T) {
	d := deployment{id: "d", pool: "gpu", mode: "warm", revision: 2, min: 2, max: 2, spec: wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/m"}, MemoryGiB: 100}}
	for _, newState := range []string{"starting", "failed"} {
		s := &snapshot{nodes: []*node{h200("a"), h200("b")}, failedOn: map[string]map[string]bool{}, queue: map[string]int{},
			replicas: map[string][]replica{"d": {
				{id: "old1", deployment: "d", node: "a", desired: "stopped", state: "stopped", revision: 1, devices: []string{"GPU-a0000"}},
				{id: "new1", deployment: "d", node: "a", desired: "running", state: newState, revision: 2, devices: []string{"GPU-a0000"}},
				{id: "old2", deployment: "d", node: "b", desired: "running", state: "ready", revision: 1, devices: []string{"GPU-b0000"}},
			}}}
		s.account()
		for _, a := range plan(s, d, time.Now()) {
			if a.stop == "old2" {
				t.Fatalf("the last ready replica was stopped while the new revision's replica is %s", newState)
			}
		}
	}
}

func TestANodeThatLostItsGPUProbeIsNotUnifiedMemory(t *testing.T) {
	box := &node{id: "h200", pool: "gpu", memory: 2 << 40, taken: map[string]bool{}, cached: map[string]bool{}, ports: map[int]bool{}}
	s := &snapshot{nodes: []*node{box}, failedOn: map[string]map[string]bool{}, queue: map[string]int{}, replicas: map[string][]replica{
		"other": {{id: "pinned", deployment: "other", node: "h200", desired: "running", state: "ready", devices: []string{"GPU-0"}, share: 100 << 30}}}}
	s.account()
	d := deployment{id: "d", pool: "gpu", mode: "warm", revision: 1, min: 1, max: 1, spec: wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/m"}, MemoryGiB: 100}}
	if seats := s.place(d); seats != nil {
		t.Fatalf("placed an unpinned replica on a GPU box that reported no GPUs: %+v", seats[0])
	}
}
