package scheduler

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

func ago(d time.Duration) *time.Time {
	t := time.Now().Add(-d)
	return &t
}

func TestPlanWarmOnDemandIdleAndRollout(t *testing.T) {
	spec := wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/m"}, MemoryGiB: 40}
	nodes := []*node{{id: "n1", pool: "default", memory: 64 << 30}, {id: "n2", pool: "default", memory: 256 << 30, cached: map[string]bool{"path:/m": true}}}
	d := deployment{id: "d1", modelID: "m1", pool: "default", mode: "warm", revision: 1, spec: spec, min: 1, max: 2, idle: 600}
	s := &snapshot{nodes: nodes, replicas: map[string][]replica{}, failedOn: map[string]map[string]bool{}, queue: map[string]int{}}
	acts := plan(s, d, time.Now())
	if len(acts) != 1 || acts[0].start == nil || acts[0].start.seats[0].node.id != "n2" {
		t.Fatalf("warm start = %+v", acts)
	}
	d.mode, d.min = "on_demand", 0
	s.nodes[1].reserved = 0
	if acts := plan(s, d, time.Now()); len(acts) != 0 {
		t.Fatalf("idle on-demand deployment started: %+v", acts)
	}
	s.queue["m1"] = 3
	if acts := plan(s, d, time.Now()); len(acts) != 1 || acts[0].start == nil {
		t.Fatalf("demand did not load the model: %+v", acts)
	}
	s.queue["m1"] = 0
	s.replicas["d1"] = []replica{{id: "r1", deployment: "d1", node: "n2", desired: "running", state: "ready", revision: 1, readyAt: ago(time.Hour), lastRequest: ago(20 * time.Minute)}}
	if acts := plan(s, d, time.Now()); len(acts) != 1 || acts[0].stop != "r1" {
		t.Fatalf("idle replica kept: %+v", acts)
	}
	s.replicas["d1"][0].lastRequest = ago(time.Minute)
	if acts := plan(s, d, time.Now()); len(acts) != 0 {
		t.Fatalf("recently used replica stopped: %+v", acts)
	}
	d.mode, d.min, d.revision = "warm", 1, 2
	acts = plan(s, d, time.Now())
	if len(acts) != 1 || acts[0].start == nil {
		t.Fatalf("rollout must surge a new revision first: %+v", acts)
	}
	s.replicas["d1"] = append(s.replicas["d1"], replica{id: "r2", deployment: "d1", node: "n1", desired: "running", state: "ready", revision: 2, readyAt: ago(time.Minute)})
	acts = plan(s, d, time.Now())
	if len(acts) != 1 || acts[0].stop != "r1" {
		t.Fatalf("old revision not retired after the new one became ready: %+v", acts)
	}
}

func TestPlacementConstraints(t *testing.T) {
	spec := wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/m"}, Backend: "cuda", MemoryGiB: 100}
	s := &snapshot{replicas: map[string][]replica{}, failedOn: map[string]map[string]bool{"d1": {"gpu2": true}}, nodes: []*node{
		{id: "mac", pool: "gpu", backend: "metal", memory: 512 << 30},
		{id: "small", pool: "gpu", backend: "cuda", memory: 64 << 30},
		{id: "gpu2", pool: "gpu", backend: "cuda", memory: 1 << 40},
		{id: "other", pool: "lab", backend: "cuda", memory: 1 << 40},
		{id: "gpu1", pool: "gpu", backend: "cuda", memory: 1 << 40, labels: map[string]string{"rack": "a"}},
	}}
	d := deployment{id: "d1", pool: "gpu", spec: spec, selector: map[string]string{"rack": "a"}}
	if seats := s.place(d); len(seats) != 1 || seats[0].node.id != "gpu1" {
		t.Fatalf("placed on %+v", seats)
	}
	d.selector = map[string]string{"rack": "b"}
	if seats := s.place(d); seats != nil {
		t.Fatalf("placed despite the selector: %s", seats[0].node.id)
	}
}

func b300(id string) *node {
	n := &node{id: id, pool: "gpu", backend: "cuda", memory: 4 << 40, taken: map[string]bool{}, cached: map[string]bool{}}
	for i := 0; i < 8; i++ {
		n.gpus = append(n.gpus, wire.GPU{Name: "NVIDIA B300", MemoryBytes: 288000 << 20, UUID: fmt.Sprintf("GPU-%s-%d", id, i)})
	}
	return n
}

func TestGPUPlacementSpreadsPacksAndPins(t *testing.T) {
	gb10 := &node{id: "gb10", pool: "gpu", backend: "cuda", memory: 128 << 30, gpus: []wire.GPU{{Name: "NVIDIA GB10"}}, taken: map[string]bool{}}
	s := &snapshot{replicas: map[string][]replica{}, failedOn: map[string]map[string]bool{}, nodes: []*node{b300("a"), b300("b"), gb10}}
	one := deployment{id: "d1", pool: "gpu", spec: wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/m"}, MemoryGiB: 100}}
	place := func(d deployment) string {
		seats := s.place(d)
		if seats == nil {
			return ""
		}
		s.replicas[d.id] = append(s.replicas[d.id], replica{id: ids.New(), deployment: d.id, node: seats[0].node.id, desired: "running", state: "ready"})
		return seats[0].node.id + ":" + strings.Join(seats[0].devices, ",")
	}
	for i, want := range []string{"a:GPU-a-0", "b:GPU-b-0", "gb10:", "a:GPU-a-1"} {
		if got := place(one); got != want {
			t.Fatalf("replica %d of a one-GPU model placed %q, want %q", i, got, want)
		}
	}
	tp := deployment{id: "d2", pool: "gpu", spec: wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/big"}, MemoryGiB: 1000, GPUs: 4, TensorParallel: "glm"}}
	if got := place(tp); got != "a:GPU-a-2,GPU-a-3,GPU-a-4,GPU-a-5" {
		t.Fatalf("four-rank model placed %q; best fit packs the fuller node", got)
	}
	huge := deployment{id: "d3", pool: "gpu", spec: wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/huge"}, MemoryGiB: 300}}
	if got := place(huge); got != "" {
		t.Fatalf("300 GiB on one 281 GiB GPU placed %q", got)
	}
	small := deployment{id: "d4", pool: "gpu", spec: wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/s"}, MemoryGiB: 20, Backend: "cuda"}}
	s.nodes = []*node{gb10}
	if got := place(small); got != "gb10:" {
		t.Fatalf("unified-memory GB10 placed %q; it shares memory and pins nothing", got)
	}
	if got := place(tp); got != "" {
		t.Fatalf("a multi-rank model landed on a single unified GPU: %q", got)
	}
}

func TestFullNodeRollsOutOneReplicaAtATime(t *testing.T) {
	n := b300("a")
	n.gpus = n.gpus[:1]
	n.taken["GPU-a-0"] = true
	s := &snapshot{nodes: []*node{n}, failedOn: map[string]map[string]bool{}, queue: map[string]int{}, replicas: map[string][]replica{
		"d1": {{id: "old", deployment: "d1", node: "a", desired: "running", state: "ready", revision: 1, devices: []string{"GPU-a-0"}}}}}
	d := deployment{id: "d1", pool: "gpu", mode: "warm", revision: 2, min: 1, max: 1, spec: wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/m"}}}
	wide, elsewhere := d, d
	wide.spec.GPUs, wide.spec.TensorParallel = 2, "glm"
	elsewhere.selector = map[string]string{"rack": "b"}
	for name, unplaceable := range map[string]deployment{"a two-GPU revision on a one-GPU node": wide, "a selector no node matches": elsewhere} {
		if acts := plan(s, unplaceable, time.Now()); len(acts) != 0 {
			t.Fatalf("%s stopped the old replica for nothing: %+v", name, acts)
		}
	}
	if acts := plan(s, d, time.Now()); len(acts) != 1 || acts[0].stop != "old" {
		t.Fatalf("a full node wedged the rollout: %+v", acts)
	}
	s.replicas["d1"][0].desired = "stopped"
	if acts := plan(s, d, time.Now()); len(acts) != 0 {
		t.Fatalf("acted again while the old replica still holds its GPU: %+v", acts)
	}
}

func TestElectionAndFencing(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	a, b := &Elector{Pool: pool, Instance: ids.New()}, &Elector{Pool: pool, Instance: ids.New()}
	ea, ok, err := a.acquire(ctx)
	if err != nil || !ok {
		t.Fatalf("a acquire: %v %v", ok, err)
	}
	if _, ok, _ := b.acquire(ctx); ok {
		t.Fatal("two leaders at once")
	}
	if again, ok, _ := a.acquire(ctx); !ok || again != ea {
		t.Fatal("renewal changed the epoch")
	}
	if _, err := pool.Exec(ctx, "UPDATE controller_leader SET expires_at = now() - interval '1 second'"); err != nil {
		t.Fatal(err)
	}
	eb, ok, _ := b.acquire(ctx)
	if !ok || eb != ea+1 {
		t.Fatalf("takeover epoch %d after %d", eb, ea)
	}
	err = db.Tx(ctx, pool, func(tx pgx.Tx) error { return Fence(ctx, tx, a.Instance, ea) })
	if err != ErrFenced {
		t.Fatalf("stale leader not fenced: %v", err)
	}
}

func TestPassCreatesReplicaAndBumpsNode(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	model, dep, nodeID := ids.New(), ids.New(), ids.New()
	exec := func(sql string, args ...any) {
		if _, err := pool.Exec(ctx, sql, args...); err != nil {
			t.Fatal(err)
		}
	}
	exec("INSERT INTO models (id, name, display_name) VALUES ($1, 'qwen3.8-27b', 'Qwen')", model)
	exec(`INSERT INTO deployments (id, model_id, name, spec, mode, min_replicas) VALUES ($1, $2, 'qwen', '{"model":{"kind":"path","path":"/m"},"parallel":8}', 'warm', 1)`, dep, model)
	exec(`INSERT INTO nodes (id, name, status, last_seen_at, facts) VALUES ($1, 'studio', 'active', now(), '{"memory_bytes": 274877906944, "backend": "metal"}')`, nodeID)
	c := &Controller{Pool: pool, Instance: ids.New()}
	e := &Elector{Pool: pool, Instance: c.Instance}
	epoch, ok, err := e.acquire(ctx)
	if err != nil || !ok {
		t.Fatal(err)
	}
	if err := c.Pass(ctx, epoch); err != nil {
		t.Fatal(err)
	}
	var slots int
	var gen int64
	if err := pool.QueryRow(ctx, "SELECT r.slots, n.sync_generation FROM replicas r JOIN nodes n ON n.id = r.node_id WHERE r.deployment_id = $1", dep).Scan(&slots, &gen); err != nil {
		t.Fatal(err)
	}
	if slots != 8 || gen != 2 {
		t.Fatalf("slots %d generation %d", slots, gen)
	}
	if err := c.Pass(ctx, epoch+1); err != ErrFenced {
		t.Fatalf("pass with a stale epoch: %v", err)
	}
}

func TestPassPinsAGPUPerReplica(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	model, dep, nodeID := ids.New(), ids.New(), ids.New()
	exec := func(sql string, args ...any) {
		if _, err := pool.Exec(ctx, sql, args...); err != nil {
			t.Fatal(err)
		}
	}
	exec("INSERT INTO models (id, name, display_name) VALUES ($1, 'qwen3.8-27b', 'Qwen')", model)
	exec(`INSERT INTO deployments (id, model_id, name, spec, mode, min_replicas, max_replicas) VALUES ($1, $2, 'qwen', '{"model":{"kind":"path","path":"/m"},"memory_gib":60}', 'warm', 2, 2)`, dep, model)
	exec(`INSERT INTO nodes (id, name, status, last_seen_at, facts) VALUES ($1, 'b300', 'active', now(), '{"memory_bytes": 4398046511104, "backend": "cuda",
		"gpus": [{"name": "NVIDIA B300", "memory_bytes": 301989888000, "uuid": "GPU-0"}, {"name": "NVIDIA B300", "memory_bytes": 301989888000, "uuid": "GPU-1"}]}')`, nodeID)
	c := &Controller{Pool: pool, Instance: ids.New()}
	epoch, ok, err := (&Elector{Pool: pool, Instance: c.Instance}).acquire(ctx)
	if err != nil || !ok {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		if err := c.Pass(ctx, epoch); err != nil {
			t.Fatal(err)
		}
	}
	var pinned []string
	if err := pool.QueryRow(ctx, "SELECT array_agg(d ORDER BY d) FROM replicas, unnest(devices) d WHERE deployment_id = $1", dep).Scan(&pinned); err != nil {
		t.Fatal(err)
	}
	if strings.Join(pinned, ",") != "GPU-0,GPU-1" {
		t.Fatalf("two replicas on one B300 hold %v; each needs its own GPU", pinned)
	}
}
