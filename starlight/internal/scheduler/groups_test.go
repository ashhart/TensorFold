package scheduler

import (
	"context"
	"strings"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

func gb10(id string) *node {
	return &node{id: id, pool: "spark", backend: "cuda", memory: 128 << 30, gpus: []wire.GPU{{Name: "NVIDIA GB10"}}, taken: map[string]bool{},
		cached: map[string]bool{}, ports: map[int]bool{}}
}

func TestGroupsSpanDistinctNodes(t *testing.T) {
	pair := deployment{id: "glm", pool: "spark", spec: wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/glm"}, Nodes: 2, TensorParallel: "glm", MemoryGiB: 200}}
	s := &snapshot{replicas: map[string][]replica{}, failedOn: map[string]map[string]bool{}, nodes: []*node{gb10("a"), gb10("b")}}
	seats := s.place(pair)
	if len(seats) != 2 || seats[0].node.id == seats[1].node.id || len(seats[0].devices) != 0 {
		t.Fatalf("a two-node GLM group placed as %+v", seats)
	}
	if s.nodes[0].reserved != 100<<30 || s.nodes[1].reserved != 100<<30 {
		t.Fatalf("each node should hold half of 200 GiB: %d %d", s.nodes[0].reserved>>30, s.nodes[1].reserved>>30)
	}
	if again := (&snapshot{replicas: map[string][]replica{}, failedOn: map[string]map[string]bool{}, nodes: []*node{gb10("a")}}).place(pair); again != nil {
		t.Fatalf("a two-node group fitted on one node: %+v", again)
	}
	wide := pair
	wide.pool, wide.spec.GPUs, wide.spec.MemoryGiB = "gpu", 4, 2000
	b := &snapshot{replicas: map[string][]replica{}, failedOn: map[string]map[string]bool{}, nodes: []*node{b300("x"), b300("y")}}
	got := b.place(wide)
	if len(got) != 2 || len(got[0].devices) != 4 || len(got[1].devices) != 4 || got[0].node.id == got[1].node.id {
		t.Fatalf("2 nodes x 4 GPUs placed as %+v", got)
	}
	mac := &node{id: "m", pool: "spark", backend: "metal", memory: 512 << 30, taken: map[string]bool{}, cached: map[string]bool{}}
	if seats := (&snapshot{replicas: map[string][]replica{}, failedOn: map[string]map[string]bool{}, nodes: []*node{mac, gb10("a")}}).place(pair); seats != nil {
		t.Fatalf("a CUDA rank scheme landed on a Mac: %+v", seats)
	}
}

func TestPassPlacesAGroupAndStopsItWhenAMemberFails(t *testing.T) {
	pool := testutil.DB(t)
	ctx := context.Background()
	model, dep := ids.New(), ids.New()
	exec := func(sql string, args ...any) {
		if _, err := pool.Exec(ctx, sql, args...); err != nil {
			t.Fatal(err)
		}
	}
	exec("INSERT INTO models (id, name, display_name) VALUES ($1, 'glm-flash', 'GLM')", model)
	exec(`INSERT INTO deployments (id, model_id, name, pool, spec, mode, min_replicas) VALUES ($1, $2, 'glm-pair', 'spark',
		'{"model":{"kind":"path","path":"/glm"},"nodes":2,"tensor_parallel":"glm","memory_gib":200}', 'warm', 1)`, dep, model)
	for _, name := range []string{"spark-1", "spark-2"} {
		exec(`INSERT INTO nodes (id, name, pool, status, address, last_seen_at, facts) VALUES ($1, $2, 'spark', 'active', $3, now(),
			('{"memory_bytes": 137438953472, "backend": "cuda", "gpus": [{"name": "NVIDIA GB10", "memory_bytes": 0}], "rank_host": "10.0.0.' || $4 || '"}')::jsonb)`,
			ids.New(), name, name+".example:7443", name[len(name)-1:])
	}
	c := &Controller{Pool: pool, Instance: ids.New()}
	epoch, ok, err := (&Elector{Pool: pool, Instance: c.Instance}).acquire(ctx)
	if err != nil || !ok {
		t.Fatal(err)
	}
	if err := c.Pass(ctx, epoch); err != nil {
		t.Fatal(err)
	}
	var replicaID, lead string
	var port int
	if err := pool.QueryRow(ctx, "SELECT id, node_id, rank_port FROM replicas WHERE deployment_id = $1", dep).Scan(&replicaID, &lead, &port); err != nil {
		t.Fatal(err)
	}
	var member string
	if err := pool.QueryRow(ctx, "SELECT node_id FROM replica_members WHERE replica_id = $1 AND member = 1", replicaID).Scan(&member); err != nil {
		t.Fatal(err)
	}
	if member == lead || port != rankBase {
		t.Fatalf("group lead %s member %s port %d", lead, member, port)
	}
	exec("UPDATE replica_members SET state = 'failed', error = 'nccl: peer closed', updated_at = now() - interval '1 minute' WHERE replica_id = $1", replicaID)
	if err := c.Pass(ctx, epoch); err != nil {
		t.Fatal(err)
	}
	var desired, why string
	_ = pool.QueryRow(ctx, "SELECT desired, error FROM replicas WHERE id = $1", replicaID).Scan(&desired, &why)
	if desired != "stopped" || !strings.Contains(why, "rank member 1 failed") {
		t.Fatalf("a group with a failed member is %s (%s)", desired, why)
	}
}

func TestAMDNodesHoldWholeGPUsAndSkipTheirIGPU(t *testing.T) {
	n := &node{id: "mi", pool: "amd", backend: "hip", memory: 1 << 40, taken: map[string]bool{}, cached: map[string]bool{}, ports: map[int]bool{},
		gpus: []wire.GPU{{Index: 0, Name: "AMD Radeon Graphics"}, {Index: 1, Name: "AMD Radeon PRO W7800", MemoryBytes: 48 << 30}, {Index: 2, Name: "AMD Radeon PRO W7800", MemoryBytes: 48 << 30}}}
	s := &snapshot{replicas: map[string][]replica{}, failedOn: map[string]map[string]bool{}, nodes: []*node{n}}
	d := deployment{id: "q", pool: "amd", spec: wire.Spec{Model: wire.Artifact{Kind: "path", Path: "/q"}, GPUs: 2, TensorParallel: "rocm", MemoryGiB: 80}}
	seats := s.place(d)
	if len(seats) != 1 || strings.Join(seats[0].devices, ",") != "1,2" {
		t.Fatalf("a two-GPU ROCm replica on a workstation with an iGPU placed as %+v", seats)
	}
}
