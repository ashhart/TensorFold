package fleet_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

func TestLiveSnapshotCountsOnlyFreshStats(t *testing.T) {
	r := newRig(t)
	model, deployment, node, replica := r.fleetFixture()
	r.exec("INSERT INTO settings (installation_id, trust_domain, offline) VALUES ($1, 'starlight-test', false)", ids.New())
	stale := ids.New()
	r.exec("INSERT INTO replicas (id, deployment_id, deployment_revision, node_id, state, slots, epoch) VALUES ($1, $2, 1, $3, 'ready', 4, 1)", stale, deployment, node)
	r.exec(`UPDATE replicas SET live = '{"tokens_per_second": 42.5, "prompt_tokens_per_second": 900, "running": 3, "waiting": 1, "kv_usage": 0.6}',
		live_at = now() WHERE id = $1`, replica)
	r.exec(`UPDATE replicas SET live = '{"tokens_per_second": 1000, "running": 9}', live_at = now() - interval '20 seconds' WHERE id = $1`, stale)
	u, _ := directory.CreateUser(r.ctx, r.pool, directory.NewUser{Email: "oz@example.com", Source: "jit"})
	project := ids.New()
	r.exec("INSERT INTO projects (id, slug, name) VALUES ($1, 'eng', 'Eng')", project)
	r.exec(`INSERT INTO leases (id, request_id, gateway_id, model_id, replica_id, scopes, priority, deadline, principal_type, principal_id, project_id, client)
		VALUES ($1, 'req', $2, $3, $4, '{}', 'interactive', now() + interval '1 hour', 'user', $5, $6, 'connect/claude-code')`,
		ids.New(), ids.New(), model, replica, u.ID, project)
	r.exec(`INSERT INTO leases (id, request_id, gateway_id, model_id, replica_id, scopes, priority, deadline, heartbeat_at, principal_type, principal_id)
		VALUES ($1, 'dead', $2, $3, $4, '{}', 'batch', now() + interval '1 hour', now() - interval '5 minutes', 'user', $5)`, ids.New(), ids.New(), model, replica, u.ID)
	snap, err := fleet.Load(r.ctx, r.pool, fleet.Scope{Org: true, Usage: true, Nodes: true, Models: true})
	if err != nil {
		t.Fatal(err)
	}
	ti := snap.Tiles
	if ti.TokensPerSecond != 42.5 || ti.PromptTokensPerSecond != 900 || ti.InFlightLocal != 1 || ti.ActiveUsers != 1 || ti.NodesOnline != 1 ||
		ti.NodesTotal != 2 || ti.ModelsAvailable != 1 || ti.ModelsTotal != 1 {
		t.Fatalf("tiles %+v", ti)
	}
	if len(snap.Throughput) != 61 || snap.Throughput[60] != 42.5 || len(snap.Requests) != 60 {
		t.Fatalf("series lengths %d %d", len(snap.Throughput), len(snap.Requests))
	}
	if len(snap.InFlight) != 1 || snap.InFlight[0].Who != "oz@example.com" || snap.InFlight[0].Target != "studio-1" || snap.InFlight[0].Route != "local" {
		t.Fatalf("in flight %+v", snap.InFlight)
	}
	if len(snap.Connected) != 1 || snap.Connected[0].InFlight != 1 || snap.Connected[0].LastClient != "connect/claude-code" {
		t.Fatalf("connected %+v", snap.Connected)
	}
	var n fleet.NodeLive
	for _, x := range snap.Nodes {
		if x.Name == "studio-1" {
			n = x
		}
	}
	if n.State != "online" || n.TokensPerSecond != 42.5 || n.Running != 3 || n.KVUsage != 0.6 || !n.Fresh || n.Accelerators != "Apple M3 Ultra" {
		t.Fatalf("node live %+v", n)
	}
	if len(snap.Models) != 1 || snap.Models[0].Availability != "available" || snap.Models[0].Ready != 2 || snap.Models[0].TokensPerSecond != 42.5 {
		t.Fatalf("models %+v", snap.Models)
	}
	if !strings.Contains(strings.Join(ti.Warnings, "|"), "waiting for approval") {
		t.Fatalf("warnings %v", ti.Warnings)
	}
	r.exec("UPDATE nodes SET last_seen_at = now() - interval '5 minutes' WHERE id = $1", node)
	snap, _ = fleet.Load(r.ctx, r.pool, fleet.Scope{Org: true, Nodes: true, Models: true})
	if snap.Tiles.NodesOnline != 0 || snap.Models[0].Availability != "unavailable" || len(snap.InFlight) != 0 {
		t.Fatalf("stale node still counted: %+v %+v", snap.Tiles, snap.Models)
	}
}

func TestPersonalSnapshotHidesTheFleet(t *testing.T) {
	r := newRig(t)
	r.fleetFixture()
	u, _ := directory.CreateUser(r.ctx, r.pool, directory.NewUser{Email: "pat@example.com", Source: "jit"})
	project := ids.New()
	r.exec("INSERT INTO projects (id, slug, name) VALUES ($1, 'eng', 'Eng')", project)
	r.exec("INSERT INTO model_grants (id, model_id) SELECT $1, id FROM models", ids.New())
	snap, err := fleet.Load(r.ctx, r.pool, fleet.Scope{UserID: u.ID, ProjectIDs: []string{project}})
	if err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(snap)
	if snap.Tiles != nil || len(snap.Nodes) != 0 || strings.Contains(string(b), "studio-1") {
		t.Fatalf("personal view leaked fleet data: %s", b)
	}
	if snap.Personal == nil || len(snap.Personal.Projects) != 1 || len(snap.Personal.Projects[0].Models) != 1 {
		t.Fatalf("personal %+v", snap.Personal)
	}
}

func TestSpark(t *testing.T) {
	if got := fleet.Spark([]float64{0, 5, 10}, 100, 20); got != "0.0,19.0 50.0,10.0 100.0,1.0" {
		t.Fatalf("spark %q", got)
	}
	if fleet.Spark(nil, 1, 1) != "" || !strings.HasPrefix(fleet.Area([]float64{1, 2}, 10, 10), "0,10.0 ") {
		t.Fatal("empty or area shapes")
	}
}
