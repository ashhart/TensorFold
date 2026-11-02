package fleetpages_test

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

func TestLiveFeedShapeAndPermissions(t *testing.T) {
	r := newRig(t)
	owner := r.user("owner@example.com", "owner", "")
	r.exec(`INSERT INTO leases (id, request_id, gateway_id, model_id, replica_id, scopes, priority, deadline, principal_type, principal_id, project_id, client)
		VALUES ($1, 'req-1', $2, $3, $4, '{}', 'interactive', now() + interval '1 hour', 'user', $5, $6, 'connect/claude-code')`,
		ids.New(), ids.New(), r.model, r.replica, owner.user.ID, r.project)
	resp, body := owner.do("GET", "/console/live.json", nil)
	if resp.StatusCode != http.StatusOK || resp.Header.Get("Content-Type") != "application/json" || resp.Header.Get("Cache-Control") != "no-store" {
		t.Fatalf("live.json %d %v", resp.StatusCode, resp.Header)
	}
	var snap fleet.Snapshot
	if err := json.Unmarshal([]byte(body), &snap); err != nil {
		t.Fatal(err)
	}
	ti := snap.Tiles
	if !snap.Org || ti == nil || ti.TokensPerSecond != 87.5 || ti.InFlightLocal != 1 || ti.NodesOnline != 1 || ti.NodesTotal != 2 || ti.ModelsAvailable != 1 {
		t.Fatalf("tiles %+v", ti)
	}
	if len(snap.Throughput) != 61 || snap.Throughput[60] != 87.5 || len(snap.Requests) != 60 {
		t.Fatalf("series %d %d", len(snap.Throughput), len(snap.Requests))
	}
	if len(snap.InFlight) != 1 || snap.InFlight[0].Who != "owner@example.com" || snap.InFlight[0].Target != "studio-1" || snap.InFlight[0].Client != "connect/claude-code" {
		t.Fatalf("in flight %+v", snap.InFlight)
	}
	if len(snap.Connected) != 1 || len(snap.Nodes) != 2 || len(snap.Models) != 1 || snap.Models[0].Availability != "available" {
		t.Fatalf("connected %d nodes %d models %+v", len(snap.Connected), len(snap.Nodes), snap.Models)
	}
	for _, key := range []string{`"tokens_per_second"`, `"in_flight_local"`, `"models_available"`, `"kv_usage"`, `"availability"`, `"age_seconds"`} {
		if !strings.Contains(body, key) {
			t.Errorf("live.json is missing %s", key)
		}
	}
	budget := r.user("budget@example.com", "budget_admin", "")
	_, body = budget.get("/console/live.json")
	snap = fleet.Snapshot{}
	_ = json.Unmarshal([]byte(body), &snap)
	if len(snap.Nodes) != 0 || len(snap.Models) != 1 || len(snap.InFlight) != 1 || strings.Contains(body, `"name":"studio-1"`) {
		t.Fatalf("budget admin feed: %s", body)
	}
	if _, page := budget.get("/console"); strings.Contains(page, `data-node-id="`+r.node+`"`) || strings.Contains(page, "Take offline") {
		t.Fatal("the overview shows nodes to a role without nodes.read")
	}
}

func TestStaleStatsAreNotCounted(t *testing.T) {
	r := newRig(t)
	owner := r.user("owner@example.com", "owner", "")
	if _, page := owner.get("/console/nodes"); !strings.Contains(page, `data-col="tps">87.5<`) || !strings.Contains(page, `data-col="rw">2 · 1<`) {
		t.Fatal("fresh stats are not shown on the nodes page")
	}
	r.exec("UPDATE replicas SET live_at = now() - interval '20 seconds' WHERE id = $1", r.replica)
	_, body := owner.get("/console/live.json")
	var snap fleet.Snapshot
	_ = json.Unmarshal([]byte(body), &snap)
	if snap.Tiles.TokensPerSecond != 0 || snap.Throughput[60] != 0 {
		t.Fatalf("stale stats counted: %+v", snap.Tiles)
	}
	for _, n := range snap.Nodes {
		if n.Name == "studio-1" && (n.Fresh || n.State != "online") {
			t.Fatalf("stale node row %+v", n)
		}
	}
	if _, page := owner.get("/console/nodes"); !strings.Contains(page, `data-col="tps">—<`) {
		t.Fatal("stale stats are still shown as live")
	}
	r.exec("UPDATE nodes SET last_seen_at = now() - interval '5 minutes' WHERE id = $1", r.node)
	if _, page := owner.get("/console"); !strings.Contains(page, "pill-stale") || !strings.Contains(page, "not checked in for 90 seconds") {
		t.Fatal("a silent node is not shown as stale with a warning")
	}
	if _, page := owner.get("/console/models"); !strings.Contains(page, "pill-unavailable") {
		t.Fatal("a model whose only node went silent still shows as available")
	}
}
