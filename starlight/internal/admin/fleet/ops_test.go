package fleet_test

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/admin/fleet"
	"github.com/ashhart/TensorFold/starlight/internal/admin/keys"
	"github.com/ashhart/TensorFold/starlight/internal/control"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

func (r *rig) exec(sql string, args ...any) {
	r.t.Helper()
	if _, err := r.pool.Exec(r.ctx, sql, args...); err != nil {
		r.t.Fatalf("%s: %v", sql, err)
	}
}

// fleetFixture adds a model, an active deployment and two nodes, one running a ready replica.
func (r *rig) fleetFixture() (model, deployment, node, replica string) {
	model, deployment, node, replica = ids.New(), ids.New(), ids.New(), ids.New()
	r.exec("INSERT INTO models (id, name, display_name) VALUES ($1, 'qwen3.8-27b', 'Qwen')", model)
	r.exec(`INSERT INTO deployments (id, model_id, name, spec) VALUES ($1, $2, 'qwen-mac', '{"model":{"kind":"path","path":"/m"}}')`, deployment, model)
	r.exec(`INSERT INTO nodes (id, name, status, pool, last_seen_at, facts) VALUES ($1, 'studio-1', 'active', 'mac', now(),
		'{"backend":"metal","memory_bytes":274877906944,"gpus":[{"name":"Apple M3 Ultra"}]}'), ($2, 'studio-2', 'pending', 'mac', now(), '{}')`, node, ids.New())
	r.exec("INSERT INTO replicas (id, deployment_id, deployment_revision, node_id, state, slots, epoch) VALUES ($1, $2, 1, $3, 'ready', 4, 1)", replica, deployment, node)
	r.exec("INSERT INTO node_certificates (serial, node_id, not_after) VALUES ('abc', $1, now() + interval '30 days')", node)
	return
}

func TestNodeActions(t *testing.T) {
	r := newRig(t)
	_, _, node, replica := r.fleetFixture()
	if err := r.do(func(tx *fleet.Tx) error { _, err := tx.SetNodeStatus(r.ctx, "studio-1", fleet.NodeApprove); return err }); !invalid(err) {
		t.Fatalf("approving an active node: %v", err)
	}
	r.must(func(tx *fleet.Tx) error { _, err := tx.SetNodeStatus(r.ctx, "studio-2", fleet.NodeApprove); return err })
	r.must(func(tx *fleet.Tx) error { _, err := tx.SetNodeStatus(r.ctx, "studio-1", fleet.NodeDrain); return err })
	var desired string
	var gen int64
	_ = r.pool.QueryRow(r.ctx, "SELECT r.desired, n.sync_generation FROM replicas r JOIN nodes n ON n.id = r.node_id WHERE r.id = $1", replica).Scan(&desired, &gen)
	if desired != "stopped" || gen < 3 {
		t.Fatalf("drain: replica %s, generation %d", desired, gen)
	}
	r.must(func(tx *fleet.Tx) error { _, err := tx.SetNodeStatus(r.ctx, "studio-1", fleet.NodeOnline); return err })
	r.must(func(tx *fleet.Tx) error { _, err := tx.SetNodeStatus(r.ctx, "studio-1", fleet.NodeRevoke); return err })
	var certRevoked bool
	var state string
	_ = r.pool.QueryRow(r.ctx, "SELECT revoked_at IS NOT NULL FROM node_certificates WHERE node_id = $1", node).Scan(&certRevoked)
	_ = r.pool.QueryRow(r.ctx, "SELECT state FROM replicas WHERE id = $1", replica).Scan(&state)
	if !certRevoked || state != "lost" {
		t.Fatalf("revoke: certificate revoked %v, replica %s", certRevoked, state)
	}
	if err := r.do(func(tx *fleet.Tx) error { _, err := tx.SetNodeStatus(r.ctx, "studio-1", fleet.NodeOnline); return err }); !invalid(err) {
		t.Fatalf("revived a revoked node: %v", err)
	}
	if r.audited("node.drain") != 1 || r.audited("node.revoke") != 1 || r.audited("node.approve") != 1 {
		t.Fatal("node actions were not audited")
	}
	var tok string
	r.must(func(tx *fleet.Tx) (err error) {
		tok, err = tx.CreateEnrolmentToken(r.ctx, control.TokenSpec{Pool: "gpu", Labels: map[string]string{"rack": "a"}, AutoApprove: true, MaxUses: 2})
		return err
	})
	tokens, _ := fleet.EnrolmentTokens(r.ctx, r.pool)
	if !strings.HasPrefix(tok, "sle_") || len(tokens) != 1 || !tokens[0].Usable() || tokens[0].Labels["rack"] != "a" {
		t.Fatalf("token %q %+v", tok, tokens)
	}
	r.must(func(tx *fleet.Tx) error { return tx.RevokeEnrolmentToken(r.ctx, tokens[0].ID) })
	r.must(func(tx *fleet.Tx) error {
		return tx.UpdateNode(r.ctx, "studio-2", "gpu", map[string]string{"rack": "b"}, "restricted")
	})
	n, _ := fleet.NodeByName(r.ctx, r.pool, "studio-2")
	if n.Pool != "gpu" || n.MaxClass != "restricted" || n.State() != "online" {
		t.Fatalf("node %+v state %s", n, n.State())
	}
}

func TestLimitsAccountsAndSubscriptions(t *testing.T) {
	r := newRig(t)
	project := ids.New()
	r.exec("INSERT INTO projects (id, slug, name) VALUES ($1, 'ops', 'Ops')", project)
	r.must(func(tx *fleet.Tx) error {
		if err := tx.SetLimit(r.ctx, fleet.LimitInput{ScopeType: "org", Kind: "concurrency", Value: 100}); err != nil {
			return err
		}
		return tx.SetLimit(r.ctx, fleet.LimitInput{ScopeType: "org", Kind: "concurrency", Value: 200})
	})
	boost, until := int64(500), time.Now().Add(time.Hour)
	r.must(func(tx *fleet.Tx) error {
		return tx.SetLimit(r.ctx, fleet.LimitInput{ScopeType: "project", ScopeID: project, Kind: "cost_per_month", Value: 50_000_000, BoostValue: &boost,
			BoostUntil: &until})
	})
	org, _ := fleet.Limits(r.ctx, r.pool, []string{"org"}, "")
	proj, _ := fleet.Limits(r.ctx, r.pool, []string{"project", "member"}, project)
	if len(org) != 1 || org[0].Value != 200 || len(proj) != 1 || !proj[0].Boosted() || proj[0].Amount(proj[0].Value) != "$50.00" {
		t.Fatalf("limits org %+v project %+v", org, proj)
	}
	if err := r.do(func(tx *fleet.Tx) error {
		return tx.SetLimit(r.ctx, fleet.LimitInput{ScopeType: "org", Kind: "concurrency", Value: 1, BoostValue: &boost})
	}); !invalid(err) {
		t.Fatalf("boost without an end time: %v", err)
	}
	var sa, tok string
	var k keys.Key
	r.must(func(tx *fleet.Tx) (err error) {
		if sa, err = tx.CreateServiceAccount(r.ctx, project, "ci-runner", "pipeline"); err != nil {
			return err
		}
		tok, k, err = tx.IssueServiceKey(r.ctx, project, sa, keys.Spec{Name: "ci", Priority: "batch"})
		return err
	})
	if !strings.HasPrefix(tok, "slr_") || k.OwnerType != "service_account" {
		t.Fatalf("service key %q %+v", tok, k)
	}
	other := ids.New()
	r.exec("INSERT INTO projects (id, slug, name) VALUES ($1, 'other', 'Other')", other)
	if err := r.do(func(tx *fleet.Tx) error { return tx.RevokeProjectKey(r.ctx, other, k.ID) }); err != fleet.ErrNotFound {
		t.Fatalf("revoked a key from another project: %v", err)
	}
	r.must(func(tx *fleet.Tx) error {
		_, _, err := tx.RotateProjectKey(r.ctx, project, k.ID, time.Hour)
		return err
	})
	list, owners, _ := fleet.ProjectKeys(r.ctx, r.pool, project)
	if len(list) != 2 || owners[sa] != "ci-runner" {
		t.Fatalf("project keys %+v owners %v", list, owners)
	}
	active, _ := directory.CreateUser(r.ctx, r.pool, directory.NewUser{Email: "lee@example.com", Source: "jit"})
	gone, _ := directory.CreateUser(r.ctx, r.pool, directory.NewUser{Email: "mo@example.com", Source: "jit", Status: "suspended"})
	var sub string
	r.must(func(tx *fleet.Tx) (err error) {
		sub, err = tx.SaveSubscription(r.ctx, "", fleet.SubscriptionInput{Vendor: "Example AI", Product: "Team", Seats: 2, SeatPriceMicros: 30_000_000})
		if err != nil {
			return err
		}
		if err := tx.AssignSeat(r.ctx, sub, active.ID); err != nil {
			return err
		}
		return tx.AssignSeat(r.ctx, sub, gone.ID)
	})
	third, _ := directory.CreateUser(r.ctx, r.pool, directory.NewUser{Email: "ny@example.com", Source: "jit"})
	if err := r.do(func(tx *fleet.Tx) error { return tx.AssignSeat(r.ctx, sub, third.ID) }); !invalid(err) {
		t.Fatalf("over-assigned seats: %v", err)
	}
	seats, _ := fleet.Seats(r.ctx, r.pool)
	subs, _ := fleet.Subscriptions(r.ctx, r.pool)
	if len(seats) != 2 || !seats[0].Reclaimable() || !seats[1].Reclaimable() || subs[0].MonthlyMicros() != 60_000_000 {
		t.Fatalf("seats %+v subs %+v", seats, subs)
	}
}

func TestUsageReportAndCSV(t *testing.T) {
	r := newRig(t)
	project := ids.New()
	u, _ := directory.CreateUser(r.ctx, r.pool, directory.NewUser{Email: "=cmd@example.com", Source: "jit"})
	r.exec("INSERT INTO projects (id, slug, name) VALUES ($1, 'eng', 'Eng')", project)
	for i, model := range []string{"qwen", "qwen", "frontier"} {
		r.exec(`INSERT INTO usage_events (id, created_at, request_id, principal_type, principal_id, user_id, project_id, model_id, model_name, route_kind,
			protocol, status, outcome, prompt_tokens, completion_tokens, cost_micros, classification, priority)
			VALUES ($1, now() - make_interval(mins => $2), 'r', 'user', $3, $3, $4, $5, $6, 'local', 'openai.chat', 200, 'ok', 100, 50, 1500, 'internal', 'standard')`,
			ids.New(), i, u.ID, project, ids.New(), model)
	}
	f := fleet.UsageFilter{From: time.Now().Add(-time.Hour), To: time.Now().Add(time.Minute), GroupBy: "model"}
	rows, total, err := fleet.Usage(r.ctx, r.pool, f)
	if err != nil || len(rows) != 2 || rows[0].Label != "qwen" || rows[0].Requests != 2 || total.Tokens() != 450 || total.Cost() != "$0.0045" {
		t.Fatalf("usage %+v total %+v %v", rows, total, err)
	}
	f.GroupBy, f.Projects = "user", []string{ids.New()}
	if rows, _, _ := fleet.Usage(r.ctx, r.pool, f); len(rows) != 0 {
		t.Fatalf("project filter leaked rows: %+v", rows)
	}
	f.Projects = nil
	rows, _, _ = fleet.Usage(r.ctx, r.pool, f)
	var buf bytes.Buffer
	if err := fleet.WriteCSV(&buf, "user", rows); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "'=cmd@example.com,3,300,150,0,0.004500") {
		t.Fatalf("csv %q", buf.String())
	}
	if _, _, err := fleet.Usage(r.ctx, r.pool, fleet.UsageFilter{From: f.From, To: f.To, GroupBy: "colour"}); !invalid(err) {
		t.Fatal("unknown grouping accepted")
	}
	f.GroupBy, f.Limit = "model", 1
	rows, total, err = fleet.Usage(r.ctx, r.pool, f)
	if err != nil || len(rows) != 1 || total.Groups != 2 || total.Requests != 3 || total.Tokens() != 450 {
		t.Fatalf("a truncated report must still total every group: rows %+v total %+v %v", rows, total, err)
	}
}

func TestMoney(t *testing.T) {
	for in, want := range map[string]int64{"3": 3_000_000, "0.15": 150_000, "$12.000001": 12_000_001, "": 0, ".5": 500_000} {
		got, err := fleet.ParseDollars("price", in)
		if err != nil || got != want {
			t.Errorf("ParseDollars(%q) = %d, %v", in, got, err)
		}
	}
	for _, bad := range []string{"-1", "1.0000001", "1e3", "abc"} {
		if _, err := fleet.ParseDollars("price", bad); err == nil {
			t.Errorf("accepted %q", bad)
		}
	}
	if fleet.Dollars(1_500_000) != "$1.50" || fleet.Dollars(4500) != "$0.0045" || fleet.Count(1234567) != "1,234,567" {
		t.Fatal("formatting")
	}
}
