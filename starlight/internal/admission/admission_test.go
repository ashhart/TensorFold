package admission_test

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/admission"
	"github.com/ashhart/TensorFold/starlight/internal/authn"
	"github.com/ashhart/TensorFold/starlight/internal/catalog"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/levels"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
)

type fleet struct {
	pool       *pgxpool.Pool
	model      string
	deployment string
	replica    string
}

func newFleet(t *testing.T, slots int) fleet {
	pool := testutil.DB(t)
	ctx := context.Background()
	f := fleet{pool: pool, model: ids.New(), deployment: ids.New(), replica: ids.New()}
	node := ids.New()
	exec := func(sql string, args ...any) {
		if _, err := pool.Exec(ctx, sql, args...); err != nil {
			t.Fatal(err)
		}
	}
	exec("INSERT INTO models (id, name, display_name) VALUES ($1, 'qwen3.8-27b', 'Qwen')", f.model)
	exec(`INSERT INTO deployments (id, model_id, name, spec) VALUES ($1, $2, 'qwen-mac', '{"model":{"kind":"path","path":"/m"}}')`, f.deployment, f.model)
	exec("INSERT INTO nodes (id, name, status, address, last_seen_at) VALUES ($1, 'studio-1', 'active', '192.0.2.10:7443', now())", node)
	exec("INSERT INTO replicas (id, deployment_id, deployment_revision, node_id, state, slots, epoch) VALUES ($1, $2, 1, $3, 'ready', $4, 1)",
		f.replica, f.deployment, node, slots)
	return f
}

func (f fleet) admitter() *admission.Admitter {
	return &admission.Admitter{Pool: f.pool, GatewayID: ids.New(), Poll: 50 * time.Millisecond, Ageing: time.Hour}
}

func (f fleet) request(p levels.Priority, wait time.Duration) admission.Request {
	return admission.Request{RequestID: ids.New(), ModelID: f.model, Priority: p, Class: levels.Internal, Local: true,
		Tokens: 100, QueueUntil: time.Now().Add(wait)}
}

func (f fleet) exec(t *testing.T, sql string, args ...any) {
	t.Helper()
	if _, err := f.pool.Exec(context.Background(), sql, args...); err != nil {
		t.Fatalf("%s: %v", sql, err)
	}
}

func record() admission.Record {
	return admission.Record{RequestID: "r", PrincipalType: "user", PrincipalID: ids.New(), ProjectID: ids.New(), ModelName: "m",
		Protocol: "openai.chat", Status: 200, Outcome: "ok", Classification: "internal"}
}

func TestSlotsNeverOversubscribedAcrossGateways(t *testing.T) {
	f := newFleet(t, 3)
	ctx := context.Background()
	gateways := []*admission.Admitter{f.admitter(), f.admitter(), f.admitter(), f.admitter()}
	var inFlight, peak, done atomic.Int64
	var wg sync.WaitGroup
	for i := 0; i < 48; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			a := gateways[i%len(gateways)]
			l, err := a.Admit(ctx, f.request(levels.Standard, 30*time.Second))
			if err != nil {
				t.Error(err)
				return
			}
			n := inFlight.Add(1)
			for {
				p := peak.Load()
				if n <= p || peak.CompareAndSwap(p, n) {
					break
				}
			}
			time.Sleep(5 * time.Millisecond)
			inFlight.Add(-1)
			if err := a.Settle(ctx, l, admission.Usage{Prompt: 10, Completion: 5, Observed: true}, admission.Record{RequestID: "r", PrincipalType: "user",
				PrincipalID: ids.New(), ProjectID: ids.New(), ModelName: "qwen3.8-27b", Protocol: "openai.chat", Status: 200, Outcome: "ok", Classification: "internal"}); err != nil {
				t.Error(err)
			}
			done.Add(1)
		}(i)
	}
	wg.Wait()
	if done.Load() != 48 || peak.Load() > 3 {
		t.Fatalf("done %d, peak in flight %d (slots 3)", done.Load(), peak.Load())
	}
	var leases, usage int
	_ = f.pool.QueryRow(ctx, "SELECT count(*) FROM leases").Scan(&leases)
	_ = f.pool.QueryRow(ctx, "SELECT count(*) FROM usage_events").Scan(&usage)
	if leases != 0 || usage != 48 {
		t.Fatalf("leases left %d, usage rows %d", leases, usage)
	}
}

func TestInteractiveOvertakesQueuedBatch(t *testing.T) {
	f := newFleet(t, 1)
	ctx := context.Background()
	a, b := f.admitter(), f.admitter()
	holder, err := a.Admit(ctx, f.request(levels.Standard, time.Second))
	if err != nil {
		t.Fatal(err)
	}
	order := make(chan levels.Priority, 2)
	run := func(ad *admission.Admitter, p levels.Priority) {
		l, err := ad.Admit(ctx, f.request(p, 10*time.Second))
		if err != nil {
			t.Error(err)
			return
		}
		order <- p
		time.Sleep(20 * time.Millisecond)
		_ = ad.Settle(ctx, l, admission.Usage{Observed: true}, admission.Record{PrincipalType: "user", PrincipalID: ids.New(), ProjectID: ids.New(),
			ModelName: "m", Protocol: "openai.chat", Status: 200, Outcome: "ok", Classification: "internal"})
	}
	go run(a, levels.Batch)
	time.Sleep(150 * time.Millisecond)
	go run(b, levels.Interactive)
	time.Sleep(150 * time.Millisecond)
	if err := a.Settle(ctx, holder, admission.Usage{Observed: true}, admission.Record{PrincipalType: "user", PrincipalID: ids.New(), ProjectID: ids.New(),
		ModelName: "m", Protocol: "openai.chat", Status: 200, Outcome: "ok", Classification: "internal"}); err != nil {
		t.Fatal(err)
	}
	if first, second := <-order, <-order; first != levels.Interactive || second != levels.Batch {
		t.Fatalf("admitted %v before %v", first, second)
	}
}

func TestConcurrencyLimitAndBudgets(t *testing.T) {
	f := newFleet(t, 10)
	ctx := context.Background()
	a := f.admitter()
	req := f.request(levels.Standard, time.Second)
	req.Scopes = []admission.Scope{{Key: "conc:user:u1", Limit: 1}}
	day := time.Now().UTC().Truncate(24 * time.Hour)
	req.Budgets = []admission.Counter{{Key: "tokens:day:project:p1", Window: day, Limit: 1000, Amount: 600}}
	req.Tokens = 600
	l1, err := a.Admit(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	var le *admission.LimitError
	if _, err := a.Admit(ctx, req); !errors.As(err, &le) || le.Kind != "concurrency" {
		t.Fatalf("second concurrent request: %v", err)
	}
	other := req
	other.Scopes = nil
	if _, err := a.Admit(ctx, other); !errors.As(err, &le) || le.Kind != "budget" {
		t.Fatalf("over budget: %v", err)
	}
	if err := a.Settle(ctx, l1, admission.Usage{Prompt: 50, Completion: 50, Observed: true}, admission.Record{PrincipalType: "user",
		PrincipalID: ids.New(), ProjectID: ids.New(), ModelName: "m", Protocol: "openai.chat", Status: 200, Outcome: "ok", Classification: "internal"}); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Admit(ctx, other); err != nil {
		t.Fatalf("budget not refunded after settling actual use: %v", err)
	}
}

func TestReapChargesDeadLeasesInFull(t *testing.T) {
	f := newFleet(t, 1)
	ctx := context.Background()
	a := f.admitter()
	req := f.request(levels.Standard, time.Second)
	day := time.Now().UTC().Truncate(24 * time.Hour)
	req.Budgets = []admission.Counter{{Key: "tokens:day:project:p2", Window: day, Limit: 1000, Amount: 400}}
	req.Tokens = 400
	if _, err := a.Admit(ctx, req); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(ctx, "UPDATE leases SET heartbeat_at = now() - interval '2 minutes'"); err != nil {
		t.Fatal(err)
	}
	n, err := admission.Reap(ctx, f.pool)
	if err != nil || n != 1 {
		t.Fatalf("reaped %d, %v", n, err)
	}
	var used, reserved int64
	_ = f.pool.QueryRow(ctx, "SELECT used, reserved FROM counters WHERE key = 'tokens:day:project:p2'").Scan(&used, &reserved)
	if used != 400 || reserved != 0 {
		t.Fatalf("dead lease charged used=%d reserved=%d", used, reserved)
	}
	if _, err := a.Admit(ctx, f.request(levels.Standard, time.Second)); err != nil {
		t.Fatalf("slot not freed by reaping: %v", err)
	}
}

func TestPinsAndClassification(t *testing.T) {
	f := newFleet(t, 1)
	ctx := context.Background()
	a := f.admitter()
	req := f.request(levels.Standard, 200*time.Millisecond)
	req.Pins = []authn.Pin{{Deployment: f.deployment, Revision: 2}}
	if _, err := a.Admit(ctx, req); !errors.Is(err, admission.ErrNoRoute) {
		t.Fatalf("pinned to a missing revision: %v", err)
	}
	req.Pins = nil
	req.Class = levels.Restricted
	if _, err := a.Admit(ctx, req); !errors.Is(err, admission.ErrCapacity) {
		t.Fatalf("restricted request placed on a confidential node: %v", err)
	}
}

func TestReapLeavesLiveLeasesAlone(t *testing.T) {
	f := newFleet(t, 1)
	ctx := context.Background()
	a := f.admitter()
	if _, err := a.Admit(ctx, f.request(levels.Standard, time.Second)); err != nil {
		t.Fatal(err)
	}
	if _, err := f.pool.Exec(ctx, "UPDATE leases SET deadline = now() - interval '1 hour'"); err != nil {
		t.Fatal(err)
	}
	if n, err := admission.Reap(ctx, f.pool); err != nil || n != 0 {
		t.Fatalf("reaped %d live leases, %v", n, err)
	}
	if _, err := a.Admit(ctx, f.request(levels.Standard, 200*time.Millisecond)); !errors.Is(err, admission.ErrCapacity) {
		t.Fatalf("a second request took the busy slot: %v", err)
	}
}

func TestIsolatedProjectsAvoidSharedPromptCaches(t *testing.T) {
	f := newFleet(t, 2)
	ctx := context.Background()
	project, other := ids.New(), ids.New()
	for _, p := range []string{project, other} {
		if _, err := f.pool.Exec(ctx, "INSERT INTO projects (id, slug, name) VALUES ($1, $2, 'P')", p, "p-"+p[len(p)-8:]); err != nil {
			t.Fatal(err)
		}
	}
	grant := func(p any) {
		if _, err := f.pool.Exec(ctx, "INSERT INTO model_grants (id, model_id, project_id) VALUES ($1, $2, $3)", ids.New(), f.model, p); err != nil {
			t.Fatal(err)
		}
	}
	a := f.admitter()
	req := f.request(levels.Standard, 300*time.Millisecond)
	req.Isolated, req.Who.ProjectID = true, project
	grant(project)
	grant(other)
	if _, err := a.Admit(ctx, req); !errors.Is(err, admission.ErrNoRoute) {
		t.Fatalf("an isolated project reached a replica another project shares: %v", err)
	}
	if _, err := f.pool.Exec(ctx, "DELETE FROM model_grants WHERE project_id = $1", other); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Admit(ctx, req); err != nil {
		t.Fatalf("a model only this project uses refused it: %v", err)
	}
	grant(nil)
	if _, err := f.pool.Exec(ctx, `UPDATE deployments SET spec = spec || '{"prompt_cache": "off"}'::jsonb`); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Admit(ctx, req); err != nil {
		t.Fatalf("a deployment with its prompt cache off refused an isolated project: %v", err)
	}
}

func TestAdmitRefusesNegativeAmounts(t *testing.T) {
	f := newFleet(t, 1)
	ctx := context.Background()
	a := f.admitter()
	day := time.Now().UTC().Truncate(24 * time.Hour)
	for name, negative := range map[string]func(*admission.Request){
		"reservation": func(r *admission.Request) { r.Tokens = -1 << 62 },
		"budget": func(r *admission.Request) {
			r.Budgets = []admission.Counter{{Key: "tokens:day:project:p3", Window: day, Limit: 1000, Amount: -1 << 62}}
		},
	} {
		req := f.request(levels.Standard, 200*time.Millisecond)
		negative(&req)
		if l, err := a.Admit(ctx, req); err == nil || l != nil {
			t.Errorf("a negative %s was admitted", name)
		}
	}
	var leases, counters int
	_ = f.pool.QueryRow(ctx, "SELECT count(*) FROM leases").Scan(&leases)
	_ = f.pool.QueryRow(ctx, "SELECT count(*) FROM counters").Scan(&counters)
	if leases != 0 || counters != 0 {
		t.Fatalf("negative amounts left %d leases and %d counters", leases, counters)
	}
}

func TestFailoverCountsTheRequestOnce(t *testing.T) {
	f := newFleet(t, 4)
	ctx := context.Background()
	a := f.admitter()
	key, now := ids.New(), time.Now()
	limits := []admission.Limit{{ScopeType: "api_key", ScopeID: key, Kind: "requests_per_minute", Value: 2}}
	request := func() admission.Request {
		req := f.request(levels.Standard, 200*time.Millisecond)
		req.Scopes, req.Rates, req.Budgets = admission.Build(limits, admission.Subject{KeyID: key, ModelID: f.model}, now, 100, 0)
		return req
	}
	failed := request()
	for attempt := 0; attempt < 2; attempt++ {
		l, err := a.Admit(ctx, failed)
		if err != nil {
			t.Fatalf("attempt %d: %v", attempt, err)
		}
		if err := a.Settle(ctx, l, admission.Usage{Observed: true}, record()); err != nil {
			t.Fatal(err)
		}
		failed.Retry = true
	}
	l, err := a.Admit(ctx, request())
	if err != nil {
		t.Fatalf("the second request in the minute was refused after the first failed over: %v", err)
	}
	if err := a.Settle(ctx, l, admission.Usage{Observed: true}, record()); err != nil {
		t.Fatal(err)
	}
	var le *admission.LimitError
	if _, err := a.Admit(ctx, request()); !errors.As(err, &le) || le.Kind != "rate" {
		t.Fatalf("a third request in the minute: %v", err)
	}
}

func TestRouteLimitsAreReadAtAdmission(t *testing.T) {
	f := newFleet(t, 1)
	ctx := context.Background()
	provider, route := ids.New(), ids.New()
	f.exec(t, "INSERT INTO providers (id, name, kind, base_url) VALUES ($1, 'acme', 'openai', 'https://api.example.com')", provider)
	f.exec(t, `INSERT INTO provider_routes (id, model_id, provider_id, upstream_model, protocols, max_concurrency)
		VALUES ($1, $2, $3, 'gpt', '{openai.chat}', 64)`, route, f.model, provider)
	a := f.admitter()
	req := f.request(levels.Batch, 200*time.Millisecond)
	req.Local, req.Routes = false, []catalog.ProviderRoute{{ID: route, MaxConcurrency: 64, MaxClass: "internal", UpstreamModel: "gpt"}}
	f.exec(t, "UPDATE provider_routes SET max_concurrency = 1")
	l, err := a.Admit(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Admit(ctx, req); !errors.Is(err, admission.ErrCapacity) {
		t.Fatalf("a second lease on a route lowered to one: %v", err)
	}
	if err := a.Settle(ctx, l, admission.Usage{Observed: true}, record()); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct{ name, set, reset string }{
		{"disabled route", "UPDATE provider_routes SET enabled = false", "UPDATE provider_routes SET enabled = true"},
		{"disabled provider", "UPDATE providers SET enabled = false", "UPDATE providers SET enabled = true"},
		{"lowered ceiling", "UPDATE providers SET max_classification = 'public'", "UPDATE providers SET max_classification = 'internal'"},
	} {
		f.exec(t, c.set)
		if _, err := a.Admit(ctx, req); !errors.Is(err, admission.ErrCapacity) {
			t.Errorf("admitted after a %s: %v", c.name, err)
		}
		f.exec(t, c.reset)
	}
	if _, err := a.Admit(ctx, req); err != nil {
		t.Fatalf("the restored route refused: %v", err)
	}
}
