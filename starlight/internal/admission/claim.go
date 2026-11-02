package admission

import (
	"context"
	"errors"
	"sort"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgconn"

	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/routing"
)

// claimTries bounds how many targets one attempt tries when other gateways fill them first.
const claimTries = 3

const noQueueEntry = "00000000-0000-0000-0000-000000000000"

// preferred orders the free targets: the routing pick first, then the freest replicas, cloud routes after local ones.
func preferred(req Request, local []localCand, cloud []cloudCand) []Target {
	cands := make([]routing.Candidate, len(local))
	for i, c := range local {
		cands[i] = routing.Candidate{ID: c.t.ReplicaID, Free: c.slots - c.used, Slots: c.slots}
	}
	var near []Target
	first := routing.Pick(cands, req.Affinity)
	if first >= 0 {
		near = append(near, local[first].t)
	}
	rest := make([]localCand, 0, len(local))
	for i, c := range local {
		if i != first && c.slots > c.used {
			rest = append(rest, c)
		}
	}
	sort.SliceStable(rest, func(i, j int) bool { return rest[i].slots-rest[i].used > rest[j].slots-rest[j].used })
	for _, c := range rest {
		near = append(near, c.t)
	}
	var far []Target
	for _, c := range cloud {
		if c.route.MaxConcurrency > c.used {
			r := c.route
			far = append(far, Target{Route: &r, ServedName: r.UpstreamModel})
		}
	}
	if req.CloudFirst {
		near, far = far, near
	}
	out := append(near, far...)
	return out[:min(len(out), claimTries)]
}

// claim leases a slot on t in one statement: a lease, nil (outranked: its slot is owed to the queue), or a LimitError.
func (a *Admitter) claim(ctx context.Context, req Request, e *entry, t Target) (*Lease, bool, error) {
	l := &Lease{ID: ids.New(), Target: t, req: req}
	// The statement outlives a caller who leaves, so a lease it commits is known and released rather than orphaned.
	sctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), a.timeout())
	defer cancel()
	_, err := a.Pool.Exec(sctx, "SELECT starlight_admit($1)", a.admitArgs(req, e, l))
	if ctx.Err() != nil {
		if err == nil {
			a.release(l.ID)
		}
		return nil, false, ctx.Err()
	}
	var pg *pgconn.PgError
	switch {
	case err == nil:
		hold(l.ID, a.GatewayID)
		l.Admitted = time.Now()
		return l, false, nil
	case errors.As(err, &pg) && pg.Code == "SL002":
		return nil, pg.Message == "outranked", nil
	case errors.As(err, &pg) && pg.Code == "SL001":
		return nil, false, limitError(req, pg.Message, pg.Detail)
	default:
		return nil, false, err
	}
}

func (a *Admitter) ageing() time.Duration {
	if a.Ageing <= 0 {
		return 30 * time.Second
	}
	return a.Ageing
}

// admitArgs is starlight_admit's input; counters go bytewise by key, then window, the order every statement locks them.
func (a *Admitter) admitArgs(req Request, e *entry, l *Lease) map[string]any {
	scopes := append([]Scope(nil), req.Scopes...)
	sort.Slice(scopes, func(i, j int) bool { return scopes[i].Key < scopes[j].Key })
	scopeArgs, scopeKeys := make([]map[string]any, len(scopes)), make([]string, len(scopes))
	for i, s := range scopes {
		scopeArgs[i], scopeKeys[i] = map[string]any{"key": s.Key, "limit": s.Limit}, s.Key
	}
	type counted struct {
		c       Counter
		reserve bool
	}
	var cs []counted
	for _, c := range req.rates() {
		cs = append(cs, counted{c, false})
	}
	for _, c := range req.Budgets {
		cs = append(cs, counted{c, true})
	}
	sort.SliceStable(cs, func(i, j int) bool {
		return cs[i].c.Key < cs[j].c.Key || cs[i].c.Key == cs[j].c.Key && cs[i].c.Window.Before(cs[j].c.Window)
	})
	counterArgs := make([]map[string]any, len(cs))
	for i, x := range cs {
		counterArgs[i] = map[string]any{"key": x.c.Key, "window": x.c.Window.UTC(), "limit": x.c.Limit, "amount": x.c.Amount, "reserve": x.reserve}
	}
	runFor := req.RunFor
	if runFor <= 0 {
		runFor = 30 * time.Minute
	}
	args := map[string]any{"id": l.ID, "request": req.RequestID, "gateway": a.GatewayID, "model": req.ModelID, "class": int(req.Class),
		"base": Rank(req.Priority), "queue": noQueueEntry, "queued": e != nil, "ageing": a.ageing().Seconds(), "scopes": scopeArgs,
		"counters": counterArgs, "scope_keys": scopeKeys, "budget_keys": budgetKeys(req), "tokens": req.Tokens, "micros": req.Micros,
		"priority": req.Priority.String(), "deadline": time.Now().Add(runFor).UTC(), "principal_type": req.Who.PrincipalType,
		"principal_id": req.Who.PrincipalID, "project_id": req.Who.ProjectID, "client": req.Who.Client}
	if e != nil {
		args["queue"], args["at"] = e.id, e.enqueued.UTC()
	}
	if l.Target.Local() {
		args["replica"] = l.Target.ReplicaID
	} else {
		args["route"] = l.Target.Route.ID
	}
	return args
}

// rates are the rate counters an attempt adds to; a retry skips requests_per_minute, which its first attempt counted.
func (r Request) rates() []Counter {
	if !r.Retry {
		return r.Rates
	}
	var out []Counter
	for _, c := range r.Rates {
		if !strings.HasPrefix(c.Key, "rpm:") {
			out = append(out, c)
		}
	}
	return out
}

// budgetKeys names the counters a lease settles: budgets and per-minute token rates, each with its window.
func budgetKeys(req Request) []string {
	keys := []string{}
	for _, c := range req.Budgets {
		keys = append(keys, c.Key+"|"+c.Window.UTC().Format(time.RFC3339))
	}
	for _, c := range req.Rates {
		if isTPM(c.Key) {
			keys = append(keys, c.Key+"|"+c.Window.UTC().Format(time.RFC3339))
		}
	}
	return keys
}

func limitError(req Request, kind, key string) error {
	if kind == "concurrency" {
		return &LimitError{Kind: kind, Key: key, RetryAfter: 1}
	}
	for _, c := range append(append([]Counter(nil), req.Rates...), req.Budgets...) {
		if c.Key == key {
			return &LimitError{Kind: kind, Key: key, RetryAfter: retryAfter(c)}
		}
	}
	return &LimitError{Kind: kind, Key: key, RetryAfter: 1}
}
