package admission

import (
	"context"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/db"
)

// Admitter is one gateway's view of the shared admission state.
type Admitter struct {
	Pool      *pgxpool.Pool
	GatewayID string
	Ageing    time.Duration
	Waker     *Waker
	Poll      time.Duration
	Timeout   time.Duration
	once      sync.Once
}

type entry struct {
	id       string
	enqueued time.Time
}

// Admit leases capacity now or waits in the queue until req.QueueUntil.
func (a *Admitter) Admit(ctx context.Context, req Request) (*Lease, error) {
	start := time.Now()
	l, cands, _, err := a.attempt(ctx, req, nil, true)
	if err != nil || l != nil {
		return l, err
	}
	if ok, err := a.placeable(ctx, req); err != nil || !ok {
		if err == nil {
			err = ErrNoRoute
		}
		return nil, err
	}
	e, err := a.enqueue(ctx, req, cands)
	if err != nil {
		a.dequeue(e.id)
		return nil, err
	}
	hold(e.id, a.GatewayID)
	admitted := false
	defer func() {
		if !admitted {
			a.dequeue(e.id)
		}
		drop(e.id)
	}()
	wt, cancel := a.waker().subscribe(req.ModelID, Rank(req.Priority), e.enqueued)
	defer cancel()
	a.exec(ctx, "SELECT pg_notify('starlight_demand', $1)", req.ModelID)
	poll := a.Poll
	if poll <= 0 {
		poll = time.Second
	}
	deadline := time.NewTimer(time.Until(req.QueueUntil))
	defer deadline.Stop()
	for {
		woken := false
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-deadline.C:
			return nil, ErrCapacity
		case <-wt.ch:
			woken = true
		case <-time.After(jitter(poll)):
		}
		var owed bool
		l, cands, owed, err = a.attempt(ctx, req, &e, !woken)
		if err != nil {
			return nil, err
		}
		if l != nil {
			admitted, l.Queued = true, time.Since(start)
			a.waker().Notify(req.ModelID)
			return l, nil
		}
		if woken && owed {
			a.waker().pass(req.ModelID, wt)
		}
		a.exec(ctx, "UPDATE admission_queue SET candidates = $2 WHERE id = $1", e.id, cands)
	}
}

// waker makes the Waker once when none was given, so concurrent admissions share it.
func (a *Admitter) waker() *Waker {
	a.once.Do(func() {
		if a.Waker == nil {
			a.Waker = NewWaker()
		}
	})
	return a.Waker
}

func (a *Admitter) timeout() time.Duration {
	if a.Timeout <= 0 {
		return db.CallTimeout
	}
	return a.Timeout
}

// exec runs a statement admission does not depend on, bounded like every other admission call.
func (a *Admitter) exec(ctx context.Context, sql string, args ...any) {
	ctx, cancel := context.WithTimeout(ctx, a.timeout())
	defer cancel()
	_, _ = a.Pool.Exec(ctx, sql, args...)
}

func jitter(d time.Duration) time.Duration {
	return d + time.Duration(time.Now().UnixNano()%int64(d))
}

// attempt leases a slot in one statement per target, nil meaning none; look rules a full model out lock-free first.
func (a *Admitter) attempt(ctx context.Context, req Request, e *entry, look bool) (*Lease, []string, bool, error) {
	ctx, cancel := context.WithTimeout(ctx, a.timeout())
	defer cancel()
	local, cloud, all, err := candidates(ctx, a.Pool, req)
	if err != nil {
		return nil, nil, false, err
	}
	if look {
		if room, err := a.room(ctx, a.Pool, req, e, local, cloud); err != nil || !room {
			if err == nil && e == nil {
				err = overLimit(ctx, a.Pool, req)
			}
			return nil, all, false, err
		}
	}
	owed := false
	for _, t := range preferred(req, local, cloud) {
		l, outranked, err := a.claim(ctx, req, e, t)
		if err != nil || l != nil {
			return l, all, false, err
		}
		owed = owed || outranked
	}
	return nil, all, owed, nil
}

// overLimit refuses at once a request that would queue but already exceeds a limit, reading without reserving.
func overLimit(ctx context.Context, q db.Querier, req Request) error {
	for _, s := range req.Scopes {
		var n int64
		if err := q.QueryRow(ctx, "SELECT count(*) FROM leases WHERE scopes @> ARRAY[$1]::text[] AND heartbeat_at > now() - interval '"+staleSQL+"'",
			s.Key).Scan(&n); err != nil {
			return err
		}
		if n >= s.Limit {
			return &LimitError{Kind: "concurrency", Key: s.Key, RetryAfter: 1}
		}
	}
	rates := req.rates()
	for i, c := range append(append([]Counter(nil), rates...), req.Budgets...) {
		var used, reserved int64
		err := q.QueryRow(ctx, "SELECT used, reserved FROM counters WHERE key = $1 AND window_start = $2", c.Key, c.Window).Scan(&used, &reserved)
		if err != nil && !db.NotFound(err) {
			return err
		}
		if used+reserved+c.Amount > c.Limit {
			kind := "rate"
			if i >= len(rates) {
				kind = "budget"
			}
			return &LimitError{Kind: kind, Key: c.Key, RetryAfter: retryAfter(c)}
		}
	}
	return nil
}

func retryAfter(c Counter) int {
	next := c.Window.Add(windowLength(c.Key))
	s := int(time.Until(next).Seconds()) + 1
	return max(1, min(s, 86400))
}
