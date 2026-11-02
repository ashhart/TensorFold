package admission

import (
	"context"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// enqueue takes its time from the database clock, the one every rank is computed on.
func (a *Admitter) enqueue(ctx context.Context, req Request, cands []string) (entry, error) {
	ctx, cancel := context.WithTimeout(ctx, a.timeout())
	defer cancel()
	e := entry{id: ids.New()}
	if cands == nil {
		cands = []string{}
	}
	err := a.Pool.QueryRow(ctx, `INSERT INTO admission_queue (id, request_id, gateway_id, model_id, priority, base_rank, candidates, deadline)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8) RETURNING enqueued_at`, e.id, req.RequestID, a.GatewayID, req.ModelID, req.Priority.String(), Rank(req.Priority),
		cands, req.QueueUntil).Scan(&e.enqueued)
	return e, err
}

func (a *Admitter) dequeue(id string) {
	_ = a.retry(context.Background(), func(ctx context.Context) error {
		_, err := a.Pool.Exec(ctx, "DELETE FROM admission_queue WHERE id = $1", id)
		return err
	})
}

// QueueDepth reports live waiting requests per model, for metrics and the scheduler's demand signal.
func QueueDepth(ctx context.Context, pool *pgxpool.Pool) (map[string]int, error) {
	rows, err := pool.Query(ctx, `SELECT model_id, count(*) FROM admission_queue
		WHERE heartbeat_at > now() - interval '`+queueSQL+`' AND deadline > now() GROUP BY model_id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string]int{}
	for rows.Next() {
		var m string
		var n int
		if err := rows.Scan(&m, &n); err != nil {
			return nil, err
		}
		out[m] = n
	}
	return out, rows.Err()
}

type waiter struct {
	ch   chan struct{}
	rank int
	at   time.Time
}

// Waker fans capacity notifications out to this gateway's waiters, best-ranked first, one at a time.
type Waker struct {
	mu   sync.Mutex
	subs map[string][]*waiter
}

// NewWaker returns an idle waker; Run connects it to Postgres notifications.
func NewWaker() *Waker { return &Waker{subs: map[string][]*waiter{}} }

func (w *Waker) subscribe(model string, rank int, at time.Time) (*waiter, func()) {
	wt := &waiter{ch: make(chan struct{}, 1), rank: rank, at: at}
	w.mu.Lock()
	w.subs[model] = append(w.subs[model], wt)
	w.mu.Unlock()
	return wt, func() {
		w.mu.Lock()
		defer w.mu.Unlock()
		list := w.subs[model]
		for i, x := range list {
			if x == wt {
				w.subs[model] = append(list[:i], list[i+1:]...)
				break
			}
		}
		if len(w.subs[model]) == 0 {
			delete(w.subs, model)
		}
	}
}

// Notify wakes the best-ranked local waiter not already woken, so slots freed together are taken together.
func (w *Waker) Notify(model string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	wake(w.ranked(model))
}

// pass hands on a wake-up wt lost to a better-ranked waiter, who may sit after wt here as this order ignores ageing.
func (w *Waker) pass(model string, wt *waiter) {
	w.mu.Lock()
	defer w.mu.Unlock()
	list := w.ranked(model)
	for i, x := range list {
		if x == wt {
			wake(list[i+1:])
			return
		}
	}
}

func (w *Waker) ranked(model string) []*waiter {
	list := w.subs[model]
	sort.SliceStable(list, func(i, j int) bool {
		if list[i].rank != list[j].rank {
			return list[i].rank > list[j].rank
		}
		return list[i].at.Before(list[j].at)
	})
	return list
}

func wake(list []*waiter) {
	for _, wt := range list {
		select {
		case wt.ch <- struct{}{}:
			return
		default:
		}
	}
}

// Run listens for capacity notifications until ctx ends, reconnecting with backoff.
func (w *Waker) Run(ctx context.Context, pool *pgxpool.Pool) {
	backoff := time.Second
	for ctx.Err() == nil {
		if err := w.listen(ctx, pool); err != nil && ctx.Err() == nil {
			time.Sleep(backoff)
			backoff = min(backoff*2, 30*time.Second)
			continue
		}
		backoff = time.Second
	}
}

func (w *Waker) listen(ctx context.Context, pool *pgxpool.Pool) error {
	conn, err := pool.Acquire(ctx)
	if err != nil {
		return err
	}
	defer conn.Release()
	if _, err := conn.Exec(ctx, "LISTEN starlight_capacity"); err != nil {
		return err
	}
	for {
		n, err := conn.Conn().WaitForNotification(ctx)
		if err != nil {
			conn.Conn().Close(context.Background())
			return err
		}
		w.Notify(strings.TrimSpace(n.Payload))
	}
}
