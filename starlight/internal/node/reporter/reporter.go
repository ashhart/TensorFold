// Package reporter posts live replica stats to Starlight without ever holding up the node's other loops.
package reporter

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/node/controlclient"
)

// Reporter sends the newest stats report; stale reports are replaced, and reports during a backoff are dropped.
type Reporter struct {
	Client  *controlclient.Client
	Logger  *slog.Logger
	Backoff func(int) time.Duration
	Timeout time.Duration

	once sync.Once
	ch   chan wire.StatsReport
}

func (r *Reporter) init() {
	r.once.Do(func() {
		r.ch = make(chan wire.StatsReport, 1)
		if r.Logger == nil {
			r.Logger = slog.Default()
		}
		if r.Backoff == nil {
			r.Backoff = func(n int) time.Duration {
				d := 5 * time.Second
				for i := 1; i < n && d < 2*time.Minute; i++ {
					d *= 2
				}
				return min(d, 2*time.Minute)
			}
		}
		if r.Timeout <= 0 {
			r.Timeout = 10 * time.Second
		}
	})
}

// Offer queues rep, replacing any report not yet sent; it never blocks.
func (r *Reporter) Offer(rep wire.StatsReport) {
	r.init()
	for {
		select {
		case r.ch <- rep:
			return
		default:
		}
		select {
		case <-r.ch:
		default:
		}
	}
}

// Run posts offered reports until ctx ends.
func (r *Reporter) Run(ctx context.Context) {
	r.init()
	failures := 0
	var notBefore time.Time
	for {
		var rep wire.StatsReport
		select {
		case <-ctx.Done():
			return
		case rep = <-r.ch:
		}
		if time.Now().Before(notBefore) {
			continue
		}
		callCtx, cancel := context.WithTimeout(ctx, r.Timeout)
		err := r.Client.Stats(callCtx, rep)
		cancel()
		if ctx.Err() != nil {
			return
		}
		if err != nil {
			failures++
			wait := r.Backoff(failures)
			notBefore = time.Now().Add(wait)
			if failures == 1 {
				r.Logger.Warn("stats report failed; backing off", "error", err, "retry_in", wait)
			}
			continue
		}
		if failures > 0 {
			r.Logger.Info("stats reports resumed", "after_failures", failures)
		}
		failures = 0
	}
}
