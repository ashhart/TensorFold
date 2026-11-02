package engine

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
)

type outcome int

const (
	exited outcome = iota
	restart
	removed
)

// run is the replica's actor: stage, start, watch, and restart within the crash budget until removed.
func (r *Replica) run() {
	defer close(r.exited)
	failures := 0
	for {
		if r.ctx.Err() != nil {
			r.set(wire.StateStopped, "")
			return
		}
		asg, invalid := r.snapshot()
		switch {
		case invalid != "":
			r.set(wire.StateFailed, invalid)
			if !r.waitChange() {
				return
			}
			continue
		case asg.Desired != "running":
			r.set(wire.StateStopped, "")
			if !r.waitChange() {
				return
			}
			continue
		case r.exhausted(asg.Generation):
			if !r.waitChange() {
				return
			}
			continue
		}
		p, err := r.attempt(asg)
		if err != nil {
			if r.ctx.Err() != nil {
				r.set(wire.StateStopped, "")
				return
			}
			if !r.backoff(asg, err.Error(), &failures) {
				return
			}
			continue
		}
		failures = 0
		switch r.monitor(p, asg) {
		case exited:
			if !r.backoff(asg, p.exitError(), &failures) {
				return
			}
		case restart:
			r.drain()
			p.stop(r.sup.cfg.StopGrace)
		case removed:
			r.drain()
			p.stop(r.sup.cfg.StopGrace)
			r.set(wire.StateStopped, "")
			return
		}
	}
}

// attempt stages artifacts and starts the engine, abandoning the attempt if the assignment changes underneath it.
func (r *Replica) attempt(asg wire.Assignment) (*process, error) {
	ctx, cancel := context.WithCancel(r.ctx)
	defer cancel()
	watchDone := make(chan struct{})
	go func() {
		defer close(watchDone)
		for {
			select {
			case <-ctx.Done():
				return
			case <-r.changed:
				cur, _ := r.snapshot()
				if needsRestart(asg, cur) {
					cancel()
					return
				}
			}
		}
	}()
	defer func() { cancel(); <-watchDone }()
	// every start checks the release, so an engine upgraded in place or a check cut short is judged afresh
	if err := checkEngine(ctx, r.sup.cfg.Engine); err != nil {
		return nil, err
	}
	r.set(wire.StateStaging, "")
	model, err := r.sup.cfg.Stager.Stage(ctx, asg.Spec.Model, r.sup.offline())
	if err != nil {
		return nil, fmt.Errorf("stage model: %w", err)
	}
	drafter := ""
	if asg.Spec.Drafter != nil {
		if drafter, err = r.sup.cfg.Stager.Stage(ctx, *asg.Spec.Drafter, r.sup.offline()); err != nil {
			return nil, fmt.Errorf("stage drafter: %w", err)
		}
	}
	r.set(wire.StateStarting, "")
	if err := os.MkdirAll(r.dir, 0o700); err != nil {
		return nil, err
	}
	keys, err := WriteKeys(r.dir)
	if err != nil {
		return nil, err
	}
	port := 0
	if asg.Spec.TensorParallel != "" && asg.Group == nil {
		if port, err = commPort(); err != nil {
			return nil, err
		}
	}
	ranks := rankSpecs(asg, Args(asg.Spec, model, drafter, asg.ServedName, keys.File), childEnv(r.sup.cfg.Env, r.sup.offline()), port)
	p, err := startGroup(r.sup.cfg.Engine, ranks, r.dir, asg.ServedName, r.sup.cfg.LogMaxBytes, r.sup.job)
	if err != nil {
		return nil, err
	}
	if follower(asg) {
		r.setReady("", keys, 0)
		r.sup.cfg.Logger.Info("rank member running", "replica", r.id, "first_rank", asg.Group.FirstRank, "master", asg.Group.Master)
		return p, nil
	}
	base, batch, err := r.sup.waitReady(ctx, p, asg.ServedName, keys)
	if err != nil {
		p.stop(r.sup.cfg.StopGrace)
		return nil, err
	}
	if ctx.Err() != nil {
		p.stop(r.sup.cfg.StopGrace)
		return nil, ctx.Err()
	}
	r.setReady(base, keys, batch)
	r.sup.cfg.Logger.Info("replica ready", "replica", r.id, "served", asg.ServedName, "base", base)
	return p, nil
}

// monitor waits for the engine to exit, the assignment to change, or removal.
func (r *Replica) monitor(p *process, asg wire.Assignment) outcome {
	if cur, invalid := r.snapshot(); invalid != "" || needsRestart(asg, cur) {
		return restart
	}
	for {
		select {
		case <-p.done:
			return exited
		case <-r.ctx.Done():
			return removed
		case <-r.changed:
			cur, invalid := r.snapshot()
			if invalid != "" || needsRestart(asg, cur) {
				return restart
			}
			r.sup.notify()
		}
	}
}

// backoff records a failure and sleeps before the next attempt; it reports false once the replica is removed.
func (r *Replica) backoff(asg wire.Assignment, msg string, failures *int) bool {
	r.sup.cfg.Logger.Warn("replica attempt failed", "replica", r.id, "error", msg)
	if r.recordFailure(asg.Generation, time.Now()) {
		r.set(wire.StateFailed, fmt.Sprintf("%s (failed %d times within %s; waiting for a new assignment generation)",
			msg, r.sup.cfg.CrashLimit, r.sup.cfg.CrashWindow))
		return true
	}
	r.set(wire.StateStarting, msg)
	*failures++
	t := time.NewTimer(r.sup.cfg.Backoff(*failures))
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-r.ctx.Done():
		r.set(wire.StateStopped, "")
		return false
	case <-r.changed:
		return true
	}
}

// waitChange blocks until the assignment changes or the replica is removed.
func (r *Replica) waitChange() bool {
	select {
	case <-r.changed:
		return true
	case <-r.ctx.Done():
		r.set(wire.StateStopped, "")
		return false
	}
}

// DefaultBackoff doubles from one second to a one-minute ceiling.
func DefaultBackoff(n int) time.Duration {
	d := time.Second
	for i := 1; i < n && d < time.Minute; i++ {
		d *= 2
	}
	if d > time.Minute {
		d = time.Minute
	}
	return d
}
