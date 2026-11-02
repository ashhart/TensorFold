package engine

import (
	"context"
	"errors"
	"reflect"
	"slices"
	"sync"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
)

// Errors a proxy gets when it cannot admit a request to a replica.
var (
	ErrNotReady = errors.New("replica is not ready")
	ErrDraining = errors.New("replica is draining")
	ErrBusy     = errors.New("replica has no free slot")
)

// Replica is one assigned engine and the slots the proxy may use on it.
type Replica struct {
	sup *Supervisor
	id  string
	dir string

	mu       sync.Mutex
	asg      wire.Assignment
	invalid  string
	state    string
	err      string
	base     string
	keys     Keys
	inFlight int
	waiting  int
	maxBatch int
	draining bool
	lastUsed time.Time
	crashes  []time.Time
	spent    int64

	changed  chan struct{}
	ctx      context.Context
	cancel   context.CancelFunc
	exited   chan struct{}
	gone     bool
	stopOnce sync.Once
}

// ID returns the replica id.
func (r *Replica) ID() string { return r.id }

// Deployment returns the deployment and revision this replica serves.
func (r *Replica) Deployment() (string, int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.asg.DeploymentID, r.asg.Revision
}

// Endpoint returns the engine's loopback base URL and its keys.
func (r *Replica) Endpoint() (string, Keys) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.base, r.keys
}

// Acquire takes a slot; the returned func releases it.
func (r *Replica) Acquire() (func(), error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	switch {
	case r.draining:
		return nil, ErrDraining
	case r.state != wire.StateReady:
		return nil, ErrNotReady
	case r.inFlight >= r.asg.Slots:
		return nil, ErrBusy
	}
	r.inFlight++
	var once sync.Once
	return func() {
		once.Do(func() {
			r.mu.Lock()
			r.inFlight--
			r.lastUsed = time.Now()
			r.mu.Unlock()
		})
	}, nil
}

func (r *Replica) status() wire.ReplicaStatus {
	r.mu.Lock()
	defer r.mu.Unlock()
	st := wire.ReplicaStatus{ReplicaID: r.id, Generation: r.asg.Generation, State: r.state, Error: r.err, InFlight: r.inFlight, Waiting: r.waiting}
	if r.state == wire.StateReady {
		st.MaxBatch = r.maxBatch
		if st.MaxBatch <= 0 {
			st.MaxBatch = r.asg.Slots
		}
	}
	return st
}

func (r *Replica) snapshot() (wire.Assignment, string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.asg, r.invalid
}

func (r *Replica) set(state, msg string) {
	r.mu.Lock()
	changed := r.state != state || r.err != msg
	r.state, r.err = state, msg
	if state != wire.StateReady && state != wire.StateDraining {
		r.base = ""
	}
	r.mu.Unlock()
	if changed {
		r.sup.notify()
	}
}

func (r *Replica) setReady(base string, keys Keys, maxBatch int) {
	r.mu.Lock()
	r.state, r.err, r.base, r.keys, r.maxBatch = wire.StateReady, "", base, keys, maxBatch
	r.mu.Unlock()
	r.sup.notify()
}

// update records a new desired assignment; slot changes apply at once, others restart the engine.
func (r *Replica) update(a wire.Assignment) {
	invalid := ""
	if err := Check(a); err != nil {
		invalid = err.Error()
	}
	r.mu.Lock()
	r.asg, r.invalid = a, invalid
	r.mu.Unlock()
	select {
	case r.changed <- struct{}{}:
	default:
	}
}

func (r *Replica) remove() {
	r.stopOnce.Do(func() {
		r.mu.Lock()
		r.gone = true
		r.mu.Unlock()
		r.cancel()
	})
}

func (r *Replica) removed() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.gone
}

// needsRestart reports whether moving from a to b changes what the running engine serves.
func needsRestart(a, b wire.Assignment) bool {
	return a.DeploymentID != b.DeploymentID || a.Revision != b.Revision || a.ServedName != b.ServedName ||
		a.Desired != b.Desired || !reflect.DeepEqual(a.Spec, b.Spec) || !slices.Equal(a.Devices, b.Devices)
}

// recordFailure counts a crash and reports whether this generation has exhausted its budget.
func (r *Replica) recordFailure(gen int64, now time.Time) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	window := r.sup.cfg.CrashWindow
	kept := r.crashes[:0]
	for _, t := range r.crashes {
		if now.Sub(t) < window {
			kept = append(kept, t)
		}
	}
	r.crashes = append(kept, now)
	if len(r.crashes) >= r.sup.cfg.CrashLimit {
		r.spent = gen
		r.crashes = nil
		return true
	}
	return false
}

func (r *Replica) exhausted(gen int64) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.spent == gen
}

// drain stops new admissions and waits for in-flight requests, up to the drain timeout.
func (r *Replica) drain() {
	r.mu.Lock()
	r.draining = true
	if r.state == wire.StateReady {
		r.state = wire.StateDraining
	}
	r.mu.Unlock()
	r.sup.notify()
	deadline := time.Now().Add(r.sup.cfg.DrainTimeout)
	for time.Now().Before(deadline) {
		r.mu.Lock()
		n := r.inFlight
		r.mu.Unlock()
		if n == 0 {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	r.mu.Lock()
	r.draining = false
	r.mu.Unlock()
}

func (r *Replica) setWaiting(n int) {
	r.mu.Lock()
	r.waiting = n
	r.mu.Unlock()
}
