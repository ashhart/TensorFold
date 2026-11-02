// Package admission atomically checks limits, reserves budget and leases capacity, with a shared priority queue.
package admission

import (
	"errors"
	"fmt"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/authn"
	"github.com/ashhart/TensorFold/starlight/internal/catalog"
	"github.com/ashhart/TensorFold/starlight/internal/levels"
)

// Scope is a concurrency limit keyed by what it counts.
type Scope struct {
	Key   string
	Limit int64
}

// Counter is a windowed limit; Amount is what this request adds.
type Counter struct {
	Key    string
	Window time.Time
	Limit  int64
	Amount int64
}

// Request is everything admission needs; the gateway fills it from the caller, catalogue and body.
type Request struct {
	RequestID  string
	ModelID    string
	Priority   levels.Priority
	Class      levels.Class
	Local      bool
	CloudFirst bool
	Routes     []catalog.ProviderRoute
	Pins       []authn.Pin
	Exclude    []string
	Only       string
	Scopes     []Scope
	Rates      []Counter
	Budgets    []Counter
	Tokens     int64
	Micros     int64
	Affinity   uint64
	QueueUntil time.Time
	RunFor     time.Duration
	Who        Who
	Isolated   bool
	// Retry marks a failover attempt, which skips requests_per_minute because the first attempt counted the request.
	Retry bool
}

// Who names the caller on its lease so operators can see what is in flight.
type Who struct {
	PrincipalType string
	PrincipalID   string
	ProjectID     string
	Client        string
}

// Target is where an admitted request runs.
type Target struct {
	ReplicaID    string
	NodeID       string
	NodeAddress  string
	DeploymentID string
	Revision     int
	ServedName   string
	Route        *catalog.ProviderRoute
}

// Local reports whether the target is a node replica.
func (t Target) Local() bool { return t.Route == nil }

// Lease is an admitted request's hold on capacity and budget until it settles.
type Lease struct {
	ID       string
	Target   Target
	Queued   time.Duration
	Admitted time.Time
	req      Request
}

// LimitError reports which limit refused the request and when to retry.
type LimitError struct {
	Kind       string
	Key        string
	RetryAfter int
}

func (e *LimitError) Error() string { return fmt.Sprintf("%s limit reached for %s", e.Kind, e.Key) }

// Errors for requests that cannot be placed.
var (
	ErrCapacity = errors.New("admission: no capacity before the queue deadline")
	ErrNoRoute  = errors.New("admission: the model has no route this request may use")
)

// Rank is a queue entry's base priority score.
func Rank(p levels.Priority) int { return p.Rank() }

// staleAfter is how long a lease or queue entry survives without a heartbeat.
const (
	leaseStale = 60 * time.Second
	staleSQL   = "60 seconds"
	queueSQL   = "30 seconds"
)
