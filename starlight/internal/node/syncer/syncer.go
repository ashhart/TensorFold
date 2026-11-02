// Package syncer long-polls Starlight for the node's desired state and reports what the node is doing.
package syncer

import (
	"context"
	"log/slog"
	"sync/atomic"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/permit"
	"github.com/ashhart/TensorFold/starlight/internal/node/artifacts"
	"github.com/ashhart/TensorFold/starlight/internal/node/controlclient"
	"github.com/ashhart/TensorFold/starlight/internal/node/engine"
	"github.com/ashhart/TensorFold/starlight/internal/node/enroll"
	"github.com/ashhart/TensorFold/starlight/internal/node/state"
)

// Syncer is the node's control loop.
type Syncer struct {
	Client     *controlclient.Client
	Sup        *engine.Supervisor
	Verifier   *permit.Verifier
	Dir        *state.Dir
	Identity   state.Identity
	Creds      *state.Creds
	Stager     *artifacts.Stager
	Facts      func(context.Context) wire.Facts
	Address    string
	Offline    *atomic.Bool
	Changes    <-chan struct{}
	Logger     *slog.Logger
	Backoff    func(int) time.Duration
	FactsEvery time.Duration
	Debounce   time.Duration
}

// Status is what `starlight-node status` shows.
type Status struct {
	NodeID       string               `json:"node_id"`
	Name         string               `json:"name"`
	Server       string               `json:"server"`
	NodeStatus   string               `json:"node_status"`
	Generation   int64                `json:"generation"`
	Offline      bool                 `json:"offline"`
	LastSync     time.Time            `json:"last_sync"`
	LastError    string               `json:"last_error,omitempty"`
	CertNotAfter time.Time            `json:"cert_not_after"`
	Replicas     []wire.ReplicaStatus `json:"replicas"`
}

// Run syncs until ctx ends; transport errors back off, local state changes resend at once.
func (s *Syncer) Run(ctx context.Context) error {
	s.defaults()
	var (
		gen      int64
		failures int
		facts    wire.Facts
		factsAt  time.Time
		st       = Status{NodeID: s.Identity.NodeID, Name: s.Identity.Name, Server: s.Identity.ServerURL}
	)
	for ctx.Err() == nil {
		if factsAt.IsZero() || time.Since(factsAt) > s.FactsEvery {
			facts, factsAt = s.Facts(ctx), time.Now()
		}
		req := wire.SyncRequest{Generation: gen, Address: s.Address, Facts: facts, Replicas: s.Sup.Statuses(), CachedArtifacts: s.Stager.Cached()}
		resp, changed, err := s.poll(ctx, req)
		if ctx.Err() != nil {
			return nil
		}
		if changed {
			continue
		}
		if err != nil {
			failures++
			st.LastError = err.Error()
			if controlclient.Revoked(err) {
				s.Logger.Error("starlight refused this node's identity; it may be revoked or disabled", "error", err)
			} else {
				s.Logger.Warn("sync failed", "error", err)
			}
			s.writeStatus(st)
			if !sleep(ctx, s.Backoff(failures)) {
				return nil
			}
			continue
		}
		failures = 0
		gen = resp.Generation
		s.apply(resp)
		st.NodeStatus, st.Generation, st.Offline, st.LastSync, st.LastError = resp.NodeStatus, resp.Generation, resp.Offline, time.Now().UTC(), ""
		if enroll.NeedsRenewal(s.Creds.Leaf(), resp.RenewAfter, time.Now()) {
			if err := enroll.Renew(ctx, s.Dir, s.Client, s.Identity, s.Creds); err != nil {
				s.Logger.Warn("certificate renewal failed; will retry", "error", err)
				st.LastError = "renewal: " + err.Error()
			} else {
				s.Logger.Info("node certificate renewed", "not_after", s.Creds.Leaf().NotAfter)
			}
		}
		s.writeStatus(st)
	}
	return nil
}

// poll sends one sync, abandoning it when local state changes so Starlight hears about it promptly.
func (s *Syncer) poll(ctx context.Context, req wire.SyncRequest) (wire.SyncResponse, bool, error) {
	callCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	var changed atomic.Bool
	done := make(chan struct{})
	go func() {
		defer close(done)
		select {
		case <-callCtx.Done():
			return
		case <-s.Changes:
		}
		t := time.NewTimer(s.Debounce)
		defer t.Stop()
		select {
		case <-callCtx.Done():
			return
		case <-t.C:
		}
		changed.Store(true)
		cancel()
	}()
	resp, err := s.Client.Sync(callCtx, req)
	cancel()
	<-done
	if err != nil && changed.Load() {
		return resp, true, nil
	}
	return resp, false, err
}

func (s *Syncer) apply(resp wire.SyncResponse) {
	s.Verifier.SetKeys(state.KeyMap(resp.PermitKeys))
	if err := s.Dir.SavePermitKeys(resp.PermitKeys); err != nil {
		s.Logger.Warn("could not persist permit keys", "error", err)
	}
	s.Offline.Store(resp.Offline)
	switch resp.NodeStatus {
	case "revoked", "disabled":
		s.Sup.Apply(nil)
	default:
		s.Sup.Apply(resp.Assignments)
	}
}

func (s *Syncer) writeStatus(st Status) {
	st.Replicas = s.Sup.Statuses()
	if leaf := s.Creds.Leaf(); leaf != nil {
		st.CertNotAfter = leaf.NotAfter
	}
	if err := s.Dir.WriteStatus(st); err != nil {
		s.Logger.Warn("could not write status", "error", err)
	}
}

func (s *Syncer) defaults() {
	if s.Logger == nil {
		s.Logger = slog.Default()
	}
	if s.Backoff == nil {
		s.Backoff = Backoff
	}
	if s.FactsEvery <= 0 {
		s.FactsEvery = 5 * time.Minute
	}
	if s.Debounce <= 0 {
		s.Debounce = 300 * time.Millisecond
	}
	if s.Offline == nil {
		s.Offline = &atomic.Bool{}
	}
}

// Backoff waits 1, 2, 4 ... seconds up to a minute, with up to a quarter added as jitter.
func Backoff(n int) time.Duration {
	d := engine.DefaultBackoff(n)
	return d + time.Duration(time.Now().UnixNano()%int64(d/4+1))
}

func sleep(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}
