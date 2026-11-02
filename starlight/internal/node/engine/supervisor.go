package engine

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/node/artifacts"
)

// Config tunes the supervisor; zero durations take production defaults.
type Config struct {
	Engine         string
	Root           string
	Stager         *artifacts.Stager
	Offline        func() bool
	StartupTimeout time.Duration
	DrainTimeout   time.Duration
	StopGrace      time.Duration
	LogMaxBytes    int64
	Env            []string
	Logger         *slog.Logger
	OnChange       func()
	Backoff        func(int) time.Duration
	CrashLimit     int
	CrashWindow    time.Duration
}

// Supervisor owns every replica's engine process on this node.
type Supervisor struct {
	cfg      Config
	job      *jobGroup
	http     *http.Client
	mu       sync.Mutex
	replicas map[string]*Replica
	closed   bool
}

// New prepares a supervisor and stops engines a crashed predecessor left behind.
func New(cfg Config) (*Supervisor, error) {
	if cfg.Engine == "" {
		return nil, errors.New("engine binary is not configured")
	}
	if cfg.Stager == nil {
		return nil, errors.New("artifact stager is not configured")
	}
	defaults(&cfg)
	if err := os.MkdirAll(cfg.Root, 0o700); err != nil {
		return nil, err
	}
	job, err := newJobGroup()
	if err != nil {
		return nil, fmt.Errorf("job object: %w", err)
	}
	reap(cfg.Root, cfg.Logger)
	tr := &http.Transport{Proxy: nil, MaxIdleConnsPerHost: 4, IdleConnTimeout: 30 * time.Second}
	return &Supervisor{cfg: cfg, job: job, http: &http.Client{Transport: tr, Timeout: 10 * time.Second}, replicas: map[string]*Replica{}}, nil
}

func defaults(c *Config) {
	if c.StartupTimeout <= 0 {
		c.StartupTimeout = 30 * time.Minute
	}
	if c.DrainTimeout <= 0 {
		c.DrainTimeout = 2 * time.Minute
	}
	if c.StopGrace <= 0 {
		c.StopGrace = 25 * time.Second
	}
	if c.Logger == nil {
		c.Logger = slog.Default()
	}
	if c.Backoff == nil {
		c.Backoff = DefaultBackoff
	}
	if c.CrashLimit <= 0 {
		c.CrashLimit = 5
	}
	if c.CrashWindow <= 0 {
		c.CrashWindow = 10 * time.Minute
	}
	if c.Offline == nil {
		c.Offline = func() bool { return false }
	}
}

func (s *Supervisor) offline() bool { return s.cfg.Offline() }

func (s *Supervisor) notify() {
	if s.cfg.OnChange != nil {
		s.cfg.OnChange()
	}
}

// Apply makes the running set exactly asgs: new replicas start, changed ones update, missing ones drain and stop.
func (s *Supervisor) Apply(asgs []wire.Assignment) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	want := make(map[string]wire.Assignment, len(asgs))
	for _, a := range asgs {
		if !ids.Valid(a.ReplicaID) {
			s.cfg.Logger.Warn("ignoring assignment with an invalid replica id", "replica", a.ReplicaID)
			continue
		}
		if _, dup := want[a.ReplicaID]; !dup {
			want[a.ReplicaID] = a
		}
	}
	for id, r := range s.replicas {
		if _, ok := want[id]; !ok {
			r.remove()
		}
	}
	for id, a := range want {
		if r, ok := s.replicas[id]; ok {
			if !r.removed() {
				r.update(a)
			}
			continue
		}
		s.start(a)
	}
	go s.notify()
}

func (s *Supervisor) start(a wire.Assignment) {
	ctx, cancel := context.WithCancel(context.Background())
	r := &Replica{sup: s, id: a.ReplicaID, dir: filepath.Join(s.cfg.Root, a.ReplicaID), state: wire.StateAssigned, spent: -1,
		changed: make(chan struct{}, 1), ctx: ctx, cancel: cancel, exited: make(chan struct{})}
	r.update(a)
	<-r.changed
	s.replicas[a.ReplicaID] = r
	go func() {
		r.run()
		_ = os.RemoveAll(r.dir)
		s.mu.Lock()
		if s.replicas[r.id] == r {
			delete(s.replicas, r.id)
		}
		s.mu.Unlock()
		s.notify()
	}()
}

// Get returns a replica by id for the proxy.
func (s *Supervisor) Get(id string) (*Replica, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, ok := s.replicas[id]
	if !ok || r.removed() {
		return nil, false
	}
	return r, true
}

// Statuses reports every replica, sorted by id.
func (s *Supervisor) Statuses() []wire.ReplicaStatus {
	s.mu.Lock()
	rs := make([]*Replica, 0, len(s.replicas))
	for _, r := range s.replicas {
		rs = append(rs, r)
	}
	s.mu.Unlock()
	out := make([]wire.ReplicaStatus, 0, len(rs))
	for _, r := range rs {
		out = append(out, r.status())
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ReplicaID < out[j].ReplicaID })
	return out
}

// Shutdown drains and stops every replica, waiting until ctx expires.
func (s *Supervisor) Shutdown(ctx context.Context) {
	s.mu.Lock()
	s.closed = true
	rs := make([]*Replica, 0, len(s.replicas))
	for _, r := range s.replicas {
		r.remove()
		rs = append(rs, r)
	}
	s.mu.Unlock()
	for _, r := range rs {
		select {
		case <-r.exited:
		case <-ctx.Done():
			return
		}
	}
}

// waitReady waits for the port, /health and the served id in /v1/models, returning any max_batch_size.
func (s *Supervisor) waitReady(ctx context.Context, p *process, served string, keys Keys) (string, int, error) {
	ctx, cancel := context.WithTimeout(ctx, s.cfg.StartupTimeout)
	defer cancel()
	var port int
	select {
	case port = <-p.port:
	case <-p.done:
		return "", 0, errors.New(p.exitError() + " before it announced a port")
	case <-ctx.Done():
		return "", 0, fmt.Errorf("engine did not announce a port within %s", s.cfg.StartupTimeout)
	}
	base := fmt.Sprintf("http://127.0.0.1:%d", port)
	tick := time.NewTicker(250 * time.Millisecond)
	defer tick.Stop()
	for {
		if ok, batch := s.probe(ctx, base, served, keys); ok {
			return base, batch, nil
		}
		select {
		case <-tick.C:
		case <-p.done:
			return "", 0, errors.New(p.exitError() + " during startup")
		case <-ctx.Done():
			return "", 0, fmt.Errorf("engine was not ready within %s", s.cfg.StartupTimeout)
		}
	}
}

// probe checks readiness; max_batch_size is only in /health when the engine reveals it, else 0.
func (s *Supervisor) probe(ctx context.Context, base, served string, keys Keys) (bool, int) {
	var health struct {
		MaxBatch int `json:"max_batch_size"`
	}
	if !s.get(ctx, base+"/health", keys.Read, &health) {
		return false, 0
	}
	var models struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if !s.get(ctx, base+"/v1/models", keys.Inference, &models) {
		return false, 0
	}
	for _, m := range models.Data {
		if m.ID == served {
			return true, health.MaxBatch
		}
	}
	return false, 0
}

func (s *Supervisor) get(ctx context.Context, url, key string, out any) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return false
	}
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := s.http.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return false
	}
	if out == nil {
		return true
	}
	return json.NewDecoder(http.MaxBytesReader(nil, resp.Body, 1<<20)).Decode(out) == nil
}
