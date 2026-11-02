// Package artifacts stages model files on a node from pinned sources only.
package artifacts

import (
	"context"
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
)

// Stager resolves artifacts to local directories the engine can serve.
type Stager struct {
	Engine          string
	ModelsDir       string
	HFHome          string
	HFEndpoint      string
	Client          *http.Client
	MaxArchiveBytes int64
	MaxExtractBytes int64
	MaxFiles        int
	PullTimeout     time.Duration
	Logger          *slog.Logger

	flight flight
	mu     sync.Mutex
	staged map[string]bool
}

// ErrOffline means an artifact is absent and the node may not fetch it.
var ErrOffline = errors.New("artifact is not on this node and the installation is offline")

// Stage returns the directory holding a; offline refuses any network fetch except an internal HF mirror.
func (s *Stager) Stage(ctx context.Context, a wire.Artifact, offline bool) (string, error) {
	if err := a.Validate(); err != nil {
		return "", err
	}
	path, err := s.flight.do(a.Key(), func() (string, error) {
		switch a.Kind {
		case "path":
			return stagePath(a.Path)
		case "hf":
			return s.stageHF(ctx, a, offline)
		default:
			return s.stageURL(ctx, a)
		}
	})
	if err == nil {
		s.mu.Lock()
		if s.staged == nil {
			s.staged = map[string]bool{}
		}
		s.staged[a.Key()] = true
		s.mu.Unlock()
	}
	return path, err
}

func stagePath(dir string) (string, error) {
	fi, err := os.Stat(dir)
	if err != nil {
		return "", fmt.Errorf("model path %s: %w", dir, err)
	}
	if !fi.IsDir() {
		return "", fmt.Errorf("model path %s is not a directory", dir)
	}
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		return "", fmt.Errorf("model path %s has no config.json", dir)
	}
	return dir, nil
}

// Cached lists artifact keys present on this node, for placement affinity.
func (s *Stager) Cached() []string {
	seen := map[string]bool{}
	s.mu.Lock()
	for k := range s.staged {
		seen[k] = true
	}
	s.mu.Unlock()
	for _, k := range s.cachedHF() {
		seen[k] = true
	}
	for _, k := range s.cachedURL() {
		seen[k] = true
	}
	out := make([]string, 0, len(seen))
	for k := range seen {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func (s *Stager) logger() *slog.Logger {
	if s.Logger != nil {
		return s.Logger
	}
	return slog.Default()
}

type call struct {
	done chan struct{}
	path string
	err  error
}

// flight collapses concurrent stagings of one artifact into a single fetch.
type flight struct {
	mu    sync.Mutex
	calls map[string]*call
}

func (f *flight) do(key string, fn func() (string, error)) (string, error) {
	f.mu.Lock()
	if f.calls == nil {
		f.calls = map[string]*call{}
	}
	if c, ok := f.calls[key]; ok {
		f.mu.Unlock()
		<-c.done
		return c.path, c.err
	}
	c := &call{done: make(chan struct{})}
	f.calls[key] = c
	f.mu.Unlock()
	c.path, c.err = fn()
	f.mu.Lock()
	delete(f.calls, key)
	f.mu.Unlock()
	close(c.done)
	return c.path, c.err
}
