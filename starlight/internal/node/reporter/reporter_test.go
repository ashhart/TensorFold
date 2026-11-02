package reporter

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/node/controlclient"
)

func client(t *testing.T, h http.HandlerFunc) *controlclient.Client {
	t.Helper()
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	u, _ := url.Parse(srv.URL)
	return &controlclient.Client{HTTP: srv.Client(), Base: u}
}

func report(id string, tps float64) wire.StatsReport {
	return wire.StatsReport{Replicas: []wire.ReplicaStats{{ReplicaID: id, TokensPerSecond: tps}}}
}

func TestReportsReachStarlight(t *testing.T) {
	var mu sync.Mutex
	var got []wire.StatsReport
	c := client(t, func(w http.ResponseWriter, r *http.Request) {
		var rep wire.StatsReport
		if r.URL.Path != wire.PathStats || json.NewDecoder(r.Body).Decode(&rep) != nil {
			w.WriteHeader(400)
			return
		}
		mu.Lock()
		got = append(got, rep)
		mu.Unlock()
		w.WriteHeader(http.StatusNoContent)
	})
	r := &Reporter{Client: c}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go r.Run(ctx)
	r.Offer(report("a", 42.5))
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		mu.Lock()
		n := len(got)
		mu.Unlock()
		if n > 0 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(got) != 1 || got[0].Replicas[0].ReplicaID != "a" || got[0].Replicas[0].TokensPerSecond != 42.5 {
		t.Fatalf("received %+v", got)
	}
}

func TestOfferNeverBlocksAndKeepsTheNewest(t *testing.T) {
	release := make(chan struct{})
	var seen atomic.Value
	c := client(t, func(w http.ResponseWriter, r *http.Request) {
		var rep wire.StatsReport
		_ = json.NewDecoder(r.Body).Decode(&rep)
		seen.Store(rep.Replicas[0].ReplicaID)
		<-release
		w.WriteHeader(http.StatusNoContent)
	})
	r := &Reporter{Client: c, Timeout: 5 * time.Second}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go r.Run(ctx)
	r.Offer(report("first", 1))
	time.Sleep(100 * time.Millisecond)
	start := time.Now()
	for i := 0; i < 1000; i++ {
		r.Offer(report("stale", float64(i)))
	}
	r.Offer(report("newest", 2))
	if time.Since(start) > 100*time.Millisecond {
		t.Fatalf("Offer blocked for %s while a post was in flight", time.Since(start))
	}
	close(release)
	deadline := time.Now().Add(5 * time.Second)
	for seen.Load() != "newest" && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if seen.Load() != "newest" {
		t.Fatalf("last report sent was %v, want the newest", seen.Load())
	}
}

func TestFailuresBackOff(t *testing.T) {
	var hits atomic.Int32
	c := client(t, func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	})
	r := &Reporter{Client: c, Backoff: func(int) time.Duration { return time.Hour }}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go r.Run(ctx)
	for i := 0; i < 20; i++ {
		r.Offer(report("a", 1))
		time.Sleep(10 * time.Millisecond)
	}
	if n := hits.Load(); n != 1 {
		t.Fatalf("posted %d times during a backoff, want 1", n)
	}
}
