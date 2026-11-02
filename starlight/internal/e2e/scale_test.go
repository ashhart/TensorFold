package e2e_test

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/config"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// TestEightThousandUsers sends one request from each of 8,000 users through one gateway; STARLIGHT_SCALE=1 runs it.
func TestEightThousandUsers(t *testing.T) {
	if os.Getenv("STARLIGHT_SCALE") == "" {
		t.Skip("set STARLIGHT_SCALE=1 to run the 8,000-user load test")
	}
	users, clients, gateways, nodes := 8000, envInt("STARLIGHT_SCALE_CLIENTS", 512), envInt("STARLIGHT_SCALE_GATEWAYS", 1), envInt("STARLIGHT_SCALE_NODES", 1)
	f := startStarlight(t, func(c *config.Config) {
		c.Gateway.QueueStandard.Duration = 2 * time.Minute
		c.Database.MaxConns = 96
	})
	exec(t, f.pool, "UPDATE deployments SET spec = jsonb_set(spec, '{parallel}', '128'), min_replicas = $1, max_replicas = $1", nodes)
	for i := 1; i <= nodes; i++ {
		f.enrolNamed(f.nodeURL, fmt.Sprintf("studio-%d", i), i)
	}
	urls := []string{f.public.URL}
	for len(urls) < gateways {
		urls = append(urls, f.gateway())
	}
	tokens := f.manyUsers(users)
	var dials atomic.Int64
	dialer := &net.Dialer{Timeout: 10 * time.Second}
	worker := func() *http.Client {
		tr := &http.Transport{MaxConnsPerHost: 1, MaxIdleConnsPerHost: 1, IdleConnTimeout: time.Minute,
			DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
				dials.Add(1)
				return dialer.DialContext(ctx, network, addr)
			}}
		return &http.Client{Transport: tr, Timeout: 3 * time.Minute}
	}
	body := `{"model":"qwen3.8-27b","max_tokens":8,"messages":[{"role":"user","content":"hello"}]}`
	work := make(chan string)
	var mu sync.Mutex
	codes, failures := map[int]int{}, map[string]int{}
	var latencies []time.Duration
	var wg, warm sync.WaitGroup
	ready, dialing := make(chan struct{}), make(chan struct{}, 64)
	for i := 0; i < clients; i++ {
		wg.Add(1)
		warm.Add(1)
		base := urls[i%len(urls)]
		go func() {
			defer wg.Done()
			client := worker()
			dialing <- struct{}{}
			warmConnection(client, base)
			<-dialing
			warm.Done()
			<-ready
			for tok := range work {
				req, _ := http.NewRequestWithContext(f.ctx, "POST", base+"/v1/chat/completions", strings.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
				req.Header.Set("Authorization", "Bearer "+tok)
				t0 := time.Now()
				resp, err := client.Do(req)
				code := 0
				if err == nil {
					code = resp.StatusCode
					b, _ := io.ReadAll(resp.Body)
					if code != http.StatusOK {
						err = fmt.Errorf("%d %s", code, b)
					}
					resp.Body.Close()
				}
				mu.Lock()
				codes[code]++
				if err != nil {
					failures[err.Error()]++
				}
				latencies = append(latencies, time.Since(t0))
				mu.Unlock()
			}
		}()
	}
	warm.Wait()
	dials.Store(0)
	start := time.Now()
	close(ready)
	for _, tok := range tokens {
		work <- tok
	}
	close(work)
	wg.Wait()
	elapsed := time.Since(start)
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	pct := func(p float64) time.Duration {
		return latencies[int(p*float64(len(latencies)-1))].Round(time.Millisecond)
	}
	var leases, usage, distinct int
	ctx := context.Background()
	_ = f.pool.QueryRow(ctx, "SELECT count(*) FROM leases").Scan(&leases)
	_ = f.pool.QueryRow(ctx, "SELECT count(*), count(DISTINCT principal_id) FROM usage_events").Scan(&usage, &distinct)
	t.Logf("%d users, %d concurrent clients, %d gateways, %d replicas of 128 slots: %s, %.0f requests/s; latency p50 %s p95 %s p99 %s max %s",
		users, clients, gateways, nodes, elapsed.Round(time.Millisecond), float64(users)/elapsed.Seconds(), pct(0.50), pct(0.95), pct(0.99), pct(1))
	t.Logf("status codes %v; leases left %d; usage rows %d for %d distinct users; %d connections dialled", codes, leases, usage, distinct, dials.Load())
	for msg, n := range failures {
		t.Logf("%d failed: %s", n, msg)
	}
	if codes[200] != users || leases != 0 || usage != users || distinct != users {
		t.Fatalf("not every user was served exactly once: codes %v leases %d usage %d distinct %d", codes, leases, usage, distinct)
	}
}

// warmConnection opens a worker's one keep-alive connection; 64 dial at a time, within macOS's 128-entry backlog.
func warmConnection(c *http.Client, base string) {
	for i := 0; i < 20; i++ {
		resp, err := c.Get(base + "/healthz")
		if err == nil {
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			return
		}
		time.Sleep(time.Duration(10+i*10) * time.Millisecond)
	}
}

// manyUsers creates n users, each a project member with an API key of their own, and returns the keys.
func (f *fleet) manyUsers(n int) []string {
	userIDs, keyIDs, publics, hashes, emails := make([]string, n), make([]string, n), make([]string, n), make([][]byte, n), make([]string, n)
	tokens := make([]string, n)
	for i := range n {
		tok, hash := secret.Generate(secret.APIKey)
		userIDs[i], keyIDs[i], publics[i], hashes[i], tokens[i] = ids.New(), ids.New(), secret.PublicID(tok), hash, tok
		emails[i] = fmt.Sprintf("user%05d@example.com", i)
	}
	exec(f.t, f.pool, `INSERT INTO users (id, email, display_name, status, source) SELECT u, e, e, 'active', 'jit' FROM unnest($1::uuid[], $2::text[]) AS t(u, e)`,
		userIDs, emails)
	exec(f.t, f.pool, `INSERT INTO role_bindings (id, role, subject_type, subject_id, project_id) SELECT gen_random_uuid(), 'member', 'user', u, $2
		FROM unnest($1::uuid[]) AS t(u)`, userIDs, f.project)
	exec(f.t, f.pool, `INSERT INTO api_keys (id, public_id, secret_hash, name, owner_type, owner_id, project_id)
		SELECT k, p, h, 'load', 'user', u, $5 FROM unnest($1::uuid[], $2::text[], $3::bytea[], $4::uuid[]) AS t(k, p, h, u)`,
		keyIDs, publics, hashes, userIDs, f.project)
	return tokens
}

func envInt(key string, def int) int {
	if n, err := strconv.Atoi(os.Getenv(key)); err == nil && n > 0 {
		return n
	}
	return def
}
