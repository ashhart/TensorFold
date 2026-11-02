package engine

import (
	"bufio"
	"context"
	"errors"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
)

// Engine metric families the node reads for live stats.
const (
	famGenTotal   = "tensorfold:generation_tokens_total"
	famGenRunning = "tensorfold:generation_tokens_running"
	famPrompt     = "tensorfold:prompt_tokens_total"
	famRunning    = "tensorfold:requests_running"
	famWaiting    = "tensorfold:requests_waiting"
	famKV         = "tensorfold:kv_cache_usage_ratio"
)

// sample is one scrape of an engine process: every value of each family, across its label sets.
type sample struct {
	at   time.Time
	base string
	fam  map[string][]float64
}

func (s *sample) sum(name string) (float64, bool) {
	vs, ok := s.fam[name]
	total := 0.0
	for _, v := range vs {
		total += v
	}
	return total, ok
}

func (s *sample) max(name string) (float64, bool) {
	vs, ok := s.fam[name]
	m := 0.0
	for _, v := range vs {
		m = max(m, v)
	}
	return m, ok
}

// produced counts finished and still-streaming tokens, so long replies show throughput while they run.
func produced(s *sample) (float64, bool) {
	total, ok := s.sum(famGenTotal)
	if !ok {
		return 0, false
	}
	running, _ := s.sum(famGenRunning)
	return total + running, true
}

func prompted(s *sample) (float64, bool) { return s.sum(famPrompt) }

// rate is the per-second growth of value; a first sample, a new process or a counter reset gives no rate.
func rate(prev, cur *sample, value func(*sample) (float64, bool)) float64 {
	if prev == nil || cur == nil || prev.base != cur.base {
		return 0
	}
	a, okA := value(prev)
	b, okB := value(cur)
	dt := cur.at.Sub(prev.at).Seconds()
	if !okA || !okB || b < a || dt <= 0 {
		return 0
	}
	return (b - a) / dt
}

// parseSample reads the Prometheus text exposition, keeping only the families the node uses.
func parseSample(r io.Reader) map[string][]float64 {
	want := map[string]bool{famGenTotal: true, famGenRunning: true, famPrompt: true, famRunning: true, famWaiting: true, famKV: true}
	out := map[string][]float64{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || line[0] == '#' {
			continue
		}
		end := strings.IndexAny(line, "{ ")
		if end <= 0 || !want[line[:end]] {
			continue
		}
		rest := line[end:]
		if rest[0] == '{' {
			brace := strings.LastIndexByte(rest, '}')
			if brace < 0 {
				continue
			}
			rest = rest[brace+1:]
		}
		fields := strings.Fields(rest)
		if len(fields) == 0 {
			continue
		}
		if v, err := strconv.ParseFloat(fields[0], 64); err == nil {
			out[line[:end]] = append(out[line[:end]], v)
		}
	}
	return out
}

// RunMetrics scrapes each ready replica every interval, updates its queue depth and passes one report to report.
func (s *Supervisor) RunMetrics(ctx context.Context, interval time.Duration, report func(wire.StatsReport)) {
	if interval <= 0 {
		interval = 5 * time.Second
	}
	prev := map[string]*sample{}
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
		rep := s.round(ctx, prev)
		if len(rep.Replicas) > 0 && report != nil {
			report(rep)
		}
	}
}

func (s *Supervisor) round(ctx context.Context, prev map[string]*sample) wire.StatsReport {
	s.mu.Lock()
	rs := make([]*Replica, 0, len(s.replicas))
	for _, r := range s.replicas {
		rs = append(rs, r)
	}
	s.mu.Unlock()
	var rep wire.StatsReport
	live := map[string]bool{}
	for _, r := range rs {
		r.mu.Lock()
		base, key, ready, inFlight := r.base, r.keys.Read, r.state == wire.StateReady, r.inFlight
		r.mu.Unlock()
		if !ready || base == "" {
			continue
		}
		cur, err := s.scrape(ctx, base, key)
		if err != nil {
			continue
		}
		live[r.id] = true
		p := prev[r.id]
		prev[r.id] = cur
		running, ok := cur.sum(famRunning)
		if !ok {
			running = float64(inFlight)
		}
		waiting, ok := cur.sum(famWaiting)
		if ok {
			r.setWaiting(int(waiting))
		}
		kv, _ := cur.max(famKV)
		rep.Replicas = append(rep.Replicas, wire.ReplicaStats{ReplicaID: r.id, TokensPerSecond: rate(p, cur, produced),
			PromptTokensPerSecond: rate(p, cur, prompted), Running: int(running), Waiting: int(waiting), KVUsage: kv})
	}
	for id := range prev {
		if !live[id] {
			delete(prev, id)
		}
	}
	sort.Slice(rep.Replicas, func(i, j int) bool { return rep.Replicas[i].ReplicaID < rep.Replicas[j].ReplicaID })
	return rep
}

func (s *Supervisor) scrape(ctx context.Context, base, key string) (*sample, error) {
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/metrics", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+key)
	resp, err := s.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, errors.New("metrics answered " + resp.Status)
	}
	return &sample{at: time.Now(), base: base, fam: parseSample(io.LimitReader(resp.Body, 4<<20))}, nil
}
