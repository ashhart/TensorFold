package fakeengine

import (
	"fmt"
	"io"
	"net/http"
	"time"
)

// counters are the engine's token totals as its /metrics publishes them.
type counters struct {
	start      time.Time
	genTotal   int64
	genRunning int64
	prompt     int64
}

// tally attributes one request's tokens: streamed tokens count as running until the reply ends.
type tally struct {
	e          *Engine
	prompt     int
	completion int
	stream     bool
	streamed   int
}

func (e *Engine) tally(prompt, completion int, stream bool) *tally {
	return &tally{e: e, prompt: prompt, completion: completion, stream: stream}
}

func (t *tally) token() {
	t.e.mu.Lock()
	t.e.count.genRunning++
	t.streamed++
	t.e.mu.Unlock()
}

func (t *tally) finish() {
	done := t.completion
	if t.stream {
		done = t.streamed
	}
	t.e.mu.Lock()
	t.e.count.genRunning -= int64(t.streamed)
	t.e.count.genTotal += int64(done)
	t.e.count.prompt += int64(t.prompt)
	t.e.mu.Unlock()
}

// metrics publishes the families the node reads; SyntheticTPS adds steady background generation for demos.
func (e *Engine) metrics(w http.ResponseWriter) {
	e.mu.Lock()
	running, c := e.stats.InFlight, e.count
	e.mu.Unlock()
	synthetic := e.opts.SyntheticTPS * time.Since(c.start).Seconds()
	gen := float64(c.genTotal) + synthetic
	prompt := float64(c.prompt) + 4*synthetic
	kv := min(1, float64(running)/8)
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	_, _ = io.WriteString(w, fmt.Sprintf(`# TYPE tensorfold:requests_running gauge
tensorfold:requests_running %d
# TYPE tensorfold:requests_waiting gauge
tensorfold:requests_waiting 0
# TYPE tensorfold:prompt_tokens_total counter
tensorfold:prompt_tokens_total %.0f
# TYPE tensorfold:generation_tokens_total counter
tensorfold:generation_tokens_total %.0f
# TYPE tensorfold:generation_tokens_running gauge
tensorfold:generation_tokens_running %d
# TYPE tensorfold:kv_cache_usage_ratio gauge
tensorfold:kv_cache_usage_ratio{pool="0"} %.4f
`, running, prompt, gen, c.genRunning, kv))
}
