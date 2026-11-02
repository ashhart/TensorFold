package fakeengine

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

var words = []string{"Hello", " from", " the", " fake", " engine", "."}

// replyTokens is the deterministic reply: up to six tokens, capped by max_tokens when given.
func replyTokens(req map[string]any, fields ...string) []string {
	n := len(words)
	for _, f := range fields {
		if v, ok := req[f].(float64); ok && v > 0 && int(v) < n {
			n = int(v)
		}
	}
	return words[:n]
}

// promptTokens counts whitespace-separated words in every string the request carries as content.
func promptTokens(req map[string]any) int {
	n := 0
	var walk func(v any)
	walk = func(v any) {
		switch t := v.(type) {
		case string:
			n += len(strings.Fields(t))
		case []any:
			for _, x := range t {
				walk(x)
			}
		case map[string]any:
			for k, x := range t {
				if k == "content" || k == "text" || k == "input" || k == "prompt" || k == "messages" || k == "system" || k == "instructions" {
					walk(x)
				}
			}
		}
	}
	walk(req)
	if n == 0 {
		n = 1
	}
	return n
}

func streaming(req map[string]any) bool {
	b, _ := req["stream"].(bool)
	return b
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, status int, code, msg string) {
	writeJSON(w, status, map[string]any{"error": map[string]any{"message": msg, "type": "invalid_request_error", "code": code}})
}

func bearer(r *http.Request) string {
	if h := r.Header.Get("Authorization"); strings.HasPrefix(h, "Bearer ") {
		return strings.TrimPrefix(h, "Bearer ")
	}
	return r.Header.Get("X-Api-Key")
}

type keyCtx struct{}

func withKey(ctx context.Context, k Key) context.Context { return context.WithValue(ctx, keyCtx{}, k) }

func keyFrom(ctx context.Context) (Key, bool) {
	k, ok := ctx.Value(keyCtx{}).(Key)
	return k, ok
}

// sse writes server-sent events and flushes each one.
type sse struct {
	w     http.ResponseWriter
	delay time.Duration
	ctx   context.Context
}

func newSSE(w http.ResponseWriter, r *http.Request, delay time.Duration) *sse {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	return &sse{w: w, delay: delay, ctx: r.Context()}
}

// send writes one event; it reports false once the client has gone.
func (s *sse) send(event string, data any) bool {
	if s.delay > 0 {
		select {
		case <-s.ctx.Done():
			return false
		case <-time.After(s.delay):
		}
	}
	if s.ctx.Err() != nil {
		return false
	}
	var payload string
	if str, ok := data.(string); ok {
		payload = str
	} else {
		b, _ := json.Marshal(data)
		payload = string(b)
	}
	if event != "" {
		fmt.Fprintf(s.w, "event: %s\n", event)
	}
	fmt.Fprintf(s.w, "data: %s\n\n", payload)
	if f, ok := s.w.(http.Flusher); ok {
		f.Flush()
	}
	return true
}

func (e *Engine) nextID(prefix string) string {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.seq++
	return fmt.Sprintf("%s_fake%06d", prefix, e.seq)
}

func (e *Engine) pause(ctx context.Context) bool {
	if e.opts.Delay <= 0 {
		return true
	}
	select {
	case <-ctx.Done():
		return false
	case <-time.After(e.opts.Delay):
		return true
	}
}
