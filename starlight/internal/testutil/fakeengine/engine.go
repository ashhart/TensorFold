package fakeengine

import (
	"encoding/json"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Options configure a fake engine; SyntheticTPS adds that many generated tokens per second to the counters.
type Options struct {
	Name         string
	Keys         []Key
	Delay        time.Duration
	SyntheticTPS float64
}

// Request is what the last call delivered, for tests to assert on.
type Request struct {
	Method  string            `json:"method"`
	Path    string            `json:"path"`
	Query   string            `json:"query"`
	Label   string            `json:"label"`
	Headers map[string]string `json:"headers"`
	Body    json.RawMessage   `json:"body"`
}

// Stats counts traffic; MaxConcurrent is the peak number of model requests in flight.
type Stats struct {
	InFlight      int            `json:"in_flight"`
	MaxConcurrent int            `json:"max_concurrent"`
	Requests      int            `json:"requests"`
	Paths         map[string]int `json:"paths"`
	Last          *Request       `json:"last,omitempty"`
}

// Engine is an http.Handler with the engine's routes, auth rules and stream shapes.
type Engine struct {
	opts      Options
	mu        sync.Mutex
	stats     Stats
	responses map[string]json.RawMessage
	seq       int
	count     counters
}

// New returns a fake engine serving opts.Name.
func New(opts Options) *Engine {
	if opts.Name == "" {
		opts.Name = "fake-model"
	}
	return &Engine{opts: opts, stats: Stats{Paths: map[string]int{}}, responses: map[string]json.RawMessage{}, count: counters{start: time.Now()}}
}

// Stats returns a copy of the counters.
func (e *Engine) Stats() Stats {
	e.mu.Lock()
	defer e.mu.Unlock()
	s := e.stats
	s.Paths = make(map[string]int, len(e.stats.Paths))
	for k, v := range e.stats.Paths {
		s.Paths[k] = v
	}
	return s
}

var openRoutes = map[string]bool{"GET /health": true, "GET /": true, "GET /fake/stats": true}

var scopeRoutes = map[string]string{
	"POST /v1/chat/completions": "inference", "POST /v1/completions": "inference", "POST /v1/responses": "inference",
	"POST /v1/messages": "inference", "POST /v1/messages/count_tokens": "inference", "POST /v1/tokenize": "inference",
	"POST /v1/detokenize": "inference", "POST /v1/embeddings": "inference", "GET /v1/models": "inference|read",
	"GET /metrics": "read", "GET /v1/metrics": "read",
}

// ServeHTTP enforces Host, Origin and key rules, then dispatches.
func (e *Engine) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	host, _, _ := net.SplitHostPort(r.Host)
	if host != "127.0.0.1" && host != "localhost" && host != "::1" {
		writeErr(w, http.StatusForbidden, "host_not_allowed", "Host is not loopback")
		return
	}
	if o := r.Header.Get("Origin"); o != "" && o != "http://"+r.Host {
		writeErr(w, http.StatusForbidden, "origin_not_allowed", "Origin does not match Host")
		return
	}
	route := r.Method + " " + strings.TrimSuffix(r.URL.Path, "/")
	if r.URL.Path == "/" {
		route = "GET /"
	}
	label := ""
	if !openRoutes[route] && len(e.opts.Keys) > 0 {
		k, ok := e.match(bearer(r))
		if !ok {
			w.Header().Set("WWW-Authenticate", "Bearer")
			writeErr(w, http.StatusUnauthorized, "invalid_api_key", "missing or invalid key")
			return
		}
		if !allowed(k, route) {
			writeErr(w, http.StatusForbidden, "insufficient_scope", "this key's scopes do not include the route")
			return
		}
		label = k.Label
		r = r.WithContext(withKey(r.Context(), k))
	}
	body, _ := io.ReadAll(io.LimitReader(r.Body, 64<<20))
	if !openRoutes[route] {
		e.record(r, label, body)
	}
	e.dispatch(w, r, route, body)
}

func allowed(k Key, route string) bool {
	if k.full() {
		return true
	}
	need, ok := scopeRoutes[route]
	if !ok {
		return false
	}
	for _, s := range strings.Split(need, "|") {
		if k.has(s) {
			return true
		}
	}
	return false
}

func (e *Engine) dispatch(w http.ResponseWriter, r *http.Request, route string, body []byte) {
	switch {
	case route == "GET /health" || route == "GET /":
		e.health(w)
	case route == "GET /fake/stats":
		writeJSON(w, http.StatusOK, e.Stats())
	case route == "GET /v1/models":
		writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": []map[string]any{{"id": e.opts.Name, "object": "model", "owned_by": "tensorfold"}}})
	case route == "GET /metrics" || route == "GET /v1/metrics":
		e.metrics(w)
	case strings.HasPrefix(route, "POST /v1/"):
		e.model(w, r, route, body)
	case strings.HasPrefix(route, "GET /v1/responses/") || strings.HasPrefix(route, "DELETE /v1/responses/"):
		e.storedResponse(w, r)
	default:
		writeErr(w, http.StatusNotFound, "not_found", "unknown route")
	}
}

func (e *Engine) model(w http.ResponseWriter, r *http.Request, route string, body []byte) {
	e.enter()
	defer e.leave()
	var req map[string]any
	if len(body) > 0 && json.Unmarshal(body, &req) != nil {
		writeErr(w, http.StatusBadRequest, "invalid_json", "body is not JSON")
		return
	}
	switch route {
	case "POST /v1/chat/completions":
		e.chat(w, r, req)
	case "POST /v1/completions":
		e.completion(w, r, req)
	case "POST /v1/messages":
		e.messages(w, r, req)
	case "POST /v1/messages/count_tokens":
		writeJSON(w, http.StatusOK, map[string]any{"input_tokens": promptTokens(req)})
	case "POST /v1/responses":
		e.respond(w, r, req)
	case "POST /v1/embeddings":
		e.embeddings(w, req)
	case "POST /v1/tokenize":
		n := promptTokens(req)
		toks := make([]int, n)
		for i := range toks {
			toks[i] = i + 1
		}
		writeJSON(w, http.StatusOK, map[string]any{"count": n, "max_model_len": 32768, "tokens": toks})
	case "POST /v1/detokenize":
		writeJSON(w, http.StatusOK, map[string]any{"prompt": "detokenized"})
	default:
		writeErr(w, http.StatusNotFound, "not_found", "unknown route")
	}
}

func (e *Engine) health(w http.ResponseWriter) {
	if len(e.opts.Keys) > 0 {
		writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "model": e.opts.Name, "model_ids": []string{e.opts.Name}, "max_batch_size": 8, "version": "fake"})
}

func (e *Engine) enter() {
	e.mu.Lock()
	e.stats.InFlight++
	if e.stats.InFlight > e.stats.MaxConcurrent {
		e.stats.MaxConcurrent = e.stats.InFlight
	}
	e.mu.Unlock()
}

func (e *Engine) leave() {
	e.mu.Lock()
	e.stats.InFlight--
	e.mu.Unlock()
}

func (e *Engine) record(r *http.Request, label string, body []byte) {
	h := map[string]string{}
	for _, k := range []string{"Content-Type", "Accept", "Origin", "Cookie", "X-Request-Id", "Anthropic-Version", "Starlight-Permit", "X-Forwarded-For"} {
		if v := r.Header.Get(k); v != "" {
			h[k] = v
		}
	}
	req := &Request{Method: r.Method, Path: r.URL.Path, Query: r.URL.RawQuery, Label: label, Headers: h}
	if json.Valid(body) {
		req.Body = append(json.RawMessage(nil), body...)
	}
	e.mu.Lock()
	e.stats.Requests++
	e.stats.Paths[r.Method+" "+r.URL.Path]++
	e.stats.Last = req
	e.mu.Unlock()
}
