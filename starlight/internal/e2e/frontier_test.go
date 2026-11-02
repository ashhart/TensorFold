package e2e_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/plans"
)

// frontier is a fake OpenAI-compatible provider that records the message bytes it receives.
type frontier struct {
	srv      *httptest.Server
	calls    atomic.Int64
	messages atomic.Value
	auth     atomic.Value
	keys     atomic.Value
}

func newFrontier(t *testing.T) *frontier {
	fr := &frontier{}
	fr.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fr.calls.Add(1)
		fr.auth.Store(r.Header.Get("Authorization"))
		var body map[string]json.RawMessage
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &body)
		fr.messages.Store(string(body["messages"]))
		var names []string
		for k := range body {
			names = append(names, k)
		}
		fr.keys.Store(strings.Join(names, ","))
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chatcmpl-1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant",`+
			`"content":"# Goal\nAdd retries.\n# Steps\n1. Edit client.go"},"finish_reason":"stop"}],"usage":{"prompt_tokens":120,"completion_tokens":30}}`)
	}))
	t.Cleanup(fr.srv.Close)
	return fr
}

func (f *fleet) addFrontier(fr *frontier) {
	provider, model := ids.New(), ids.New()
	sealed, err := f.starlight.Keys.Seal([]byte("sk-frontier-test"), "providers.credential:"+provider)
	if err != nil {
		f.t.Fatal(err)
	}
	exec(f.t, f.pool, `INSERT INTO providers (id, name, kind, base_url, credential, max_classification, internal) VALUES ($1, 'frontier', 'openai_compatible', $2, $3, 'internal', true)`,
		provider, fr.srv.URL, sealed)
	exec(f.t, f.pool, "INSERT INTO models (id, name, display_name, route_policy) VALUES ($1, 'frontier-planner', 'Frontier', 'cloud_only')", model)
	exec(f.t, f.pool, "INSERT INTO model_grants (id, model_id, project_id) VALUES ($1, $2, $3)", ids.New(), model, f.project)
	exec(f.t, f.pool, `INSERT INTO provider_routes (id, model_id, provider_id, upstream_model, protocols, price_input_micros, price_output_micros)
		VALUES ($1, $2, $3, 'frontier-large', '{openai.chat}', 3000000, 15000000)`, ids.New(), model, provider)
	f.starlight.Egress.Allow(map[string]bool{"127.0.0.1": true}, false)
}

func TestFrontierPlanLocalExecution(t *testing.T) {
	f := startFleet(t)
	fr := newFrontier(t)
	f.addFrontier(fr)
	draft := `{"title":"Retries","planner_model":"frontier-planner","executor_model":"qwen3.8-27b","brief":"Add retries to the HTTP client.",` +
		`"context":[{"name":"client.go","content":"package client"}],"budget_tokens":50000,"max_concurrency":2,"max_depth":1}`
	resp, body := f.call("POST", "/v1/starlight/plans", f.key, draft)
	var p plans.Plan
	_ = json.Unmarshal([]byte(body), &p)
	if resp.StatusCode != 201 || p.Status != "pending_approval" || !strings.Contains(p.Text, "Add retries") || p.TaskToken != "" {
		t.Fatalf("draft %d %s", resp.StatusCode, body)
	}
	sum := sha256.Sum256([]byte(fr.messages.Load().(string)))
	if p.EgressSHA256 != hex.EncodeToString(sum[:]) || fr.auth.Load().(string) != "Bearer sk-frontier-test" {
		t.Fatalf("egress record %s does not match what the provider received", p.EgressSHA256)
	}
	if resp, _ := f.call("POST", "/v1/starlight/plans/"+p.ID+"/approve", f.key, `{"plan_sha256":"`+strings.Repeat("0", 64)+`"}`); resp.StatusCode != 409 {
		t.Fatalf("approval of a different plan hash answered %d", resp.StatusCode)
	}
	resp, body = f.call("POST", "/v1/starlight/plans/"+p.ID+"/approve", f.key, `{"plan_sha256":"`+p.PlanSHA256+`"}`)
	_ = json.Unmarshal([]byte(body), &p)
	if resp.StatusCode != 200 || p.Status != "approved" || p.TaskToken == "" {
		t.Fatalf("approve %d %s", resp.StatusCode, body)
	}
	task := p.TaskToken
	_, body = f.call("GET", "/v1/starlight/plans/"+p.ID, f.key, "")
	if strings.Contains(body, "task_token") {
		t.Fatal("task token was shown twice")
	}
	resp, body = f.call("POST", "/v1/chat/completions", task, `{"model":"qwen3.8-27b","messages":[{"role":"user","content":"start step 1"}]}`)
	if resp.StatusCode != 200 || resp.Header.Get("X-Starlight-Route") != "local" {
		t.Fatalf("executor via task token %d %s", resp.StatusCode, body)
	}
	before := fr.calls.Load()
	if resp, _ := f.call("POST", "/v1/chat/completions", task, `{"model":"frontier-planner","messages":[{"role":"user","content":"x"}]}`); resp.StatusCode != 404 {
		t.Fatalf("task token reached the frontier model: %d", resp.StatusCode)
	}
	if fr.calls.Load() != before {
		t.Fatal("the frontier provider was called with a local-only task token")
	}
}

func TestCredentialsNeverLeaveAndOfflineMeansOffline(t *testing.T) {
	f := startFleet(t)
	fr := newFrontier(t)
	f.addFrontier(fr)
	resp, body := f.call("POST", "/v1/chat/completions", f.key, `{"model":"frontier-planner","draft":false,"top_k":20,"messages":[{"role":"user","content":"hello"}]}`)
	if resp.StatusCode != 200 || resp.Header.Get("X-Starlight-Route") != "cloud" {
		t.Fatalf("cloud route %d %s", resp.StatusCode, body)
	}
	if k := fr.keys.Load().(string); strings.Contains(k, "draft") || strings.Contains(k, "top_k") {
		t.Fatalf("engine-only fields reached the cloud provider: %s", k)
	}
	calls := fr.calls.Load()
	leaky := `{"model":"frontier-planner","messages":[{"role":"user","content":"why does ` + "AKIA" + `ABCDEFGHIJKLMNOP fail?"}]}`
	resp, body = f.call("POST", "/v1/chat/completions", f.key, leaky)
	if resp.StatusCode != 400 || !strings.Contains(body, "credentials_detected") || fr.calls.Load() != calls {
		t.Fatalf("credential leak attempt: %d %s (provider calls %d -> %d)", resp.StatusCode, body, calls, fr.calls.Load())
	}
	var audited int
	_ = f.pool.QueryRow(f.ctx, "SELECT count(*) FROM audit_events WHERE action = 'guard.credentials_detected'").Scan(&audited)
	if audited != 1 {
		t.Fatalf("guard audit events %d", audited)
	}
	exec(t, f.pool, "UPDATE settings SET offline = true")
	f.starlight.Policies.Flush()
	resp, body = f.call("POST", "/v1/chat/completions", f.key, `{"model":"frontier-planner","messages":[{"role":"user","content":"hello"}]}`)
	if resp.StatusCode != 503 || !strings.Contains(body, "no_route") || fr.calls.Load() != calls {
		t.Fatalf("offline cloud request %d %s", resp.StatusCode, body)
	}
	resp, _ = f.call("POST", "/v1/chat/completions", f.key, `{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hello"}]}`)
	if resp.StatusCode != 200 {
		t.Fatalf("local route offline %d", resp.StatusCode)
	}
}
