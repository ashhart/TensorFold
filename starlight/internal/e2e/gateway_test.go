package e2e_test

import (
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

func TestLocalFleetServesEveryProtocol(t *testing.T) {
	f := startFleet(t)
	resp, body := f.call("POST", "/v1/chat/completions", f.key, `{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"hello"}]}`)
	if resp.StatusCode != 200 || resp.Header.Get("X-Starlight-Route") != "local" || resp.Header.Get("X-Starlight-Model") != "qwen3.8-27b" {
		t.Fatalf("chat %d %v %s", resp.StatusCode, resp.Header, body)
	}
	resp, body = f.call("POST", "/v1/chat/completions", f.key, `{"model":"qwen3.8-27b","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != 200 || !strings.Contains(body, "data: [DONE]") || strings.Contains(body, `"choices":[]`) {
		t.Fatalf("stream %d %s", resp.StatusCode, body)
	}
	resp, body = f.call("POST", "/v1/messages", "", `{"model":"claude-sonnet-4-5","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`,
		"X-Api-Key", f.key, "Anthropic-Version", "2023-06-01")
	if resp.StatusCode != 200 || !strings.Contains(body, "message_stop") {
		t.Fatalf("messages %d %s", resp.StatusCode, body)
	}
	resp, body = f.call("GET", "/v1/models", f.key, "")
	if resp.StatusCode != 200 || !strings.Contains(body, `"qwen3.8-27b"`) || !strings.Contains(body, `"claude-sonnet-4-5"`) {
		t.Fatalf("models %d %s", resp.StatusCode, body)
	}
	resp, body = f.call("POST", "/v1/chat/completions", f.key, `{"model":"gpt-9","messages":[]}`)
	if resp.StatusCode != 404 || !strings.Contains(body, "model_not_found") {
		t.Fatalf("unknown model %d %s", resp.StatusCode, body)
	}
	var rows, tokens int
	if err := f.pool.QueryRow(f.ctx, "SELECT count(*), coalesce(sum(prompt_tokens + completion_tokens), 0) FROM usage_events WHERE outcome = 'ok'").
		Scan(&rows, &tokens); err != nil {
		t.Fatal(err)
	}
	if rows != 3 || tokens == 0 {
		t.Fatalf("usage rows %d tokens %d", rows, tokens)
	}
}

func TestConcurrencyLimitAndRevocation(t *testing.T) {
	f := startFleet(t)
	exec(t, f.pool, "INSERT INTO limits (id, scope_type, scope_id, kind, value) VALUES ($1, 'user', $2, 'concurrency', 1)", ids.New(), f.user.ID)
	body := `{"model":"qwen3.8-27b","stream":true,"max_tokens":40,"messages":[{"role":"user","content":"count slowly"}]}`
	var wg sync.WaitGroup
	codes := make(chan int, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, _ := f.call("POST", "/v1/chat/completions", f.key, body)
			codes <- resp.StatusCode
		}()
		time.Sleep(50 * time.Millisecond)
	}
	wg.Wait()
	close(codes)
	got := map[int]int{}
	for c := range codes {
		got[c]++
	}
	if got[200] != 1 || got[429] != 1 {
		t.Fatalf("status counts %v, want one 200 and one 429", got)
	}
	exec(t, f.pool, "UPDATE api_keys SET revoked_at = now()")
	time.Sleep(200 * time.Millisecond)
	if resp, _ := f.call("GET", "/v1/models", f.key, ""); resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("revoked key answered %d", resp.StatusCode)
	}
}
