package fakeengine

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestMain(m *testing.M) {
	code := m.Run()
	RemoveBinary()
	os.Exit(code)
}

func engine() *Engine {
	return New(Options{Name: "m", Keys: []Key{
		{Label: "full", Secret: "full-key"},
		{Label: "inf", Scopes: []string{"inference"}, Secret: "inf-key"},
		{Label: "rd", Scopes: []string{"read"}, Secret: "read-key"},
	}})
}

func do(e *Engine, method, path, key, body string, mut ...func(*http.Request)) *httptest.ResponseRecorder {
	r := httptest.NewRequest(method, "http://127.0.0.1:8080"+path, strings.NewReader(body))
	if key != "" {
		r.Header.Set("Authorization", "Bearer "+key)
	}
	for _, m := range mut {
		m(r)
	}
	w := httptest.NewRecorder()
	e.ServeHTTP(w, r)
	return w
}

func TestAuthScopesHostAndOrigin(t *testing.T) {
	e := engine()
	cases := []struct {
		method, path, key string
		want              int
	}{
		{"GET", "/health", "", 200},
		{"GET", "/v1/models", "", 401},
		{"GET", "/v1/models", "inf-key", 200},
		{"GET", "/metrics", "inf-key", 403},
		{"GET", "/metrics", "read-key", 200},
		{"POST", "/v1/chat/completions", "read-key", 403},
		{"POST", "/v1/slide/learn", "inf-key", 403},
		{"POST", "/v1/chat/completions", "wrong", 401},
	}
	for _, c := range cases {
		if w := do(e, c.method, c.path, c.key, `{}`); w.Code != c.want {
			t.Errorf("%s %s with %q = %d, want %d", c.method, c.path, c.key, w.Code, c.want)
		}
	}
	if w := do(e, "GET", "/v1/models", "inf-key", "", func(r *http.Request) { r.Host = "starlight.example:8080" }); w.Code != 403 {
		t.Errorf("foreign Host = %d", w.Code)
	}
	if w := do(e, "GET", "/v1/models", "inf-key", "", func(r *http.Request) { r.Header.Set("Origin", "https://evil.example") }); w.Code != 403 {
		t.Errorf("foreign Origin = %d", w.Code)
	}
	if !strings.Contains(do(e, "GET", "/health", "", "").Body.String(), `"status":"ok"`) || strings.Contains(do(e, "GET", "/health", "", "").Body.String(), "max_batch") {
		t.Error("keyed health must reveal only its status")
	}
}

func events(body string) []string {
	var out []string
	sc := bufio.NewScanner(strings.NewReader(body))
	for sc.Scan() {
		if d, ok := strings.CutPrefix(sc.Text(), "data: "); ok {
			out = append(out, d)
		}
	}
	return out
}

func TestChatStreamUsagePlacement(t *testing.T) {
	e := engine()
	w := do(e, "POST", "/v1/chat/completions", "inf-key", `{"stream":true,"messages":[{"role":"user","content":"one two three"}]}`)
	ev := events(w.Body.String())
	if ev[len(ev)-1] != "[DONE]" || !strings.Contains(ev[len(ev)-2], `"usage"`) || !strings.Contains(ev[len(ev)-2], `"finish_reason":"stop"`) {
		t.Fatalf("finish chunk must carry usage: %v", ev)
	}
	w = do(e, "POST", "/v1/chat/completions", "inf-key", `{"stream":true,"stream_options":{"include_usage":true},"max_tokens":2,"messages":[]}`)
	ev = events(w.Body.String())
	if !strings.Contains(ev[len(ev)-2], `"choices":[]`) || !strings.Contains(ev[len(ev)-3], `"length"`) {
		t.Fatalf("include_usage must send a separate usage chunk: %v", ev)
	}
	var whole struct {
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	_ = json.Unmarshal(do(e, "POST", "/v1/chat/completions", "inf-key", `{"messages":[{"role":"user","content":"a b"}]}`).Body.Bytes(), &whole)
	if whole.Usage.PromptTokens != 2 || whole.Usage.CompletionTokens != 6 {
		t.Fatalf("usage %+v", whole.Usage)
	}
}

func TestMessagesEventSequence(t *testing.T) {
	w := do(engine(), "POST", "/v1/messages", "inf-key", `{"stream":true,"max_tokens":3,"messages":[{"role":"user","content":"hi"}]}`)
	var kinds []string
	for _, line := range strings.Split(w.Body.String(), "\n") {
		if k, ok := strings.CutPrefix(line, "event: "); ok {
			kinds = append(kinds, k)
		}
	}
	want := "message_start content_block_start content_block_delta content_block_delta content_block_delta content_block_stop message_delta message_stop"
	if strings.Join(kinds, " ") != want {
		t.Fatalf("events %v", kinds)
	}
}

func TestResponsesStoreRules(t *testing.T) {
	e := engine()
	if w := do(e, "POST", "/v1/responses", "inf-key", `{"input":"hi"}`); w.Code != 403 {
		t.Fatalf("scoped store:true = %d", w.Code)
	}
	if w := do(e, "POST", "/v1/responses", "inf-key", `{"input":"hi","store":false}`); w.Code != 200 {
		t.Fatalf("scoped store:false = %d", w.Code)
	}
	var resp struct{ ID string }
	_ = json.Unmarshal(do(e, "POST", "/v1/responses", "full-key", `{"input":"hi"}`).Body.Bytes(), &resp)
	if w := do(e, "GET", "/v1/responses/"+resp.ID, "inf-key", ""); w.Code != 403 {
		t.Fatalf("scoped read of a stored response = %d", w.Code)
	}
	if w := do(e, "GET", "/v1/responses/"+resp.ID, "full-key", ""); w.Code != 200 {
		t.Fatalf("stored read = %d", w.Code)
	}
	if w := do(e, "DELETE", "/v1/responses/"+resp.ID, "full-key", ""); w.Code != 200 {
		t.Fatalf("delete = %d", w.Code)
	}
	if w := do(e, "GET", "/v1/responses/"+resp.ID, "full-key", ""); w.Code != 404 {
		t.Fatalf("read after delete = %d", w.Code)
	}
	w := do(e, "POST", "/v1/responses", "full-key", `{"input":"hi","stream":true}`)
	if !strings.Contains(w.Body.String(), "event: response.completed") || !strings.Contains(w.Body.String(), `"sequence_number":0`) {
		t.Fatalf("stream %s", w.Body)
	}
}

func TestKeyFileFormatAndPermissions(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "keys")
	body := "# comment\nstarlight[inference]: a\nops[admin]: b\nbare\nmon[read,inference]: c\n"
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	keys, err := ParseKeyFile(p)
	if err != nil || len(keys) != 4 || keys[0].Label != "starlight" || !keys[1].full() || !keys[2].full() || len(keys[3].Scopes) != 2 {
		t.Fatalf("keys %+v %v", keys, err)
	}
	if runtime.GOOS != "windows" {
		_ = os.Chmod(p, 0o644)
		if _, err := ParseKeyFile(p); err == nil {
			t.Fatal("accepted a world-readable key file")
		}
	}
	_ = os.WriteFile(p, []byte("x[root]: a\n"), 0o600)
	if _, err := ParseKeyFile(p); err == nil {
		t.Fatal("accepted an unknown scope")
	}
}

func TestBinaryVersionAndPull(t *testing.T) {
	bin := Binary(t)
	out, err := exec.Command(bin, "--version").Output()
	if err != nil || strings.TrimSpace(string(out)) != Version {
		t.Fatalf("version %q %v", out, err)
	}
	home := t.TempDir()
	cmd := exec.Command(bin, "pull", "org/model@"+strings.Repeat("a", 40))
	cmd.Env = append(os.Environ(), "HF_HOME="+home, "HF_HUB_CACHE=")
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("pull: %v %s", err, b)
	}
	if _, err := os.Stat(filepath.Join(home, "hub", "models--org--model", "snapshots", strings.Repeat("a", 40), "config.json")); err != nil {
		t.Fatal(err)
	}
}

func metric(t *testing.T, e *Engine, name string) float64 {
	t.Helper()
	body := do(e, "GET", "/metrics", "read-key", "").Body.String()
	for _, line := range strings.Split(body, "\n") {
		if strings.HasPrefix(line, name+" ") || strings.HasPrefix(line, name+"{") {
			var v float64
			if _, err := fmt.Sscanf(line[strings.LastIndexByte(line, ' ')+1:], "%g", &v); err == nil {
				return v
			}
		}
	}
	t.Fatalf("%s missing from:\n%s", name, body)
	return 0
}

func TestCountersFollowServedTokens(t *testing.T) {
	e := engine()
	do(e, "POST", "/v1/chat/completions", "inf-key", `{"stream":true,"messages":[{"role":"user","content":"a b c"}]}`)
	do(e, "POST", "/v1/messages", "inf-key", `{"max_tokens":2,"messages":[{"role":"user","content":"d e"}]}`)
	if g := metric(t, e, "tensorfold:generation_tokens_total"); g != 8 {
		t.Fatalf("generation total %v, want 6 streamed + 2", g)
	}
	if p := metric(t, e, "tensorfold:prompt_tokens_total"); p != 5 {
		t.Fatalf("prompt total %v", p)
	}
	if r := metric(t, e, "tensorfold:generation_tokens_running"); r != 0 {
		t.Fatalf("running after replies ended %v", r)
	}
	if kv := metric(t, e, "tensorfold:kv_cache_usage_ratio"); kv != 0 {
		t.Fatalf("kv while idle %v", kv)
	}
	synthetic := New(Options{Name: "m", SyntheticTPS: 1000})
	time.Sleep(50 * time.Millisecond)
	if g := metric(t, synthetic, "tensorfold:generation_tokens_total"); g < 40 {
		t.Fatalf("synthetic generation %v after 50ms at 1000 tok/s", g)
	}
}
