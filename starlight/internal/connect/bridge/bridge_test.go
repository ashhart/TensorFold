package bridge

import (
	"bufio"
	"context"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/connect/appkeys"
	"github.com/ashhart/TensorFold/starlight/internal/connect/fakestarlight"
	"github.com/ashhart/TensorFold/starlight/internal/connect/store"
	"github.com/ashhart/TensorFold/starlight/internal/connect/tokens"
)

type fixture struct {
	starlight *fakestarlight.Starlight
	tokens    *tokens.Manager
	srv       *Server
	key       string
}

func setup(t *testing.T, signedIn bool) fixture {
	t.Helper()
	f := fakestarlight.New(t)
	dir := t.TempDir()
	m := tokens.New(store.NewFile(dir), filepath.Join(dir, "lock"), http.DefaultClient, f.Doc())
	if signedIn {
		if err := m.Save(f.Issue()); err != nil {
			t.Fatal(err)
		}
	}
	apps, _ := appkeys.Open(filepath.Join(dir, "apps.json"))
	key, _ := apps.Add("claude-code")
	s, err := New(Config{Upstream: f.URL, HTTP: http.DefaultClient, Tokens: m, Apps: apps, Project: "research",
		AllowedOrigins: []string{"https://notebook.example"}})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Shutdown(context.Background()) })
	return fixture{starlight: f, tokens: m, srv: s, key: key}
}

func (fx fixture) do(t *testing.T, method, path, body string, edit func(*http.Request)) *http.Response {
	t.Helper()
	req, _ := http.NewRequest(method, fx.srv.URL()+path, strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+fx.key)
	req.Header.Set("Content-Type", "application/json")
	if edit != nil {
		edit(req)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

func TestGuards(t *testing.T) {
	fx := setup(t, true)
	cases := []struct {
		name string
		edit func(*http.Request)
		path string
		want int
		body string
	}{
		{"rebinding host", func(r *http.Request) { r.Host = "evil.example:7171" }, "/v1/models", 403, "host_not_allowed"},
		{"browser origin", func(r *http.Request) { r.Header.Set("Origin", "https://evil.example") }, "/v1/models", 403, "origin_not_allowed"},
		{"no key", func(r *http.Request) { r.Header.Del("Authorization") }, "/v1/models", 401, "invalid_api_key"},
		{"starlight token", func(r *http.Request) { r.Header.Set("Authorization", "Bearer "+fx.starlight.Issue().AccessToken) }, "/v1/models", 401, "invalid_api_key"},
		{"anthropic envelope", func(r *http.Request) { r.Header.Del("Authorization") }, "/v1/messages", 401, "authentication_error"},
		{"unlisted route", nil, "/v1/slide/learn", 404, "not_found"},
		{"oauth via traversal", nil, "/v1/../oauth/token", 404, "not_found"},
	}
	for _, c := range cases {
		method := "GET"
		if strings.HasPrefix(c.path, "/v1/messages") || strings.Contains(c.path, "slide") || strings.Contains(c.path, "oauth") {
			method = "POST"
		}
		resp := fx.do(t, method, c.path, "{}", c.edit)
		got, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != c.want || !strings.Contains(string(got), c.body) {
			t.Errorf("%s: %d %s", c.name, resp.StatusCode, got)
		}
	}
	allowed := fx.do(t, "GET", "/v1/models", "", func(r *http.Request) { r.Header.Set("Origin", "https://notebook.example") })
	if allowed.StatusCode != 200 {
		t.Fatalf("allowlisted origin got %d", allowed.StatusCode)
	}
}

func TestForwardsAsTheSignedInUser(t *testing.T) {
	fx := setup(t, true)
	resp := fx.do(t, "POST", "/v1/chat/completions", `{"model":"qwen3.8-27b"}`, func(r *http.Request) {
		r.Header.Set("Cookie", "session=leak")
		r.Header.Set("X-Api-Key", "should-not-pass")
		r.Header.Set("X-Starlight-Client", "spoofed")
	})
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
	reqs := fx.starlight.Requests()
	if len(reqs) != 1 {
		t.Fatalf("%d upstream requests", len(reqs))
	}
	h := reqs[0].Header
	cur, _ := fx.tokens.Current()
	if h.Get("Authorization") != "Bearer "+cur.AccessToken || h.Get("X-Starlight-Client") != "connect/claude-code" || h.Get("X-Starlight-Project") != "research" {
		t.Fatalf("upstream headers %v", h)
	}
	if h.Get("Cookie") != "" || h.Get("X-Api-Key") != "" || reqs[0].Body != `{"model":"qwen3.8-27b"}` {
		t.Fatalf("client credentials leaked upstream: %v %q", h, reqs[0].Body)
	}
}

func TestStreamsIncrementally(t *testing.T) {
	fx := setup(t, true)
	fx.starlight.Gate = make(chan struct{})
	resp := fx.do(t, "POST", "/v1/chat/completions", `{"stream":true}`, nil)
	r := bufio.NewReader(resp.Body)
	first := make(chan string, 1)
	go func() {
		line, _ := r.ReadString('\n')
		first <- line
	}()
	select {
	case line := <-first:
		if !strings.Contains(line, `"chunk":0`) {
			t.Fatalf("first line %q", line)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("first event was held back until the stream finished")
	}
	close(fx.starlight.Gate)
	rest, _ := io.ReadAll(r)
	if !strings.Contains(string(rest), "[DONE]") || resp.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("rest %q headers %v", rest, resp.Header)
	}
}

func TestRetriesOnceAfterStarlightRejectsToken(t *testing.T) {
	fx := setup(t, true)
	cur, _ := fx.tokens.Current()
	fx.starlight.Expire(cur.AccessToken)
	resp := fx.do(t, "POST", "/v1/responses", `{"input":"hi"}`, nil)
	if resp.StatusCode != 200 || fx.starlight.RefreshCalls.Load() != 1 {
		t.Fatalf("status %d after %d refreshes", resp.StatusCode, fx.starlight.RefreshCalls.Load())
	}
	reqs := fx.starlight.Requests()
	if len(reqs) != 1 || reqs[0].Body != `{"input":"hi"}` {
		t.Fatalf("retried body %+v", reqs)
	}
}

func TestSignedOut(t *testing.T) {
	fx := setup(t, false)
	resp := fx.do(t, "POST", "/v1/messages", `{}`, nil)
	got, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 401 || !strings.Contains(string(got), "tensorfold-connect login") {
		t.Fatalf("signed-out reply %d %s", resp.StatusCode, got)
	}
}

func TestAllowed(t *testing.T) {
	ok := [][2]string{{"GET", "/v1/models"}, {"GET", "/v1/models/TensorFold/Qwen3.8-27B"}, {"DELETE", "/v1/responses/resp_1"}, {"POST", "/v1/messages/count_tokens"}}
	for _, c := range ok {
		if !Allowed(c[0], c[1]) {
			t.Errorf("refused %v", c)
		}
	}
	no := [][2]string{{"POST", "/v1/models"}, {"GET", "/v1/models/../x"}, {"GET", "/metrics"}, {"POST", "/oauth/token"}, {"GET", "/v1//models"}}
	for _, c := range no {
		if Allowed(c[0], c[1]) {
			t.Errorf("allowed %v", c)
		}
	}
}
