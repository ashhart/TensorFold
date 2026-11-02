package e2e_test

import (
	"bufio"
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/connect/core"
	"github.com/ashhart/TensorFold/starlight/internal/connect/managed"
	"github.com/ashhart/TensorFold/starlight/internal/connect/profile"
	"github.com/ashhart/TensorFold/starlight/internal/connect/store"
	"github.com/ashhart/TensorFold/starlight/internal/discovery"
	"github.com/ashhart/TensorFold/starlight/internal/identity/session"
)

var userCode = regexp.MustCompile(`\b[BCDFGHJKLMNPQRSTVWXZ]{4}-[BCDFGHJKLMNPQRSTVWXZ]{4}\b`)

// TestConnectAgainstRealStarlight signs Connect in with the device grant, then sends tool traffic through it.
func TestConnectAgainstRealStarlight(t *testing.T) {
	t.Setenv(store.BackendEnv, "file")
	f := startFleet(t)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	c, err := core.New(t.TempDir(), profile.Profile{}, managed.Policy{})
	if err != nil {
		t.Fatal(err)
	}
	if err := c.UseServer(f.public.URL); err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	s, err := f.starlight.Sessions.Create(ctx, rec, httptest.NewRequest("GET", "/", nil), session.Start{UserID: f.user.ID})
	if err != nil {
		t.Fatal(err)
	}
	cookie := rec.Result().Cookies()[0]
	out, in := io.Pipe()
	type result struct {
		me  discovery.Me
		err error
	}
	done := make(chan result, 1)
	go func() {
		me, err := c.Login(ctx, core.LoginOptions{Device: true, Out: in})
		in.Close()
		done <- result{me, err}
	}()
	code := make(chan string, 1)
	go func() {
		sc := bufio.NewScanner(out)
		sent := false
		for sc.Scan() {
			if m := userCode.FindString(sc.Text()); m != "" && !sent {
				code <- m
				sent = true
			}
		}
	}()
	var uc string
	select {
	case uc = <-code:
	case <-ctx.Done():
		t.Fatal("Connect never printed a device code")
	}
	browser := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	for _, step := range []struct{ path, decision string }{{"/device", ""}, {"/device/decide", "allow"}} {
		form := url.Values{"user_code": {uc}, "_csrf": {s.CSRF()}}
		if step.decision != "" {
			form.Set("decision", step.decision)
		}
		req, _ := http.NewRequest("POST", f.public.URL+step.path, strings.NewReader(form.Encode()))
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		req.Header.Set("Origin", f.public.URL)
		req.AddCookie(cookie)
		resp, err := browser.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatalf("%s answered %d", step.path, resp.StatusCode)
		}
	}
	r := <-done
	if r.err != nil || r.me.Email != f.user.Email || len(r.me.Projects) == 0 {
		t.Fatalf("Connect login = %+v, %v", r.me, r.err)
	}
	port := freePort(t)
	text, _, err := c.Setup("openai", "", "", "posix", port)
	if err != nil {
		t.Fatal(err)
	}
	key := strings.SplitN(strings.SplitN(text, `OPENAI_API_KEY="`, 2)[1], `"`, 2)[0]
	b, err := c.Bridge(ctx, port)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Start(); err != nil {
		t.Fatal(err)
	}
	defer b.Shutdown(context.Background())
	req, _ := http.NewRequestWithContext(ctx, "POST", b.URL()+"/v1/chat/completions",
		strings.NewReader(`{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"hi from a tool"}]}`))
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 || resp.Header.Get("X-Starlight-Route") != "local" {
		t.Fatalf("bridge request %d %s", resp.StatusCode, body)
	}
	var client string
	_ = f.pool.QueryRow(ctx, "SELECT client FROM usage_events ORDER BY created_at DESC LIMIT 1").Scan(&client)
	if !strings.HasPrefix(client, "connect/") {
		t.Fatalf("usage not attributed to the Connect app: %q", client)
	}
}

func TestLiveThroughputReachesStarlight(t *testing.T) {
	f := startFleet(t)
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		var tps float64
		_ = f.pool.QueryRow(f.ctx, "SELECT coalesce((live->>'tokens_per_second')::float8, 0) FROM replicas WHERE live_at > now() - interval '15 seconds' LIMIT 1").Scan(&tps)
		if tps > 0 {
			return
		}
		time.Sleep(300 * time.Millisecond)
	}
	t.Fatal("no live tokens-per-second report reached Starlight")
}

func freePort(t *testing.T) int {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}
