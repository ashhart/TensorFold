package core

import (
	"context"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"

	"github.com/ashhart/TensorFold/starlight/internal/connect/fakestarlight"
	"github.com/ashhart/TensorFold/starlight/internal/connect/managed"
	"github.com/ashhart/TensorFold/starlight/internal/connect/profile"
	"github.com/ashhart/TensorFold/starlight/internal/connect/store"
)

func browser(u string) error {
	go func() {
		if resp, err := http.Get(u); err == nil {
			resp.Body.Close()
		}
	}()
	return nil
}

func freePort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port
}

func TestLoginSetupAndBridgeEndToEnd(t *testing.T) {
	t.Setenv(store.BackendEnv, "file")
	f := fakestarlight.New(t)
	dir := t.TempDir()
	c, err := New(dir, profile.Profile{}, managed.Policy{})
	if err != nil {
		t.Fatal(err)
	}
	if err := c.UseServer(f.URL); err != nil {
		t.Fatal(err)
	}
	me, err := c.Login(context.Background(), LoginOptions{Open: browser, Out: io.Discard})
	if err != nil || me.OrgName != "Example Org" {
		t.Fatalf("login = %+v, %v", me, err)
	}
	reloaded, err := New(dir, mustProfile(t, dir), managed.Policy{})
	if err != nil {
		t.Fatal(err)
	}
	api, err := reloaded.API(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if models, err := api.Models(context.Background(), ""); err != nil || len(models) != 2 {
		t.Fatalf("models after reload = %v, %v", models, err)
	}
	port := freePort(t)
	text, replaced, err := reloaded.Setup("openai", "", "", "posix", port)
	if err != nil || replaced || !strings.Contains(text, "OPENAI_API_KEY") {
		t.Fatalf("setup = %q, %v, %v", text, replaced, err)
	}
	key := between(text, `OPENAI_API_KEY="`, `"`)
	b, err := reloaded.Bridge(context.Background(), port)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Start(); err != nil {
		t.Fatal(err)
	}
	defer b.Shutdown(context.Background())
	if !reloaded.BridgeRunning(context.Background(), port) {
		t.Fatal("bridge health check failed")
	}
	req, _ := http.NewRequest("POST", b.URL()+"/v1/chat/completions", strings.NewReader(`{"model":"qwen3.8-27b"}`))
	req.Header.Set("Authorization", "Bearer "+key)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 || len(f.Requests()) != 1 || f.Requests()[0].Header.Get("X-Starlight-Client") != "connect/openai" {
		t.Fatalf("bridge forward %d %+v", resp.StatusCode, f.Requests())
	}
	if _, replaced, _ := reloaded.Setup("openai", "", "", "posix", port); !replaced {
		t.Fatal("second setup did not report the rotation")
	}
	if err := reloaded.Logout(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := api.Me(context.Background()); err == nil {
		t.Fatal("still signed in after logout")
	}
}

func TestManagedServerIsLocked(t *testing.T) {
	t.Setenv(store.BackendEnv, "file")
	c, err := New(t.TempDir(), profile.Profile{}, managed.Policy{ServerURL: "https://starlight.example", DisableBridge: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := c.UseServer("https://other.example"); err == nil || !strings.Contains(err.Error(), "manages") {
		t.Fatalf("changed a managed server: %v", err)
	}
	if err := c.UseServer("https://starlight.example/"); err != nil {
		t.Fatalf("same server refused: %v", err)
	}
	if _, err := c.Bridge(context.Background(), 0); err == nil || !strings.Contains(err.Error(), "turned off") {
		t.Fatalf("bridge started against policy: %v", err)
	}
}

func mustProfile(t *testing.T, dir string) profile.Profile {
	t.Helper()
	p, err := profile.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func between(s, start, end string) string {
	_, rest, _ := strings.Cut(s, start)
	v, _, _ := strings.Cut(rest, end)
	return v
}
