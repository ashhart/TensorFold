package providers

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/catalog"
)

func TestEgressRefusesUnapprovedAndOffline(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, "ok") }))
	defer srv.Close()
	e := NewEgress()
	if _, err := e.Client().Get(srv.URL); err == nil || !strings.Contains(err.Error(), "not approved") {
		t.Fatalf("unapproved host: %v", err)
	}
	e.Allow(map[string]bool{"127.0.0.1": false}, false)
	if _, err := e.Client().Get(srv.URL); err == nil {
		t.Fatal("plain http to a public route allowed")
	}
	e.Allow(map[string]bool{"127.0.0.1": true}, false)
	resp, err := e.Client().Get(srv.URL)
	if err != nil {
		t.Fatalf("internal provider refused: %v", err)
	}
	resp.Body.Close()
	e.Allow(map[string]bool{"127.0.0.1": true}, true)
	if _, err := e.Client().Get(srv.URL); err == nil || !strings.Contains(err.Error(), "offline") {
		t.Fatalf("offline egress: %v", err)
	}
}

func TestDoSetsProviderAuth(t *testing.T) {
	var got http.Header
	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got, path = r.Header.Clone(), r.URL.Path
		io.WriteString(w, "{}")
	}))
	defer srv.Close()
	e := NewEgress()
	e.Allow(map[string]bool{"127.0.0.1": true}, false)
	creds := &Credentials{hits: map[string]cred{"p1": {value: "sk-test", headers: map[string]string{"OpenAI-Organization": "org-x", "Authorization": "evil"}, expires: farFuture()}}}
	c := &Client{Egress: e, Creds: creds}
	route := catalog.ProviderRoute{ProviderID: "p1", Kind: "anthropic", BaseURL: srv.URL, Protocols: []string{Messages}}
	in := http.Header{"Anthropic-Beta": {"tools-2024"}}
	resp, err := c.Do(context.Background(), route, Messages, []byte(`{}`), in)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if path != "/v1/messages" || got.Get("X-Api-Key") != "sk-test" || got.Get("Anthropic-Version") != "2023-06-01" || got.Get("Anthropic-Beta") != "tools-2024" {
		t.Fatalf("anthropic headers %v path %s", got, path)
	}
	if got.Get("Authorization") != "" || got.Get("Openai-Organization") != "org-x" {
		t.Fatalf("header handling %v", got)
	}
	if _, err := c.Do(context.Background(), route, Responses, nil, in); err != ErrProtocol {
		t.Fatalf("protocol mismatch: %v", err)
	}
}

func farFuture() time.Time { return time.Now().Add(time.Hour) }

func TestInternalProviderDoesNotOpenPrivateAddressesToOthers(t *testing.T) {
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, "ok") }))
	defer srv.Close()
	_, port, _ := net.SplitHostPort(srv.Listener.Addr().String())
	e := NewEgress()
	e.Allow(map[string]bool{"vllm.internal.example": true, "localhost": false}, false)
	resp, err := e.Client().Get("https://localhost:" + port)
	if err == nil {
		resp.Body.Close()
	}
	if err == nil || !strings.Contains(err.Error(), "refusing internal address") {
		t.Fatalf("a public provider dialled a loopback address because another provider is internal: %v", err)
	}
}
