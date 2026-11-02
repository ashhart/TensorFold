package proxy

import (
	"bufio"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/permit"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/node/engine"
	"github.com/ashhart/TensorFold/starlight/internal/node/nodetest"
	"github.com/ashhart/TensorFold/starlight/internal/testutil/fakeengine"
)

type target struct {
	mu       sync.Mutex
	base     string
	keys     engine.Keys
	slots    int
	inFlight int
	err      error
}

func (t *target) Deployment() (string, int)       { return "dep-1", 1 }
func (t *target) Endpoint() (string, engine.Keys) { return t.base, t.keys }
func (t *target) Acquire() (func(), error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.err != nil {
		return nil, t.err
	}
	if t.inFlight >= t.slots {
		return nil, engine.ErrBusy
	}
	t.inFlight++
	return func() { t.mu.Lock(); t.inFlight--; t.mu.Unlock() }, nil
}

type backend map[string]Target

func (b backend) Lookup(id string) (Target, bool) {
	t, ok := b[id]
	return t, ok
}

type fixture struct {
	srv     *Server
	signer  permit.Signer
	replica string
	tgt     *target
	eng     *fakeengine.Engine
}

func newFixture(t *testing.T, delay time.Duration) *fixture {
	t.Helper()
	eng := fakeengine.New(fakeengine.Options{Name: "m", Delay: delay, Keys: []fakeengine.Key{
		{Label: "starlight", Scopes: []string{"inference"}, Secret: "engine-key"}, {Label: "starlight-responses", Secret: "full-key"}}})
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	hs := &http.Server{Handler: eng}
	go hs.Serve(ln)
	t.Cleanup(func() { hs.Close() })
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	replica := ids.New()
	tgt := &target{base: "http://" + ln.Addr().String(), keys: engine.Keys{Inference: "engine-key", Responses: "full-key"}, slots: 1}
	srv := &Server{NodeID: "node-1", Verifier: permit.NewVerifier(map[string]ed25519.PublicKey{"k1": pub}), Replay: permit.NewReplay(),
		Backend: backend{replica: tgt}, Transport: NewTransport()}
	return &fixture{srv: srv, signer: permit.Signer{KID: "k1", Key: priv}, replica: replica, tgt: tgt, eng: eng}
}

func (f *fixture) claims(path string) permit.Claims {
	return permit.Claims{RequestID: ids.New(), Node: "node-1", Replica: f.replica, Deployment: "dep-1", Revision: 1, Method: "POST", Path: path, Priority: "standard"}
}

func (f *fixture) request(t *testing.T, method, path string, body string, tok string) *http.Request {
	t.Helper()
	r := httptest.NewRequest(method, wire.ReplicaPrefix+f.replica+path, strings.NewReader(body))
	r.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{{}}}
	if tok != "" {
		r.Header.Set(wire.PermitHeader, tok)
	}
	r.Header.Set("Content-Type", "application/json")
	return r
}

const chatBody = `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi there"}]}`

func TestValidPermitStreams(t *testing.T) {
	f := newFixture(t, 0)
	tok := nodetest.Permit(t, f.signer, f.claims("/v1/chat/completions"), []byte(chatBody))
	r := f.request(t, "POST", "/v1/chat/completions", chatBody, tok)
	r.Header.Set("Origin", "https://evil.example")
	r.Header.Set("Cookie", "session=1")
	r.Header.Set("Authorization", "Bearer client-token")
	w := httptest.NewRecorder()
	f.srv.ServeHTTP(w, r)
	if w.Code != 200 || !strings.Contains(w.Body.String(), "data: [DONE]") || w.Header().Get(RefusedHeader) != "" {
		t.Fatalf("status %d refused=%q body %s", w.Code, w.Header().Get(RefusedHeader), w.Body)
	}
	if !strings.Contains(w.Body.String(), `"usage"`) {
		t.Fatal("usage chunk missing")
	}
	last := f.eng.Stats().Last
	if last == nil || last.Label != "starlight" || last.Headers["Origin"] != "" || last.Headers["Cookie"] != "" {
		t.Fatalf("engine saw %+v", last)
	}
	if f.tgt.inFlight != 0 {
		t.Fatal("slot not released")
	}
}

func TestRefusals(t *testing.T) {
	f := newFixture(t, 0)
	path := "/v1/chat/completions"
	mk := func(mut func(*permit.Claims)) string {
		c := f.claims(path)
		mut(&c)
		return nodetest.Permit(t, f.signer, c, []byte(chatBody))
	}
	foreign := nodetest.Permit(t, newFixture(t, 0).signer, f.claims(path), []byte(chatBody))
	used := mk(func(*permit.Claims) {})
	f.srv.ServeHTTP(httptest.NewRecorder(), f.request(t, "POST", path, chatBody, used))
	cases := []struct {
		name, method, path, body, tok string
		status                        int
		code                          string
	}{
		{"missing permit", "POST", path, chatBody, "", 401, "permit_required"},
		{"wrong node", "POST", path, chatBody, mk(func(c *permit.Claims) { c.Node = "node-2" }), 403, "permit_mismatch"},
		{"wrong replica", "POST", path, chatBody, mk(func(c *permit.Claims) { c.Replica = ids.New() }), 403, "permit_mismatch"},
		{"wrong method", "GET", "/v1/responses/resp_1", "", mk(func(c *permit.Claims) { c.Path = "/v1/responses/resp_1" }), 403, "permit_mismatch"},
		{"wrong path", "POST", "/v1/completions", chatBody, mk(func(*permit.Claims) {}), 403, "permit_mismatch"},
		{"wrong body", "POST", path, `{"model":"m","max_tokens":100000}`, mk(func(*permit.Claims) {}), 403, "permit_mismatch"},
		{"wrong revision", "POST", path, chatBody, mk(func(c *permit.Claims) { c.Revision = 2 }), 403, "permit_mismatch"},
		{"expired", "POST", path, chatBody, mk(func(c *permit.Claims) {
			c.IssuedAt = time.Now().Add(-2 * time.Minute).UnixMilli()
			c.Expires = time.Now().Add(-time.Minute).UnixMilli()
		}), 403, "permit_expired"},
		{"foreign signer", "POST", path, chatBody, foreign, 403, "permit_invalid"},
		{"replayed", "POST", path, chatBody, used, 403, "permit_replayed"},
		{"route not allowed", "POST", "/v1/slide/learn", "{}", mk(func(c *permit.Claims) { c.Path = "/v1/slide/learn" }), 404, "route_not_allowed"},
	}
	for _, c := range cases {
		w := httptest.NewRecorder()
		f.srv.ServeHTTP(w, f.request(t, c.method, c.path, c.body, c.tok))
		if w.Code != c.status || !strings.Contains(w.Body.String(), `"code":"`+c.code+`"`) || w.Header().Get(RefusedHeader) != "1" {
			t.Errorf("%s: %d %s refused=%q", c.name, w.Code, w.Body, w.Header().Get(RefusedHeader))
		}
	}
	if n := f.eng.Stats().Requests; n != 1 {
		t.Fatalf("refused requests reached the engine: %d requests", n)
	}
}

func TestBusyDrainingAndUnknown(t *testing.T) {
	f := newFixture(t, 0)
	path := "/v1/chat/completions"
	f.tgt.inFlight = 1
	w := httptest.NewRecorder()
	f.srv.ServeHTTP(w, f.request(t, "POST", path, chatBody, nodetest.Permit(t, f.signer, f.claims(path), []byte(chatBody))))
	if w.Code != 429 || w.Header().Get("Retry-After") != "1" || w.Header().Get(RefusedHeader) != "1" || !strings.Contains(w.Body.String(), "replica_busy") {
		t.Fatalf("busy: %d %v %s", w.Code, w.Header(), w.Body)
	}
	f.tgt.inFlight, f.tgt.err = 0, engine.ErrDraining
	w = httptest.NewRecorder()
	f.srv.ServeHTTP(w, f.request(t, "POST", path, chatBody, nodetest.Permit(t, f.signer, f.claims(path), []byte(chatBody))))
	if w.Code != 503 || w.Header().Get(RefusedHeader) != "1" || !strings.Contains(w.Body.String(), "replica_draining") {
		t.Fatalf("draining: %d %s", w.Code, w.Body)
	}
	c := f.claims(path)
	c.Replica = ids.New()
	r := httptest.NewRequest("POST", wire.ReplicaPrefix+c.Replica+path, strings.NewReader(chatBody))
	r.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{{}}}
	r.Header.Set(wire.PermitHeader, nodetest.Permit(t, f.signer, c, []byte(chatBody)))
	w = httptest.NewRecorder()
	f.srv.ServeHTTP(w, r)
	if w.Code != 503 || w.Header().Get(RefusedHeader) != "1" || !strings.Contains(w.Body.String(), "replica_not_ready") {
		t.Fatalf("unknown: %d %s", w.Code, w.Body)
	}
}

func TestDeadEngineIsRefusedButUncertainFailureIsNot(t *testing.T) {
	f := newFixture(t, 0)
	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	f.tgt.base = "http://" + ln.Addr().String()
	ln.Close()
	path := "/v1/chat/completions"
	w := httptest.NewRecorder()
	f.srv.ServeHTTP(w, f.request(t, "POST", path, chatBody, nodetest.Permit(t, f.signer, f.claims(path), []byte(chatBody))))
	if w.Code != 503 || w.Header().Get(RefusedHeader) != "1" {
		t.Fatalf("dial failure: %d refused=%q", w.Code, w.Header().Get(RefusedHeader))
	}
	hang, _ := net.Listen("tcp", "127.0.0.1:0")
	defer hang.Close()
	go func() {
		c, err := hang.Accept()
		if err == nil {
			_, _ = io.ReadAll(io.LimitReader(c, 1))
			c.Close()
		}
	}()
	f.tgt.base = "http://" + hang.Addr().String()
	w = httptest.NewRecorder()
	f.srv.ServeHTTP(w, f.request(t, "POST", path, chatBody, nodetest.Permit(t, f.signer, f.claims(path), []byte(chatBody))))
	if w.Code != 502 || w.Header().Get(RefusedHeader) != "" {
		t.Fatalf("mid-request failure: %d refused=%q", w.Code, w.Header().Get(RefusedHeader))
	}
}

func TestStreamIsIncrementalAndCancellationReachesEngine(t *testing.T) {
	f := newFixture(t, 150*time.Millisecond)
	ts := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{{}}}
		f.srv.ServeHTTP(w, r)
	}))
	ts.Start()
	defer ts.Close()
	path := "/v1/chat/completions"
	req, _ := http.NewRequest("POST", ts.URL+wire.ReplicaPrefix+f.replica+path, strings.NewReader(chatBody))
	req.Header.Set(wire.PermitHeader, nodetest.Permit(t, f.signer, f.claims(path), []byte(chatBody)))
	start := time.Now()
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	line, err := bufio.NewReader(resp.Body).ReadString('\n')
	if err != nil || !strings.HasPrefix(line, "data: ") {
		t.Fatalf("first line %q %v", line, err)
	}
	if time.Since(start) > 900*time.Millisecond {
		t.Fatalf("first event took %s; the stream was buffered", time.Since(start))
	}
	resp.Body.Close()
	deadline := time.Now().Add(3 * time.Second)
	for f.eng.Stats().InFlight != 0 && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if n := f.eng.Stats().InFlight; n != 0 {
		t.Fatalf("engine still has %d requests after the gateway left", n)
	}
	if n := f.eng.Stats().Requests; n != 1 {
		t.Fatalf("requests %d", n)
	}
}
