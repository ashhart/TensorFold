package e2e_test

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/admission"
	"github.com/ashhart/TensorFold/starlight/internal/config"
	"github.com/ashhart/TensorFold/starlight/internal/control"
	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/keyring"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/secret"
	"github.com/ashhart/TensorFold/starlight/internal/db"
	"github.com/ashhart/TensorFold/starlight/internal/identity/directory"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/node/agent"
	"github.com/ashhart/TensorFold/starlight/internal/node/enroll"
	"github.com/ashhart/TensorFold/starlight/internal/node/state"
	"github.com/ashhart/TensorFold/starlight/internal/scheduler"
	"github.com/ashhart/TensorFold/starlight/internal/server"
	"github.com/ashhart/TensorFold/starlight/internal/testutil"
	"github.com/ashhart/TensorFold/starlight/internal/testutil/fakeengine"
)

func TestMain(m *testing.M) {
	code := m.Run()
	fakeengine.RemoveBinary()
	os.Exit(code)
}

// fleet is a running installation: Starlight in process, one enrolled node and its fake engine.
type fleet struct {
	agents    sync.WaitGroup
	cfg       config.Config
	t         *testing.T
	ctx       context.Context
	pool      *pgxpool.Pool
	starlight *server.Starlight
	public    *httptest.Server
	user      directory.User
	project   string
	model     string
	key       string
	nodeURL   string
}

func exec(t *testing.T, pool *pgxpool.Pool, sql string, args ...any) {
	t.Helper()
	if _, err := pool.Exec(context.Background(), sql, args...); err != nil {
		t.Fatalf("%s: %v", sql, err)
	}
}

// startFleet runs Starlight and one enrolled node with a ready replica of the catalogue model.
func startFleet(t *testing.T) *fleet {
	f := startStarlight(t, nil)
	f.enrolNode(f.nodeURL)
	return f
}

// startStarlight runs Starlight in process with the test catalogue but no nodes; edit adjusts the configuration.
func startStarlight(t *testing.T, edit func(*config.Config)) *fleet {
	pool := testutil.DB(t)
	ctx, cancel := context.WithCancel(context.Background())
	master := keyring.GenerateMasterKey()
	m, _ := keyring.LoadMaster(master, "")
	if err := db.Tx(ctx, pool, func(tx pgx.Tx) error {
		_, _, err := server.Init(ctx, tx, keyring.New(m), server.InitOptions{OrgName: "Example Corp"})
		return err
	}); err != nil {
		t.Fatal(err)
	}
	var handler http.Handler = http.NotFoundHandler()
	public := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { handler.ServeHTTP(w, r) }))
	cfg := config.Defaults()
	cfg.PublicURL, cfg.Database.URL, cfg.Keys.MasterKey = public.URL, pool.Config().ConnString(), master
	cfg.Sessions.AuthCache.Duration = 100 * time.Millisecond
	cfg.Gateway.QueueInteractive.Duration, cfg.Gateway.QueueStandard.Duration = 5*time.Second, 5*time.Second
	if edit != nil {
		edit(&cfg)
	}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	tw, err := server.Open(ctx, cfg, log)
	if err != nil {
		t.Fatal(err)
	}
	handler = tw.Handler()
	nodeURL, closeNodes := serveTLS(t, tw.Control.Handler(), tw.Control.TLSConfig())
	go tw.Control.Listen(ctx)
	go tw.Admitter.Waker.Run(ctx, tw.Pool)
	go admission.Heartbeat(ctx, tw.Pool, tw.Instance, "test", "test")
	ctrl := &scheduler.Controller{Pool: tw.Pool, Instance: tw.Instance, Every: 300 * time.Millisecond}
	go (&scheduler.Elector{Pool: tw.Pool, Instance: tw.Instance}).Run(ctx, ctrl.Lead)
	f := &fleet{cfg: cfg, t: t, ctx: ctx, pool: pool, starlight: tw, public: public, nodeURL: nodeURL}
	f.catalogue()
	t.Cleanup(func() {
		cancel()
		stopped := make(chan struct{})
		go func() { f.agents.Wait(); close(stopped) }()
		select {
		case <-stopped:
		case <-time.After(30 * time.Second):
			t.Error("node agents did not stop their engines within 30 seconds")
		}
		closeNodes()
		public.Close()
		tw.Close()
	})
	return f
}

func (f *fleet) catalogue() {
	t, ctx := f.t, f.ctx
	u, err := directory.CreateUser(ctx, f.pool, directory.NewUser{Email: "gus@example.com", DisplayName: "Gus", Source: "jit"})
	if err != nil {
		t.Fatal(err)
	}
	f.user, f.project, f.model = u, ids.New(), ids.New()
	exec(t, f.pool, "INSERT INTO projects (id, slug, name) VALUES ($1, 'eng', 'Engineering')", f.project)
	exec(t, f.pool, "INSERT INTO role_bindings (id, role, subject_type, subject_id, project_id) VALUES ($1, 'project_admin', 'user', $2, $3)", ids.New(), u.ID, f.project)
	exec(t, f.pool, "INSERT INTO models (id, name, display_name) VALUES ($1, 'qwen3.8-27b', 'Qwen3.8 27B')", f.model)
	exec(t, f.pool, "INSERT INTO model_grants (id, model_id, project_id) VALUES ($1, $2, $3)", ids.New(), f.model, f.project)
	exec(t, f.pool, "INSERT INTO model_aliases (alias, model_id) VALUES ('claude-sonnet-4-5', $1)", f.model)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"qwen3"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	spec, _ := json.Marshal(wire.Spec{Model: wire.Artifact{Kind: "path", Path: dir}, Parallel: 2})
	exec(t, f.pool, "INSERT INTO deployments (id, model_id, name, spec, mode, min_replicas) VALUES ($1, $2, 'qwen-mac', $3, 'warm', 1)", ids.New(), f.model, spec)
	tok, hash := secret.Generate(secret.APIKey)
	exec(t, f.pool, `INSERT INTO api_keys (id, public_id, secret_hash, name, owner_type, owner_id, project_id) VALUES ($1,$2,$3,'ci','user',$4,$5)`,
		ids.New(), secret.PublicID(tok), hash, u.ID, f.project)
	f.key = tok
}

// serveTLS serves h with cfg exactly as Starlight's node listener does; httptest would substitute its own certificate.
func serveTLS(t *testing.T, h http.Handler, cfg *tls.Config) (string, func()) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := &http.Server{Handler: h, TLSConfig: cfg, ReadHeaderTimeout: 10 * time.Second}
	go func() { _ = srv.ServeTLS(ln, "", "") }()
	return "https://" + ln.Addr().String(), func() { _ = srv.Close() }
}

func freeAddr(t *testing.T) string {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().String()
}

func (f *fleet) enrolNode(controlURL string) {
	f.enrolNamed(controlURL, "studio-1", 1)
}

// enrolNamed enrols and runs one more node agent, then waits until ready replicas number at least ready.
func (f *fleet) enrolNamed(controlURL, name string, ready int) {
	f.enrolWith(controlURL, name, ready, nodeOptions{})
}

// nodeOptions lets a test present a node as other hardware and give its engines extra settings.
type nodeOptions struct {
	facts func(context.Context) wire.Facts
	env   []string
}

func (f *fleet) enrolWith(controlURL, name string, ready int, o nodeOptions) {
	t := f.t
	tok, _, err := control.CreateToken(f.ctx, f.pool, control.TokenSpec{AutoApprove: true, MaxClass: "confidential"})
	if err != nil {
		t.Fatal(err)
	}
	stateDir := filepath.Join(t.TempDir(), "node")
	dir, err := state.Open(stateDir)
	if err != nil {
		t.Fatal(err)
	}
	addr := freeAddr(t)
	if _, _, err := enroll.Run(f.ctx, dir, enroll.Options{Server: controlURL, Token: tok, Name: name, Address: addr,
		CAPin: f.starlight.Trust.Fingerprint(), Facts: wire.Facts{Platform: "darwin", Arch: "arm64", Backend: "metal", MemoryBytes: 256 << 30}}); err != nil {
		t.Fatal(err)
	}
	f.agents.Add(1)
	go func() {
		defer f.agents.Done()
		err := agent.Run(f.ctx, agent.Config{StateDir: stateDir, Listen: addr, Advertise: addr, Engine: fakeengine.Binary(t), ModelsDir: t.TempDir(),
			StartupTimeout: time.Minute, DrainTimeout: time.Second, Env: append([]string{"FAKE_ENGINE_DELAY_MS=15", "FAKE_ENGINE_SYNTHETIC_TPS=120"}, o.env...),
			Facts: o.facts, Logger: slog.New(slog.NewTextHandler(io.Discard, nil))})
		if err != nil && f.ctx.Err() == nil {
			t.Errorf("agent: %v", err)
		}
	}()
	deadline := time.Now().Add(90 * time.Second)
	for time.Now().Before(deadline) {
		var n int
		_ = f.pool.QueryRow(f.ctx, "SELECT count(*) FROM replicas WHERE state = 'ready'").Scan(&n)
		if n >= ready {
			return
		}
		time.Sleep(200 * time.Millisecond)
	}
	var st, msg string
	_ = f.pool.QueryRow(f.ctx, "SELECT state, error FROM replicas LIMIT 1").Scan(&st, &msg)
	t.Fatalf("replica never became ready: state %q error %q", st, msg)
}

// gateway starts another Starlight instance on the same database, with a public URL of its own.
func (f *fleet) gateway() string {
	t := f.t
	var handler http.Handler = http.NotFoundHandler()
	public := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { handler.ServeHTTP(w, r) }))
	cfg := f.cfg
	cfg.PublicURL = public.URL
	tw, err := server.Open(f.ctx, cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatal(err)
	}
	handler = tw.Handler()
	ctx, cancel := context.WithCancel(f.ctx)
	go tw.Admitter.Waker.Run(ctx, tw.Pool)
	go admission.Heartbeat(ctx, tw.Pool, tw.Instance, "test", "test")
	t.Cleanup(func() {
		cancel()
		public.Close()
		tw.Close()
	})
	return public.URL
}

// call sends a JSON request to Starlight with a bearer credential.
func (f *fleet) call(method, path, token, body string, header ...string) (*http.Response, string) {
	f.t.Helper()
	req, _ := http.NewRequestWithContext(f.ctx, method, f.public.URL+path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	for i := 0; i+1 < len(header); i += 2 {
		req.Header.Set(header[i], header[i+1])
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		f.t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp, string(b)
}
