package agent_test

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/permit"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/pki"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
	"github.com/ashhart/TensorFold/starlight/internal/node/agent"
	"github.com/ashhart/TensorFold/starlight/internal/node/enroll"
	"github.com/ashhart/TensorFold/starlight/internal/node/nodetest"
	"github.com/ashhart/TensorFold/starlight/internal/node/state"
	"github.com/ashhart/TensorFold/starlight/internal/testutil/fakeengine"
)

func TestMain(m *testing.M) {
	code := m.Run()
	fakeengine.RemoveBinary()
	os.Exit(code)
}

func freePort(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	return ln.Addr().String()
}

func workloadCert(t *testing.T, ca *pki.CA, kind pki.Kind, id string) tls.Certificate {
	t.Helper()
	key, _, err := pki.NewKey()
	if err != nil {
		t.Fatal(err)
	}
	_, cert, err := ca.Issue(kind, id, &key.PublicKey, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	return tls.Certificate{Certificate: [][]byte{cert.Raw}, PrivateKey: key, Leaf: cert}
}

func call(t *testing.T, ca *pki.CA, client tls.Certificate, nodeID, addr, replica, tok, body string) (*http.Response, error) {
	t.Helper()
	return send(t, ca, client, nodeID, addr, replica, "POST", "/v1/chat/completions", tok, body)
}

func send(t *testing.T, ca *pki.CA, client tls.Certificate, nodeID, addr, replica, method, path, tok, body string) (*http.Response, error) {
	t.Helper()
	cfg := pki.ClientConfig(func() (*tls.Certificate, error) { return &client, nil }, ca.Pool(), ca.TrustDomain, pki.Identity{Kind: pki.KindNode, ID: nodeID})
	c := &http.Client{Transport: &http.Transport{TLSClientConfig: cfg, ForceAttemptHTTP2: true}, Timeout: 20 * time.Second}
	req, _ := http.NewRequest(method, "https://"+addr+wire.ReplicaPrefix+replica+path, strings.NewReader(body))
	req.Header.Set(wire.PermitHeader, tok)
	req.Header.Set("Content-Type", "application/json")
	return c.Do(req)
}

func TestNodeEndToEnd(t *testing.T) {
	ctl := nodetest.NewControl(t)
	stateDir := filepath.Join(t.TempDir(), "state")
	d, err := state.Open(stateDir)
	if err != nil {
		t.Fatal(err)
	}
	addr := freePort(t)
	id, _, err := enroll.Run(context.Background(), d, enroll.Options{Server: ctl.URL(), Token: ctl.Token, Name: "gpu-01", Address: addr, CAPin: ctl.CAPin()})
	if err != nil {
		t.Fatal(err)
	}
	replica := ids.New()
	asg := nodetest.Assignment(replica, "fake-27b", nodetest.ModelDir(t), 2, 2)
	ctl.Set(2, "active", false, 0, []wire.Assignment{asg})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- agent.Run(ctx, agent.Config{StateDir: stateDir, Listen: addr, Advertise: addr, Engine: fakeengine.Binary(t),
			StartupTimeout: 20 * time.Second, DrainTimeout: 3 * time.Second, StatsInterval: 200 * time.Millisecond,
			Env: []string{"FAKE_ENGINE_SYNTHETIC_TPS=100"}, Logger: slog.New(slog.NewTextHandler(io.Discard, nil))})
	}()
	ctl.WaitSync(t, 40*time.Second, func(s wire.SyncRequest) bool {
		return len(s.Replicas) == 1 && s.Replicas[0].State == wire.StateReady && s.Replicas[0].MaxBatch == 2 && s.Address == addr
	})
	waitStats(t, ctl, replica)
	last := ctl.Syncs()[len(ctl.Syncs())-1]
	if last.Facts.AgentVersion == "" || last.Facts.EngineVersion != fakeengine.Version || last.Facts.MemoryBytes == 0 {
		t.Fatalf("facts %+v", last.Facts)
	}

	body := `{"model":"fake-27b","stream":true,"messages":[{"role":"user","content":"hello"}]}`
	claims := permit.Claims{RequestID: ids.New(), Node: id.NodeID, Replica: replica, Deployment: asg.DeploymentID, Revision: asg.Revision,
		Method: "POST", Path: "/v1/chat/completions", Priority: "interactive"}
	gw := workloadCert(t, ctl.CA, pki.KindGateway, "gw-1")
	resp, err := call(t, ctl.CA, gw, id.NodeID, addr, replica, nodetest.Permit(t, ctl.Signer, claims, []byte(body)), body)
	if err != nil {
		t.Fatal(err)
	}
	out, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 || !strings.Contains(string(out), "data: [DONE]") {
		t.Fatalf("gateway call: %d %s", resp.StatusCode, out)
	}

	storedResponseRoundTrip(t, ctl, gw, id.NodeID, addr, asg)

	impostor := workloadCert(t, ctl.CA, pki.KindNode, "node-x")
	claims.RequestID = ids.New()
	if resp, err := call(t, ctl.CA, impostor, id.NodeID, addr, replica, nodetest.Permit(t, ctl.Signer, claims, []byte(body)), body); err == nil {
		resp.Body.Close()
		t.Fatal("a node certificate was accepted as a gateway")
	}

	before := d.Path("node.pem")
	oldPEM, _ := os.ReadFile(before)
	ctl.Set(3, "active", false, time.Now().Add(-time.Minute).Unix(), []wire.Assignment{asg})
	deadline := time.Now().Add(20 * time.Second)
	for ctl.Renewals() == 0 && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
	if ctl.Renewals() == 0 {
		t.Fatal("agent did not renew when renew_after passed")
	}
	newPEM := oldPEM
	for string(newPEM) == string(oldPEM) && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
		newPEM, _ = os.ReadFile(before)
	}
	if string(newPEM) == string(oldPEM) {
		t.Fatal("renewed certificate not written")
	}
	claims.RequestID = ids.New()
	resp, err = call(t, ctl.CA, gw, id.NodeID, addr, replica, nodetest.Permit(t, ctl.Signer, claims, []byte(body)), body)
	if err != nil {
		t.Fatalf("call after renewal: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("after renewal: %d", resp.StatusCode)
	}

	ctl.Set(4, "active", false, 0, nil)
	ctl.WaitSync(t, 30*time.Second, func(s wire.SyncRequest) bool { return s.Generation == 4 && len(s.Replicas) == 0 })
	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("agent exit: %v", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("agent did not stop")
	}
	if raw, err := d.ReadStatus(); err != nil || !strings.Contains(string(raw), id.NodeID) {
		t.Fatalf("status file %s %v", raw, err)
	}
}

// storedResponseRoundTrip stores a response through the node and reads it back, which needs the full engine key.
func storedResponseRoundTrip(t *testing.T, ctl *nodetest.Control, gw tls.Certificate, nodeID, addr string, asg wire.Assignment) {
	t.Helper()
	claims := func(method, path string) permit.Claims {
		return permit.Claims{RequestID: ids.New(), Node: nodeID, Replica: asg.ReplicaID, Deployment: asg.DeploymentID, Revision: asg.Revision,
			Method: method, Path: path, Priority: "standard"}
	}
	body := `{"model":"fake-27b","input":"remember this"}`
	resp, err := send(t, ctl.CA, gw, nodeID, addr, asg.ReplicaID, "POST", "/v1/responses",
		nodetest.Permit(t, ctl.Signer, claims("POST", "/v1/responses"), []byte(body)), body)
	if err != nil {
		t.Fatal(err)
	}
	out, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	var stored struct{ ID string }
	if resp.StatusCode != 200 || json.Unmarshal(out, &stored) != nil || stored.ID == "" {
		t.Fatalf("stored response through the node: %d %s", resp.StatusCode, out)
	}
	path := "/v1/responses/" + stored.ID
	for _, method := range []string{"GET", "DELETE"} {
		resp, err := send(t, ctl.CA, gw, nodeID, addr, asg.ReplicaID, method, path, nodetest.Permit(t, ctl.Signer, claims(method, path), nil), "")
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatalf("%s %s through the node: %d", method, path, resp.StatusCode)
		}
	}
}

// waitStats waits until Starlight has received live throughput for the replica.
func waitStats(t *testing.T, ctl *nodetest.Control, replica string) {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		for _, rep := range ctl.Stats() {
			for _, r := range rep.Replicas {
				if r.ReplicaID == replica && r.TokensPerSecond > 0 && r.PromptTokensPerSecond > 0 {
					return
				}
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("no stats report with throughput for %s; got %+v", replica, ctl.Stats())
}
