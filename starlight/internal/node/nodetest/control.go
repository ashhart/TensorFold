// Package nodetest provides a minimal Starlight control endpoint for node agent tests.
package nodetest

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/permit"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/pki"
	"github.com/ashhart/TensorFold/starlight/internal/ids"
)

// Control is an in-process Starlight control endpoint with its own workload CA and permit key.
type Control struct {
	CA       *pki.CA
	Server   *httptest.Server
	Token    string
	Signer   permit.Signer
	Validity time.Duration

	mu          sync.Mutex
	nodeID      string
	generation  int64
	assignments []wire.Assignment
	nodeStatus  string
	offline     bool
	renewAfter  int64
	syncs       []wire.SyncRequest
	renewals    int
	stats       []wire.StatsReport
	changed     chan struct{}
	// EnrollHook may rewrite an enrolment response, for negative tests.
	EnrollHook func(*wire.EnrollResponse)
}

// NewControl starts a TLS control endpoint; enrolment needs Token, sync and renew need a node certificate.
func NewControl(t testing.TB) *Control {
	t.Helper()
	ca, keyDER, err := pki.NewCA("starlight.example", 24*time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	if ca, err = pki.LoadCA(ca.CertPEM, keyDER); err != nil {
		t.Fatal(err)
	}
	_, priv, _ := ed25519.GenerateKey(rand.Reader)
	c := &Control{CA: ca, Token: "sle_test_token", Signer: permit.Signer{KID: "k1", Key: priv}, Validity: time.Hour,
		nodeStatus: "active", generation: 1, changed: make(chan struct{})}
	mux := http.NewServeMux()
	mux.HandleFunc("POST "+wire.PathEnroll, c.enroll)
	mux.HandleFunc("POST "+wire.PathSync, c.sync)
	mux.HandleFunc("POST "+wire.PathRenew, c.renew)
	mux.HandleFunc("POST "+wire.PathStats, c.recordStats)
	leaf, err := c.controlLeaf()
	if err != nil {
		t.Fatal(err)
	}
	c.Server = httptest.NewUnstartedServer(mux)
	c.Server.TLS = &tls.Config{ClientAuth: tls.RequestClientCert, MinVersion: tls.VersionTLS12, Certificates: []tls.Certificate{leaf}}
	c.Server.StartTLS()
	t.Cleanup(c.Server.Close)
	return c
}

// URL is the endpoint's base URL.
func (c *Control) URL() string { return c.Server.URL }

// CAPin is the SHA-256 of the workload CA certificate, as the console prints it for --ca-sha256.
func (c *Control) CAPin() string {
	sum := sha256.Sum256(c.CA.Cert.Raw)
	return hex.EncodeToString(sum[:])
}

// CABundle is the workload CA in PEM, for --ca-file.
func (c *Control) CABundle() []byte { return c.CA.CertPEM }

// controlLeaf issues the listener's own certificate, sent with the CA as Starlight's node listener does.
func (c *Control) controlLeaf() (tls.Certificate, error) {
	key, _, err := pki.NewKey()
	if err != nil {
		return tls.Certificate{}, err
	}
	_, cert, err := c.CA.Issue(pki.KindControl, "control-1", &key.PublicKey, time.Hour)
	if err != nil {
		return tls.Certificate{}, err
	}
	return tls.Certificate{Certificate: [][]byte{cert.Raw, c.CA.Cert.Raw}, PrivateKey: key, Leaf: cert}, nil
}

// PermitKeys publishes the signer's public key.
func (c *Control) PermitKeys() []wire.PermitKey {
	return []wire.PermitKey{{KID: c.Signer.KID, PublicKey: c.Signer.Key.Public().(ed25519.PublicKey)}}
}

// NodeID returns the id issued at enrolment.
func (c *Control) NodeID() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.nodeID
}

// Set replaces the desired state and wakes waiting syncs.
func (c *Control) Set(gen int64, status string, offline bool, renewAfter int64, asgs []wire.Assignment) {
	c.mu.Lock()
	c.generation, c.nodeStatus, c.offline, c.renewAfter, c.assignments = gen, status, offline, renewAfter, asgs
	close(c.changed)
	c.changed = make(chan struct{})
	c.mu.Unlock()
}

// Syncs returns every sync request received so far.
func (c *Control) Syncs() []wire.SyncRequest {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]wire.SyncRequest(nil), c.syncs...)
}

// Stats returns every stats report received so far.
func (c *Control) Stats() []wire.StatsReport {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]wire.StatsReport(nil), c.stats...)
}

// Renewals counts certificate renewals served.
func (c *Control) Renewals() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.renewals
}

// WaitSync waits for a sync request satisfying ok.
func (c *Control) WaitSync(t testing.TB, timeout time.Duration, ok func(wire.SyncRequest) bool) wire.SyncRequest {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		for _, s := range c.Syncs() {
			if ok(s) {
				return s
			}
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("no matching sync within %s; last syncs: %+v", timeout, last(c.Syncs()))
	return wire.SyncRequest{}
}

func last(s []wire.SyncRequest) any {
	if len(s) == 0 {
		return nil
	}
	return s[len(s)-1]
}

func (c *Control) enroll(w http.ResponseWriter, r *http.Request) {
	var req wire.EnrollRequest
	if json.NewDecoder(r.Body).Decode(&req) != nil || req.Token != c.Token {
		writeErr(w, http.StatusUnauthorized, "invalid_token")
		return
	}
	id := ids.New()
	certPEM, _, err := c.CA.SignCSR(pki.KindNode, id, []byte(req.CSR), c.Validity)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "invalid_csr")
		return
	}
	c.mu.Lock()
	c.nodeID = id
	c.mu.Unlock()
	resp := wire.EnrollResponse{NodeID: id, Status: "active", Certificate: string(certPEM), CA: string(c.CA.CertPEM),
		TrustDomain: c.CA.TrustDomain, PermitKeys: c.PermitKeys()}
	if c.EnrollHook != nil {
		c.EnrollHook(&resp)
	}
	writeJSON(w, resp)
}

// identity accepts only a node certificate from this CA naming the enrolled node.
func (c *Control) identity(r *http.Request) bool {
	if r.TLS == nil || len(r.TLS.PeerCertificates) == 0 {
		return false
	}
	id, _, err := pki.Verify(r.TLS.PeerCertificates, c.CA.Pool(), c.CA.TrustDomain, x509.ExtKeyUsageClientAuth)
	return err == nil && id.Kind == pki.KindNode && id.ID == c.NodeID()
}

func (c *Control) sync(w http.ResponseWriter, r *http.Request) {
	if !c.identity(r) {
		writeErr(w, http.StatusUnauthorized, "node_certificate_required")
		return
	}
	var req wire.SyncRequest
	if json.NewDecoder(r.Body).Decode(&req) != nil {
		writeErr(w, http.StatusBadRequest, "invalid_json")
		return
	}
	c.mu.Lock()
	c.syncs = append(c.syncs, req)
	gen, wake := c.generation, c.changed
	c.mu.Unlock()
	if req.Generation == gen {
		select {
		case <-wake:
		case <-r.Context().Done():
			return
		case <-time.After(2 * time.Second):
		}
	}
	c.mu.Lock()
	resp := wire.SyncResponse{Generation: c.generation, NodeStatus: c.nodeStatus, Offline: c.offline, Assignments: c.assignments,
		PermitKeys: c.PermitKeys(), RenewAfter: c.renewAfter}
	c.mu.Unlock()
	writeJSON(w, resp)
}

func (c *Control) renew(w http.ResponseWriter, r *http.Request) {
	if !c.identity(r) {
		writeErr(w, http.StatusUnauthorized, "node_certificate_required")
		return
	}
	var req wire.RenewRequest
	if json.NewDecoder(r.Body).Decode(&req) != nil {
		writeErr(w, http.StatusBadRequest, "invalid_json")
		return
	}
	certPEM, _, err := c.CA.SignCSR(pki.KindNode, c.NodeID(), []byte(req.CSR), c.Validity)
	if err != nil {
		writeErr(w, http.StatusBadRequest, "invalid_csr")
		return
	}
	c.mu.Lock()
	c.renewals++
	c.renewAfter = 0
	c.mu.Unlock()
	writeJSON(w, wire.RenewResponse{Certificate: string(certPEM)})
}

func (c *Control) recordStats(w http.ResponseWriter, r *http.Request) {
	if !c.identity(r) {
		writeErr(w, http.StatusUnauthorized, "node_certificate_required")
		return
	}
	var rep wire.StatsReport
	if json.NewDecoder(r.Body).Decode(&rep) != nil {
		writeErr(w, http.StatusBadRequest, "invalid_json")
		return
	}
	c.mu.Lock()
	c.stats = append(c.stats, rep)
	c.mu.Unlock()
	w.WriteHeader(http.StatusNoContent)
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, status int, code string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"code": code, "message": code}})
}
