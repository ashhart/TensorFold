package gateway

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"net"
	"net/http"
	"net/http/httptrace"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ashhart/TensorFold/starlight/internal/admission"
	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/permit"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/pki"
)

// Nodes dispatches to replicas over mTLS, naming each node by SPIFFE ID and carrying a fresh permit.
type Nodes struct {
	TrustDomain string
	Roots       *x509.CertPool
	Cert        func() (*tls.Certificate, error)
	Signer      func() (permit.Signer, error)
	mu          sync.Mutex
	clients     map[string]*http.Client
}

func (n *Nodes) client(nodeID string) *http.Client {
	n.mu.Lock()
	defer n.mu.Unlock()
	if c, ok := n.clients[nodeID]; ok {
		return c
	}
	if n.clients == nil {
		n.clients = map[string]*http.Client{}
	}
	cfg := pki.ClientConfig(n.Cert, n.Roots, n.TrustDomain, pki.Identity{Kind: pki.KindNode, ID: nodeID})
	t := &http.Transport{
		TLSClientConfig:       cfg,
		DialContext:           (&net.Dialer{Timeout: 5 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConnsPerHost:   64,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   5 * time.Second,
		ResponseHeaderTimeout: 30 * time.Minute,
	}
	c := &http.Client{Transport: t}
	n.clients[nodeID] = c
	return c
}

// Forget closes idle connections to a node, for example after its certificate changes.
func (n *Nodes) Forget(nodeID string) {
	n.mu.Lock()
	defer n.mu.Unlock()
	if c, ok := n.clients[nodeID]; ok {
		c.CloseIdleConnections()
		delete(n.clients, nodeID)
	}
}

// Dispatch is a node request and whether it ever reached the node.
type Dispatch struct {
	Resp *http.Response
	Sent bool
}

// Do sends one engine request for a lease; Sent is false when the request never left this gateway.
func (n *Nodes) Do(ctx context.Context, l *admission.Lease, method, path string, body []byte, subject, priority string, maxTokens int) (Dispatch, error) {
	signer, err := n.Signer()
	if err != nil {
		return Dispatch{}, err
	}
	sum := sha256.Sum256(body)
	now := time.Now()
	tok, err := signer.Sign(permit.Claims{RequestID: l.ID, Node: l.Target.NodeID, Replica: l.Target.ReplicaID, Deployment: l.Target.DeploymentID,
		Revision: l.Target.Revision, Method: method, Path: path, BodySHA256: hex.EncodeToString(sum[:]), IssuedAt: now.UnixMilli(),
		Expires: now.Add(30 * time.Second).UnixMilli(), Subject: subject, Priority: priority, MaxTokens: maxTokens})
	if err != nil {
		return Dispatch{}, err
	}
	var d Dispatch
	var sent atomic.Bool
	trace := &httptrace.ClientTrace{WroteRequest: func(httptrace.WroteRequestInfo) { sent.Store(true) }}
	req, err := http.NewRequestWithContext(httptrace.WithClientTrace(ctx, trace), method,
		"https://"+l.Target.NodeAddress+wire.ReplicaPrefix+l.Target.ReplicaID+path, bytes.NewReader(body))
	if err != nil {
		return d, err
	}
	req.Header.Set(wire.PermitHeader, tok)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	d.Resp, err = n.client(l.Target.NodeID).Do(req)
	d.Sent = sent.Load() || d.Resp != nil
	return d, err
}

// refused reports a node answer that proves the engine never started the request, so another replica may take it.
func refused(resp *http.Response) bool {
	if resp == nil {
		return false
	}
	return resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode == http.StatusServiceUnavailable && resp.Header.Get("Starlight-Refused") == "1"
}
