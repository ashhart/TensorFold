// Package control serves the node listener: enrolment, long-poll sync and certificate renewal.
package control

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ashhart/TensorFold/starlight/internal/control/wire"
	"github.com/ashhart/TensorFold/starlight/internal/crypto/pki"
	"github.com/ashhart/TensorFold/starlight/internal/httpx"
	"github.com/ashhart/TensorFold/starlight/internal/ratelimit"
	"github.com/ashhart/TensorFold/starlight/internal/trust"
)

// NodeCertValidity is how long node certificates last; nodes renew after two thirds.
const NodeCertValidity = 30 * 24 * time.Hour

// Server handles node traffic.
type Server struct {
	Pool    *pgxpool.Pool
	Trust   *trust.Trust
	Proxies *httpx.Proxies
	Log     *slog.Logger
	Poll    time.Duration
	RateKey []byte
	enrollR *ratelimit.Limiter
	waitMu  sync.Mutex
	waiters map[string][]chan struct{}
}

// Handler returns the node listener's routes.
func (s *Server) Handler() http.Handler {
	s.enrollR = ratelimit.Shared(s.Pool, s.RateKey, "enrol", 0.5, 10)
	mux := http.NewServeMux()
	mux.HandleFunc("POST "+wire.PathEnroll, s.enroll)
	mux.HandleFunc("POST "+wire.PathSync, s.sync)
	mux.HandleFunc("POST "+wire.PathRenew, s.renew)
	mux.HandleFunc("POST "+wire.PathStats, s.stats)
	return httpx.WithRequestID(mux)
}

// TLSConfig presents the control certificate and asks for, but does not require, a node certificate.
func (s *Server) TLSConfig() *tls.Config {
	return &tls.Config{
		MinVersion: tls.VersionTLS13,
		GetCertificate: func(*tls.ClientHelloInfo) (*tls.Certificate, error) {
			return s.Trust.ControlCert()
		},
		ClientAuth: tls.RequestClientCert,
		VerifyConnection: func(cs tls.ConnectionState) error {
			if len(cs.PeerCertificates) == 0 {
				return nil
			}
			id, _, err := pki.Verify(cs.PeerCertificates, s.Trust.Roots(), s.Trust.TrustDomain, x509.ExtKeyUsageClientAuth)
			if err != nil {
				return err
			}
			if id.Kind != pki.KindNode {
				return errors.New("control: only node certificates may connect")
			}
			return nil
		},
	}
}

var errNode = httpx.Errorf(http.StatusUnauthorized, "node_unauthorized", "a valid, unrevoked node certificate is required")

// node authenticates the calling node by certificate; revoked reports a known node whose access was withdrawn.
func (s *Server) node(r *http.Request) (string, *x509.Certificate, bool, error) {
	if r.TLS == nil || len(r.TLS.PeerCertificates) == 0 {
		return "", nil, false, errNode
	}
	id, cert, err := pki.Verify(r.TLS.PeerCertificates, s.Trust.Roots(), s.Trust.TrustDomain, x509.ExtKeyUsageClientAuth)
	if err != nil || id.Kind != pki.KindNode {
		return "", nil, false, errNode
	}
	var status string
	var certRevoked bool
	err = s.Pool.QueryRow(r.Context(), `SELECT n.status, c.revoked_at IS NOT NULL FROM node_certificates c JOIN nodes n ON n.id = c.node_id
		WHERE c.serial = $1 AND c.node_id = $2 AND c.not_after > now()`, cert.SerialNumber.Text(16), id.ID).Scan(&status, &certRevoked)
	if err != nil {
		return "", nil, false, errNode
	}
	return id.ID, cert, status == "revoked" || certRevoked, nil
}

func decode(r *http.Request, v any) error {
	dec := json.NewDecoder(io.LimitReader(r.Body, 1<<20))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return httpx.Errorf(http.StatusBadRequest, "invalid_body", "request body is not valid for this endpoint")
	}
	return nil
}

func (s *Server) notify(ctx context.Context, nodeID string) {
	_, _ = s.Pool.Exec(ctx, "SELECT pg_notify('starlight_node', $1)", nodeID)
}

func (s *Server) subscribe(nodeID string) (<-chan struct{}, func()) {
	ch := make(chan struct{}, 1)
	s.waitMu.Lock()
	if s.waiters == nil {
		s.waiters = map[string][]chan struct{}{}
	}
	s.waiters[nodeID] = append(s.waiters[nodeID], ch)
	s.waitMu.Unlock()
	return ch, func() {
		s.waitMu.Lock()
		defer s.waitMu.Unlock()
		list := s.waiters[nodeID]
		for i, c := range list {
			if c == ch {
				s.waiters[nodeID] = append(list[:i], list[i+1:]...)
				break
			}
		}
		if len(s.waiters[nodeID]) == 0 {
			delete(s.waiters, nodeID)
		}
	}
}

func (s *Server) wake(nodeID string) {
	s.waitMu.Lock()
	defer s.waitMu.Unlock()
	for _, ch := range s.waiters[nodeID] {
		select {
		case ch <- struct{}{}:
		default:
		}
	}
}

// Listen wakes long-polling nodes when their desired state changes, until ctx ends.
func (s *Server) Listen(ctx context.Context) {
	backoff := time.Second
	for ctx.Err() == nil {
		conn, err := s.Pool.Acquire(ctx)
		if err == nil {
			if _, err = conn.Exec(ctx, "LISTEN starlight_node"); err == nil {
				backoff = time.Second
				for {
					n, err := conn.Conn().WaitForNotification(ctx)
					if err != nil {
						break
					}
					s.wake(strings.TrimSpace(n.Payload))
				}
			}
			conn.Conn().Close(context.Background())
			conn.Release()
		}
		if ctx.Err() != nil {
			return
		}
		time.Sleep(backoff)
		backoff = min(backoff*2, 30*time.Second)
	}
}
